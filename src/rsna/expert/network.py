"""One pathology, one crop, one logit.

Dedicated rather than a branch of the twelve-target model, which changes what it can be
compared against. The wide model gets **0.774 of its 0.844 from the other eleven labels
alone, with no pixels**; a model that sees only a 48 mm crop has none of that, so its
score cannot be read against the wide model's 0.792 on this target. The controlled
comparison is two single-target models that differ only in their input — the wide view
against the crop — and that is what isolates the resolution hypothesis.

**A convolutional encoder, not the classifier's ViT.** A meniscal tear is about 1.5 mm.
At the ROI's 0.214 mm/px that is 7 pixels, against a DINOv2 patch of 14 — so the tear is
half a patch, and a patch embedding cannot represent anything smaller than its patch.
The crop would move the bottleneck without lifting it: 0.28 patches today, 0.5 in the
ROI. A convolutional stage at stride 4 sees the same tear over 1.7 cells.

The depth axis is consumed the way the wide pipeline already consumes its slots: three
neighbouring slices as the three channels of one window, windows slid along the stack
and pooled. Five slots give three windows.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import ExpertConfig


class Attention(nn.Module):
    """A weighted mean over whatever series a study actually has, once per target.

    Coverage is the reason this is not a plain mean: sagittal PD fat-suppressed is
    present on 81.3 % of studies and PD without it on 36.3 %, while at least one of the
    two covers 99.7 %. A study missing a series must contribute nothing from it rather
    than contribute a zero, which a mean over a fixed denominator would do.

    **One set of weights per target, not one shared.** The heads are already separate,
    so what two targets on one encoder actually compete over is this pooling: a single
    scorer must pick one ranking of the series for both of them, and they do not want
    the same one. A meniscal tear is signal inside the fibrocartilage, which the
    non-fat-suppressed proton density shows; osteoarthritis is marrow and bone, which
    the fat-suppressed and the T1 show. Measured, sharing the scorer cost the meniscus
    0.0084 (0.8334 against 0.8418 alone) and left osteoarthritis at 0.7866. Adding the
    coronal plane raises the series count from two to four or five, so the arbitration
    gets tighter, not looser.

    With one target this is the shared scorer: `Linear(dim, 1)`, same shape, same
    initialisation, same arithmetic. The single-target runs still reproduce.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.score = nn.Linear(dim, heads)

    def forward(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(batch, series, dim) and (batch, series) -> (batch, target, dim)."""

        w = self.score(feat)                                      # (b, series, target)
        gone = ~mask.bool() if mask.dtype == torch.bool else mask <= 0
        w = w.masked_fill(gone.unsqueeze(-1), float("-inf"))
        # A study with no series at all would give a row of -inf and a NaN softmax.
        empty = gone.all(dim=1)[:, None, None].expand_as(w)
        w = torch.where(empty, torch.zeros_like(w), w)
        return torch.einsum("bst,bsd->btd", w.softmax(dim=1), feat)


class ExpertNet(nn.Module):
    """(batch, series, slot, h, w) -> one logit per study per target.

    Several targets on one encoder, not one model each. They share a region of interest
    — the lateral compartment is one thing to look at — so they share the features that
    describe it, and the encoder sees twice the supervision for the same pixels.
    """

    def __init__(self, config: ExpertConfig, pretrained: bool = True):
        super().__init__()
        import timm

        self.config = config
        self.trunk = timm.create_model(config.encoder, pretrained=pretrained,
                                       num_classes=0, global_pool="avg")
        dim = self.trunk.num_features
        self.attend = Attention(dim, heads=len(config.targets))
        self.drop = nn.Dropout(config.dropout)
        # Not a Linear over one pooled vector: each target pools the series its own way,
        # so each reads its own vector. Weight (target, dim), applied row against row.
        self.head = nn.Linear(dim, len(config.targets))

        # Start at the corpus rate rather than at even odds: the head then learns the
        # deviation from the prior instead of first walking to it.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, config.prior_logit)

    def windows(self, imgs: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, series, slot, h, w) -> the windows, and which of them are real.

        Three neighbouring slices as three channels, one window per slot that can be a
        centre. Two rules, both of them about what to do at the edge of an acquisition.

        A window is real when **its centre slot holds an acquired slice**. Accepting one
        because *any* of its three slots is real would let a black channel through, and
        a black channel is the batch-normalisation problem one level down: that
        channel's statistics are taken over the batch too.

        Its outer channels are **clamped to the ends of the acquired run** rather than
        reaching into the padding. A slice at the edge of an acquisition has no
        neighbour, so it is its own — which is what `rsna.landmark.loop.to_input`
        already does for the same situation.
        """

        b, s, slots = mask.shape
        half = self.config.group // 2
        m = mask.int()
        first = m.argmax(dim=2)
        last = slots - 1 - m.flip(2).argmax(dim=2)

        wins, live = [], []
        for centre in range(half, slots - half):
            chans = []
            for offset in range(-half, half + 1):
                j = torch.clamp(torch.full_like(first, centre + offset),
                                min=first, max=last)
                chans.append(torch.gather(
                    imgs, 2, j[..., None, None, None].expand(b, s, 1, *imgs.shape[-2:])
                ).squeeze(2))
            wins.append(torch.stack(chans, dim=2))
            live.append(mask[:, :, centre])
        return torch.stack(wins, dim=2), torch.stack(live, dim=2)

    def encode(self, imgs: torch.Tensor, mask: torch.Tensor) -> tuple:
        """One ROI group -> (batch, series, dim) and (batch, series) presence.

        imgs: (batch, series, slot, h, w). mask: (batch, series, slot).

        **Only the windows that hold an acquired slice reach the encoder.** Four studies
        in five carry one of the two series and not the other, and their slots are
        padding, so a plain pass sends 44 % of a batch through the trunk as black
        images. That is not merely wasted work: a ResNet has twenty BatchNorm layers,
        whose statistics are taken over the whole batch, so the black half drags the
        per-channel mean and variance and every real image is normalised with them.
        Measured on a real batch, the features of the real images moved by **83 % in
        relative norm** depending on whether the padding was in it.

        The same masking fixes the pooling over windows: a study with three slices in
        five slots has one window that is entirely padding, and a plain mean counted it.
        """

        b, s = mask.shape[:2]
        w, live = self.windows(imgs, mask)          # (b, s, win, 3, h, w), (b, s, win)
        n_win = w.shape[2]

        flat = w.reshape(b * s * n_win, *w.shape[3:])
        keep = live.reshape(-1)
        feat = flat.new_zeros(b * s * n_win, self.trunk.num_features)
        if bool(keep.any()):
            feat[keep] = self.trunk(flat[keep])
        feat = feat.reshape(b, s, n_win, -1)

        count = live.sum(dim=2, keepdim=True).clamp(min=1)
        return (feat * live.unsqueeze(-1)).sum(dim=2) / count, live.any(dim=2)

    def forward(self, groups) -> torch.Tensor:
        """A list of (imgs, mask), one per ROI spec -> (batch, target).

        The specs do not share a tensor: a sagittal crop is 224x126 and a coronal one
        224x140, because 48x27 mm and 48x30 mm at the same millimetres per pixel are not
        the same number of rows. Running the trunk once per group and letting the
        features meet at the attention costs nothing and keeps each box free to be the
        right size for what it frames.
        """

        if torch.is_tensor(groups):
            groups = [(groups, None)]
        feats, masks = [], []
        for imgs, mask in groups:
            f, m = self.encode(imgs, mask)
            feats.append(f)
            masks.append(m)
        feat = torch.cat(feats, dim=1)
        mask = torch.cat(masks, dim=1)
        pooled = self.drop(self.attend(feat, mask))          # (batch, target, dim)
        return (torch.einsum("btd,td->bt", pooled, self.head.weight)
                + self.head.bias)
