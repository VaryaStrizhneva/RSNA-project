"""What each pathology is, and how it is read.

Textbook radiology, written down so the atlas is not twelve folders of pictures with
no idea what to look for. It is **not** expert knowledge and not a clinical reference:
check it against a real one before acting on it.

Kept apart from the rendering so that adding a pathology is writing one dictionary,
not editing a page generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    """One study shown in the atlas."""

    uid_tail: str                 # last 11 characters, enough to identify it here
    positive: bool
    quote: str                    # the report's own words about this finding
    language: str = "English"
    translation: str = ""
    why: str = ""                 # why this case was chosen
    claude: str = ""              # my own reading. UNVERIFIED. See the page's warning.
    claude_slices: tuple = ()     # slices I would look at, if any


@dataclass
class Pathology:
    """Everything the page for one finding needs, apart from pixels."""

    key: str
    name: str
    one_line: str
    what_it_is: str
    how_it_is_read: str
    what_to_look_for: list[str]
    pitfalls: list[str]
    sequences: str
    planes: tuple[str, ...]
    preferred_weight: tuple[str, ...]
    numbers: dict[str, str] = field(default_factory=dict)
    cases: list[Case] = field(default_factory=list)


LATERAL_MENISCUS = Pathology(
    key="lateral_meniscus",
    name="Lateral Meniscus",
    one_line="A tear of the outer meniscus — a thin bright line inside a structure "
             "that should be uniformly black, and only a tear if it reaches a surface.",

    what_it_is="""
Two wedges of <b>fibrocartilage</b> sit between the femur and the tibia, one on each
side of the knee. In cross-section each is a triangle with its point facing inward;
seen from above, the medial one is a <b>C</b> and the lateral one is almost a closed
<b>O</b>, covering more of the tibial plateau.

They spread load, absorb shock and deepen the socket the femur sits in. Fibrocartilage
is dense, highly ordered collagen with almost no free water — which is why a normal
meniscus is <b>uniformly black on every MRI sequence</b>. That uniform blackness is the
whole basis of the diagnosis: anything bright inside it is abnormal.

Each meniscus is described in three parts, front to back: the <b>anterior horn</b>, the
<b>body</b>, and the <b>posterior horn</b>. Reports name the part, which is why the
report text beside each case below is worth reading first.
""",

    how_it_is_read="""
The entire diagnosis rests on one criterion:

<blockquote><b>Bright signal inside the black meniscus that reaches an articular
surface.</b></blockquote>

That is the standard three-grade scheme, and grades 2 and 3 can look nearly identical —
the difference is whether a thin line <i>touches an edge</i>:

<table class="grades">
<tr><th>grade</th><th>what is seen</th><th>what it means</th></tr>
<tr><td>1</td><td>globular bright spot inside, not touching a surface</td>
    <td>degeneration — <b>not a tear</b></td></tr>
<tr><td>2</td><td>linear bright signal inside, not touching a surface</td>
    <td>degeneration — <b>not a tear</b></td></tr>
<tr><td>3</td><td>bright signal <b>reaching the surface</b></td><td><b>tear</b></td></tr>
</table>

A second tool is the <b>bow-tie sign</b>. On sagittal slices through the periphery of
the meniscus the anterior and posterior horns join into one continuous band — a bow
tie. You normally see two or three of those in a row. Fewer suggests a displaced
bucket-handle tear, where part of the meniscus has flipped into the middle of the joint.

A surgeon adds what MRI cannot show: the mechanism of injury, joint-line tenderness,
McMurray's test, and whether the knee locks or catches. Arthroscopy remains the ground
truth, and MRI sensitivity is reported as meaningfully <b>lower for lateral than for
medial</b> tears.
""",

    what_to_look_for=[
        "Find the joint line first — the dark femoral condyle above, the tibial plateau "
        "below, and between them at front and back two dark triangles pointing inward. "
        "Those triangles are the meniscus.",
        "Scroll toward the <b>lateral</b> end of the stack. Each case below tells you "
        "which end that is, because it is not the same end on a left and a right knee.",
        "The triangles should be <b>solid black</b>. Look for any grey or bright line "
        "inside them.",
        "If you find one, ask the only question that matters: <b>does it reach an "
        "edge?</b> An internal blur is degeneration; a line running out to the surface "
        "is a tear.",
        "Check two or three consecutive slices. A real tear persists; noise and partial "
        "volume do not.",
        "Confirm on the coronal series when one is shown — standard practice, and it is "
        "how a false positive gets caught.",
    ],

    pitfalls=[
        "<b>The popliteal hiatus.</b> The popliteus tendon passes <i>through a gap</i> "
        "in the lateral meniscus's peripheral attachment at the back corner. On MRI "
        "that gap is a bright cleft in exactly the place a posterior-horn tear would "
        "be. It is the classic lateral-meniscus false positive.",
        "<b>The transverse ligament</b>, joining the two anterior horns, can mimic an "
        "anterior-horn tear where it inserts.",
        "<b>Discoid variant.</b> In a few percent of people the lateral meniscus is a "
        "disc rather than a C. Its normal appearance is simply different — and one of "
        "the cases below has exactly this.",
        "<b>Magic angle.</b> Collagen oriented near 55° to the main field can look "
        "bright on short-TE sequences without any tear being present.",
        "<b>More mobile than the medial side</b>, less firmly anchored to the capsule, "
        "so its position varies more between patients.",
    ],

    sequences="""
Short-TE sequences — <b>proton density (PD)</b> and T1 — show signal <i>inside</i> the
meniscus best, which is what the grading depends on. Fat-suppressed T2 is better for
surrounding fluid and bone oedema but worse for the meniscus itself. Every case below
is shown on sagittal PD where one exists, because that is the sequence the diagnosis is
actually made on.
""",

    planes=("Sagittal", "Coronal"),
    preferred_weight=("PD", "T1"),

    numbers={
        "How well the reports capture it":
            "Extraction AUC <b>0.841</b> against the 58 expert studies, with only "
            "<b>10% UNK</b> — the radiologists do write it down. This is not an "
            "information failure.",
        "How well our model detects it":
            "Holdout AUC <b>0.792</b> — our worst target of twelve.",
        "How much of that is actually vision":
            "Predicting this target from the other eleven labels alone, with no image, "
            "gives 0.757. Our model adds <b>+0.035</b>. It is very nearly not seeing "
            "this pathology at all.",
        "Why, in one number":
            "A tear line is about 1.5 mm. One DINOv2 patch token covers <b>5.4 mm</b> "
            "at our resolution. The finding is roughly a quarter of the encoder's "
            "smallest unit of position, and the slot feature then averages over 576 of "
            "them.",
    },

    cases=[
        Case(
            uid_tail="48839028269", positive=True, language="English",
            quote="Horizontal tear at anterior horn of the lateral meniscus is noted.",
            why="The report names the part and the tear type, so you know exactly where "
                "to look. A horizontal tear is one of the harder patterns — a line "
                "running parallel to the tibial plateau, easy to mistake for the normal "
                "dark band between meniscus and cartilage.",
            claude="""
I can locate the anterior horn: it is the dark wedge at the <b>front</b> of the joint
line, and on this right knee the lateral compartment is the <b>high</b> slice numbers.
Across slices 19&ndash;22 that wedge does not look solid black to me &mdash; it is
irregular and striated, with bright material tracking into it from in front. That is
<i>consistent with</i> the reported horizontal tear.
<p>But I cannot confirm the criterion that actually decides it: whether the bright
signal <b>reaches the articular surface</b>. At this magnification I would be guessing,
and a grade&nbsp;2 degeneration would look much the same to me. Treat this as a pointer
to where to look, not as a finding.</p>
<p>Separately, the very bright focus at slices 24&ndash;25 is <b>not</b> the meniscus.
The report describes an osteochondral defect and subchondral bone oedema at the lateral
patellar facet, and that is what I believe it is &mdash; a good illustration of how the
most eye-catching thing on the image is often not the thing being scored.</p>""",
            claude_slices=(19, 20, 21, 22)),

        Case(
            uid_tail="67488111973", positive=True, language="English",
            quote="The lateral compartment, there is longitudinal vertical oblique tear "
                  "at the posterior horn of the lateral meniscus.",
            why="A different part (posterior horn) and a different tear pattern "
                "(longitudinal vertical) from the first case. It is also a <b>left</b> "
                "knee, so the lateral compartment sits at the <i>opposite end</i> of the "
                "stack &mdash; which is the single most confusing thing about reading "
                "these series, and the reason each viewer below is labelled individually.",
            claude="""
This is a left knee, so slice&nbsp;0 is <b>lateral</b> and the compartment of interest is
slices 3&ndash;8 &mdash; the opposite end from the previous case, on an otherwise
identical-looking acquisition.
<p>I can find the joint line and the dark meniscal triangles there, but I could not
confidently identify the posterior-horn tear. A longitudinal vertical tear runs
perpendicular to the tibial plateau and is often only a hairline; I do not trust my own
eye on it here. I have flagged the slices where the lateral compartment is in view, not
slices where I claim to see the tear.</p>""",
            claude_slices=(3, 4, 5, 6, 7, 8)),

        Case(
            uid_tail="56139890254", positive=True, language="English",
            quote="The lateral compartment, there is complex tear with buckethandle "
                  "tear at the anterior horn, body and posterior horn of the lateral "
                  "meniscus.",
            why="The gross end of the spectrum. A bucket-handle tear displaces a "
                "fragment into the middle of the joint, so the meniscus is abnormal "
                "along its whole length &mdash; if any lateral meniscus finding were "
                "going to be visible at low resolution, it would be this one.",
            claude="""
There is an obvious large <b>joint effusion</b> &mdash; bright fluid filling the
compartment through slices 16&ndash;23 &mdash; and the joint looks globally abnormal to
me. What I could <b>not</b> do is identify the displaced bucket-handle fragment itself,
which is the actual finding.
<p>I think that is worth stating plainly, because it is the most useful thing in this
page. This is the most severe lateral meniscus tear among the five cases, I was told in
advance what it was and where, I am looking at full-resolution images with no crop and
no downsampling &mdash; and I still cannot confirm it. The model is working from
twelve slices at 0.39&nbsp;mm/px through a 5.4&nbsp;mm token grid.</p>""",
            claude_slices=(16, 17, 18, 19, 20, 21, 22, 23)),

        Case(
            uid_tail="80836752052", positive=False, language="English",
            quote="Lateral meniscus: Degenerative signal throughout the lateral "
                  "meniscus without surfacing tear.",
            why="<b>The most important case on this page.</b> This meniscus <i>does</i> "
                "contain bright signal &mdash; it is degenerate throughout &mdash; and it "
                "is still scored negative, because none of that signal reaches a surface. "
                "This is grade&nbsp;2 versus grade&nbsp;3 made explicit by the "
                "radiologist. If you only learn one contrast from this page, learn this "
                "one: bright inside the meniscus is not the finding; bright <i>touching "
                "an edge</i> is.",
            claude="""
Note first that this series is <b>PD without fat suppression</b>, unlike the others.
Fat stays bright, so the whole image looks paler &mdash; and this is the classic
sequence for reading menisci, because it shows signal <i>inside</i> the fibrocartilage
best.
<p>Right knee, so the lateral compartment is the high numbers, roughly slices
15&ndash;20. The meniscal triangles there look grey and inhomogeneous to me rather than
solid black, which matches the reported degeneration. Whether anything reaches the
surface I genuinely cannot tell &mdash; and neither, apparently, could the original
radiologist without saying so explicitly, which is why they wrote the words "without
surfacing tear" rather than just describing the signal.</p>""",
            claude_slices=(15, 16, 17, 18, 19, 20)),

        Case(
            uid_tail="03448882465", positive=False, language="English",
            quote="Normal medial and lateral menisci.",
            why="The clean baseline. Scroll this one first and learn what a solid black "
                "meniscal triangle looks like, then go back to the others.",
            claude="""
Right knee, lateral compartment at the high slice numbers. The meniscal triangles here
look uniformly dark to me, with crisp edges, which is what the report says. I would
call this normal &mdash; but note that I would also have called several of the positive
cases above normal, so my agreeing with the report here is weak evidence of anything.""",
            claude_slices=()),
    ],
)

PATHOLOGIES = {p.key: p for p in (LATERAL_MENISCUS,)}
