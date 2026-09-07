"""
Read knee MRI reports with a hosted model, as a second reader where the rules go silent.

Relationship to the other two extractors
----------------------------------------
`report_labeler` is a lexicon: fast, free, and strong on the vocabulary it knows. Its
failure mode is silence, and silence is measurable without labels - for each (report,
finding) pair, did any rule match? That measurement says the gaps are concentrated in a
few languages rather than spread evenly, which is what makes a second reader worth
buying: it is only asked to read the pairs the rules left blank.

`llm_labeler` runs local weights and answers the same question at roughly eighteen
seconds a report. That is affordable while a prompt is being tuned and unaffordable
across the corpus several times over.

This module is the same prompt against a hosted model. Three things change:

* **The output shape is enforced, not parsed.** The local path asks for twelve numbered
  lines and reads them with a tolerant regex, because a quantised model asked for JSON
  will sometimes emit prose. Here the response is constrained to a schema, so a reply
  that does not fit the twelve-finding shape cannot be returned at all. There is no
  parser to be lenient with, and therefore no silent mis-parse.

* **The instructions are hoisted out of the per-report text.** Caching matches on a
  prefix, so the half of the prompt that never changes - the task and the twelve
  definitions - is sent as a system block and marked cacheable, and the report is the
  only thing that varies. Across thousands of reports the fixed half is charged once
  per cache lifetime instead of once per report.

* **Requests are submitted in bulk.** Nothing here is latency-sensitive; this runs
  offline and its output is a CSV. Bulk submission halves the price for the same work.

The result is a per-study record with the same fields the local path writes, so the two
are interchangeable downstream and can be compared against the annotated studies on
equal terms before either is trusted.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from llm_labeler import FINDING_SPEC, TARGETS, to_scores

# Order is load-bearing: these keys index the same twelve findings as TARGETS, and the
# scoring code reads them positionally.
KEYS = [
    "acl", "mcl", "medial_meniscus", "lateral_meniscus",
    "medial_oa", "lateral_oa", "pf_oa", "effusion",
    "synovitis", "bakers", "contusion", "fracture",
]

SYSTEM = """You are an expert musculoskeletal radiologist reading a knee MRI report.
Reports may be written in any language, including Greek, Bulgarian, Turkish, Croatian,
Dutch, German, Spanish, French or English. Read each one in its original language.

For each of the twelve findings below, state what THE REPORT IN FRONT OF YOU says.

{spec}
verdict:
  YES  the report asserts the finding is present
  NO   the report denies it, or describes that structure as normal, intact or preserved
  UNK  the report does not mention it at all, or is explicitly uncertain

severity (0 whenever the verdict is not YES):
  1  minimal, trace, small, mild, slight, grade 1
  2  stated without qualification, moderate
  3  marked, large, severe, extensive, complete, full-thickness, grade 3-4

Rules you must follow:
- Judge only what the report states. Do not infer a finding from a related one.
- A structure described as normal or intact is NO, not UNK.
- A finding the report never names is UNK, not NO.
- Medial and lateral are distinct. If the report names one, the other is not thereby
  answered.
""".format(spec=FINDING_SPEC)

# A schema, not a format request. The response is constrained to this shape, so the
# twelve keys are always present and the verdict is always one of three strings; there
# is no reply that parses into a partial answer.
_FINDING = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO", "UNK"]},
        "severity": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["verdict", "severity"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {k: {"$ref": "#/$defs/finding"} for k in KEYS},
    "required": KEYS,
    "additionalProperties": False,
    "$defs": {"finding": _FINDING},
}


def _key(explicit=None):
    """Resolve the credential without ever putting it on a command line."""
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    p = Path.home() / ".config" / "rsna_llm_key"
    if p.is_file():
        return p.read_text().strip()
    raise RuntimeError(
        "No API key. Put it in ~/.config/rsna_llm_key (chmod 600) or set ANTHROPIC_API_KEY."
    )


def _verdicts(obj):
    """Schema-validated object -> the (verdict, severity) pairs the scorer expects."""
    out = []
    for k in KEYS:
        f = obj[k]
        v = f["verdict"]
        out.append((v, f["severity"] if v == "YES" else 0))
    return out


def _record(uid, obj, secs, usage=None):
    verdicts = _verdicts(obj)
    score, conf = to_scores(verdicts)
    rec = {
        "uid": uid,
        "verdict": [v for v, _ in verdicts],
        "severity": [s for _, s in verdicts],
        "score": score,
        "conf": conf,
        "secs": round(secs, 2),
        "raw": json.dumps(obj, ensure_ascii=False),
    }
    if usage:
        rec["usage"] = usage
    return rec


class ApiLabeler:
    """One report at a time. Use this to tune the prompt; use `submit_batch` to run it.

    `effort` is the cost lever. Twelve bounded judgements over a page of text is a short,
    scoped task, so the low setting is the starting point rather than a compromise; raise
    it only if the annotated studies say the reading is shallow.
    """

    def __init__(self, model="claude-opus-5", effort="low", api_key=None,
                 cache_path=None, max_tokens=2000):
        import anthropic
        self.client = anthropic.Anthropic(api_key=_key(api_key))
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["uid"]] = d
        self.n_new = 0
        # Reports are independent, so the obvious way to run a few hundred is a thread
        # pool. Only the shared cache needs guarding; the client is already safe.
        self._lock = threading.Lock()

    def _store(self, rec):
        with self._lock:
            self.cache[rec["uid"]] = rec
            self.n_new += 1
            if self.cache_path:
                with self.cache_path.open("a") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _params(self, report):
        """The request body, shared by the single-call and bulk paths.

        The system block carries everything that does not vary and is marked cacheable;
        the user turn carries only the report. Any content placed above a changing byte
        stops being reusable, so the ordering here is the whole point.
        """
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": report.strip()[:12000]}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
        )

    def label(self, uid, report):
        if uid in self.cache:
            return self.cache[uid]
        t = time.time()
        resp = self.client.messages.create(**self._params(report))
        obj = json.loads(next(b.text for b in resp.content if b.type == "text"))
        u = resp.usage
        rec = _record(uid, obj, time.time() - t, {
            "in": u.input_tokens, "out": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0),
            "cache_write": getattr(u, "cache_creation_input_tokens", 0),
        })
        self._store(rec)
        return rec

    def label_many(self, items, workers=8, progress=None):
        """items: [(uid, report), ...] -> records, in the order given."""
        from concurrent.futures import ThreadPoolExecutor
        done = [0]

        def one(it):
            r = self.label(*it)
            done[0] += 1
            if progress and done[0] % progress == 0:
                print(f"  {done[0]}/{len(items)}", flush=True)
            return r

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, items))

    def _map_path(self, batch_id):
        base = self.cache_path or Path("batch")
        return base.with_suffix(f".{batch_id}.map.json")

    def submit_batch(self, items):
        """items: [(uid, report), ...] -> batch id. Skips anything already cached.

        A study UID contains dots and can reach 64 characters, and `custom_id` accepts
        only [A-Za-z0-9_-] within 64. So requests are keyed by position and the mapping
        back to UIDs is written beside the cache - results can then be collected by a
        later process, which matters because a batch may take an hour.
        """
        todo = [(u, r) for u, r in items if u not in self.cache]
        if not todo:
            return None
        ids = {f"r{i:06d}": u for i, (u, _) in enumerate(todo)}
        batch = self.client.messages.batches.create(
            requests=[{"custom_id": f"r{i:06d}", "params": self._params(r)}
                      for i, (_, r) in enumerate(todo)]
        )
        self._map_path(batch.id).write_text(json.dumps(ids))
        return batch.id

    def collect_batch(self, batch_id, poll_s=60, timeout_s=7200):
        """Block until the batch ends, then fold its results into the cache."""
        t0 = time.time()
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            if b.processing_status == "ended":
                break
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"{batch_id} still {b.processing_status}")
            print(f"  {b.processing_status}  {b.request_counts}", flush=True)
            time.sleep(poll_s)

        ids = json.loads(self._map_path(batch_id).read_text())
        ok = err = 0
        # Results come back in arbitrary order; custom_id is the only safe key.
        for res in self.client.messages.batches.results(batch_id):
            if res.result.type != "succeeded":
                err += 1
                continue
            msg = res.result.message
            obj = json.loads(next(x.text for x in msg.content if x.type == "text"))
            u = msg.usage
            rec = _record(ids[res.custom_id], obj, 0.0, {
                "in": u.input_tokens, "out": u.output_tokens,
                "cache_read": getattr(u, "cache_read_input_tokens", 0),
                "cache_write": getattr(u, "cache_creation_input_tokens", 0),
            })
            self._store(rec)
            ok += 1
        return ok, err

    def cost(self, rate_in=5.0, rate_out=25.0):
        """Dollars spent so far, from the usage the API reported.

        Cache reads are charged at a tenth of the input rate and writes at 1.25x, so the
        fixed half of the prompt is only expensive the first time it is sent.
        """
        ti = to = tr = tw = 0
        for r in self.cache.values():
            u = r.get("usage") or {}
            ti += u.get("in", 0); to += u.get("out", 0)
            tr += u.get("cache_read", 0); tw += u.get("cache_write", 0)
        d = (ti + tw * 1.25 + tr * 0.1) / 1e6 * rate_in + to / 1e6 * rate_out
        return {"input": ti, "output": to, "cache_read": tr, "cache_write": tw,
                "usd_full_rate": round(d, 4), "usd_batch_rate": round(d / 2, 4)}


def count_prefix_tokens(model="claude-opus-5", api_key=None):
    """How large is the cacheable half? Below the model's minimum it silently won't cache."""
    import anthropic
    c = anthropic.Anthropic(api_key=_key(api_key))
    r = c.messages.count_tokens(
        model=model,
        system=[{"type": "text", "text": SYSTEM}],
        messages=[{"role": "user", "content": "x"}],
    )
    return r.input_tokens


def frame(records):
    """Same column layout the other two extractors emit."""
    import pandas as pd
    rows = []
    for r in records:
        row = {"StudyInstanceUID": r["uid"]}
        for j, t in enumerate(TARGETS):
            row[t] = r["score"][j]
            row[t + "__conf"] = r["conf"][j]
            row[t + "__verdict"] = r["verdict"][j]
        rows.append(row)
    return pd.DataFrame(rows)
