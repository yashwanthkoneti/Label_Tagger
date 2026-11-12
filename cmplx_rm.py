"""
run_monitor
===========

Batch scoring of policy-RAG interactions from a CSV, writing a CSV.

Edit the CONFIG block below and run:

    python run_monitor.py

No arguments, no CLI. Everything is a constant.

Input columns expected (rename them in COLUMNS if yours differ):
    transaction_id, request, response, document_chunks,
    faithfulness_score, request_created_on

`document_chunks` is parsed defensively. All of these work:
    - JSON list of objects:  [{"text": "...", "score": 0.81,
                               "doc_id": "leave-in",
                               "metadata": {"country": "IN"}}, ...]
    - JSON list of strings:  ["chunk one", "chunk two"]
    - a Python repr of either of the above (single quotes)
    - plain text separated by a blank line, "|||" or "---"

Anything the parser cannot find degrades safely rather than guessing:
    - no per-chunk scores  -> THIN_COVERAGE / DIFFUSE_MATCH cannot fire
                              (the baseline has zero variance, so z = 0)
    - no chunk metadata    -> BRANCH_UNRESOLVED / STALE_SOURCE / SENSITIVE
                              cannot fire
Both cases are reported in the summary so you know which signals were live.
Carrying `doc_id` and `metadata` through from your retriever is what unlocks
the signals faithfulness cannot see.

Two passes, deliberately:
    1. Build the baseline over the WHOLE file, so every row is flagged against
       the same distribution. A streaming warm-up would under-flag early rows
       purely because they arrived first, which is an artefact, not a finding.
    2. Score every row against that fixed baseline.
Drift is then measured separately, in time order: the most recent
CURRENT_WINDOW rows against everything before them.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from policy_monitor import (
    Baseline, Chunk, Interaction, Monitor, SIGNAL_NAMES,
    extract_signals, from_faithfulness, score_risk,
)

# ===========================================================================
# CONFIG - edit these
# ===========================================================================

INPUT_CSV = "interactions.csv"
OUTPUT_CSV = "risk_scores.csv"
SUMMARY_CSV = "risk_summary.csv"

# Map your CSV's column names onto the fields this script needs.
COLUMNS = {
    "transaction_id": "transaction_id",
    "request": "request",
    "response": "response",
    "document_chunks": "document_chunks",
    "faithfulness_score": "faithfulness_score",
    "request_created_on": "request_created_on",
    # Optional. A JSON object of what you already know about the asker,
    # e.g. {"country": "IN", "contract_type": "full_time"}. If present, those
    # attributes resolve branches instead of being flagged as unknown.
    "user_attributes": "user_attributes",
}

TOP_K = 5              # chunks considered per interaction
CURRENT_WINDOW = 200   # most-recent rows compared against the rest for drift
ALERT_SIGMA = 2.0      # standard errors before a drift alert fires
FLAG_SIGMA = 1.5       # standard deviations before a per-row flag fires
PREVIEW_ROWS = 3       # print how the first N rows parsed; set 0 to silence

# ===========================================================================


csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

_TEXT_KEYS = ("text", "content", "chunk", "page_content", "document", "body", "passage")
_SCORE_KEYS = ("score", "similarity", "relevance_score", "relevance", "rerank_score", "_score")
_DISTANCE_KEYS = ("distance", "_distance", "l2", "cosine_distance")
_DOCID_KEYS = ("doc_id", "document_id", "source", "id", "file_name", "filename", "title", "uri")
_META_KEYS = ("metadata", "meta", "attributes", "tags")

_FILE_NAME = re.compile(r"\*\*\s*File\s*Name\s*:?\s*\*\*[ \t]*(.*)", re.IGNORECASE)
_DOC_TITLE = re.compile(r"\*\*\s*Document\s*Title\s*:?\s*\*\*[ \t]*(.*)", re.IGNORECASE)

_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
)


class ParseStats:
    """What the parser could and could not find. Reported, never guessed around."""

    def __init__(self) -> None:
        self.rows = 0
        self.chunk_parse_failures = 0
        self.rows_with_scores = 0
        self.rows_with_metadata = 0
        self.rows_with_faithfulness = 0
        self.distance_converted = 0
        self.unparsed_dates = 0
        self.empty_chunks = 0
        self.faithfulness_rescaled = False
        self.labelled_text_rows = 0


def _parse_date(raw: str, stats: ParseStats) -> Optional[datetime]:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    stats.unparsed_dates += 1
    return None


def _first(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _coerce_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _parse_chunks(raw: str, stats: ParseStats) -> List[Chunk]:
    """Parse the document_chunks cell into Chunk objects, whatever shape it is."""
    raw = (raw or "").strip()
    if not raw:
        stats.empty_chunks += 1
        return []

    parsed: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
            break
        except Exception:
            continue

    if parsed is None:
        # Not JSON. Try the labelled plain-text form first:
        #     **File Name:** leave_policy_in.pdf
        #     **Document Title:** India Leave Policy
        #     <body, possibly containing pipe tables>
        marks = list(_FILE_NAME.finditer(raw))
        if marks:
            out: List[Chunk] = []
            for i, m in enumerate(marks):
                end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
                block = raw[m.start():end]
                fname = (m.group(1) or "").strip()
                tm = _DOC_TITLE.search(block)
                title = (tm.group(1) or "").strip() if tm else ""
                # Body is everything after the labelled headers.
                body = _DOC_TITLE.sub("", _FILE_NAME.sub("", block)).strip()
                meta = {}
                if fname:
                    meta["file_name"] = fname
                if title:
                    meta["document_title"] = title
                out.append(Chunk(
                    text=body,
                    score=0.0,
                    doc_id=fname or title or f"chunk_{i}",
                    metadata=meta,
                ))
            stats.labelled_text_rows += 1
            return out

        # Last resort: split on blank lines or an explicit rule. NOTE: '|||' is
        # deliberately NOT a separator here - pipe-delimited tables in the body
        # would be shredded by it.
        for sep in ("\n\n\n", "\n---\n", "\n___\n"):
            if sep in raw:
                parts = [p.strip() for p in raw.split(sep) if p.strip()]
                break
        else:
            parts = [raw]
        return [Chunk(text=p, score=0.0, doc_id=f"chunk_{i}", metadata={})
                for i, p in enumerate(parts)]

    if isinstance(parsed, dict):
        # A single chunk, or a wrapper like {"chunks": [...]}
        inner = _first(parsed, ("chunks", "documents", "results", "contexts"))
        parsed = inner if isinstance(inner, list) else [parsed]
    if not isinstance(parsed, list):
        stats.chunk_parse_failures += 1
        return [Chunk(text=str(parsed), score=0.0, doc_id="chunk_0", metadata={})]

    chunks: List[Chunk] = []
    for i, item in enumerate(parsed):
        if isinstance(item, str):
            chunks.append(Chunk(text=item, score=0.0, doc_id=f"chunk_{i}", metadata={}))
            continue
        if not isinstance(item, dict):
            chunks.append(Chunk(text=str(item), score=0.0, doc_id=f"chunk_{i}", metadata={}))
            continue

        text = _first(item, _TEXT_KEYS)
        score = _coerce_float(_first(item, _SCORE_KEYS))
        if score is None:
            dist = _coerce_float(_first(item, _DISTANCE_KEYS))
            if dist is not None:
                # Lower distance is better; map to a similarity so the margin
                # and decay arithmetic stays meaningful. Monotone, so ordering
                # and z-scores are unaffected.
                score = 1.0 / (1.0 + max(0.0, dist))
                stats.distance_converted += 1
        doc_id = _first(item, _DOCID_KEYS)
        meta = _first(item, _META_KEYS)
        if not isinstance(meta, dict):
            # Treat any leftover scalar keys as metadata.
            known = set(_TEXT_KEYS + _SCORE_KEYS + _DISTANCE_KEYS + _DOCID_KEYS + _META_KEYS)
            meta = {k: str(v) for k, v in item.items()
                    if k not in known and isinstance(v, (str, int, float, bool))}
        else:
            meta = {str(k): str(v) for k, v in meta.items() if v not in (None, "")}

        chunks.append(Chunk(
            text=str(text) if text is not None else "",
            score=score if score is not None else 0.0,
            doc_id=str(doc_id) if doc_id is not None else f"chunk_{i}",
            metadata=meta,
        ))
    return chunks


def _parse_user_attributes(raw: str) -> Dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, dict):
                return {str(k): str(val) for k, val in v.items() if val not in (None, "")}
        except Exception:
            continue
    return {}


def load_rows(path: str) -> Tuple[List[Interaction], List[Optional[float]],
                                  List[Optional[datetime]], ParseStats]:
    """Read the CSV into Interactions, keeping faithfulness and timestamps aside."""
    stats = ParseStats()
    interactions: List[Interaction] = []
    faithfulness: List[Optional[float]] = []
    created: List[Optional[datetime]] = []

    C = COLUMNS
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [v for k, v in C.items()
                   if k != "user_attributes" and v not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"{path}: missing expected column(s): {', '.join(missing)}\n"
                f"found: {', '.join(reader.fieldnames or [])}\n"
                f"Fix the COLUMNS map at the top of this file."
            )

        for row in reader:
            stats.rows += 1
            chunks = _parse_chunks(row.get(C["document_chunks"], ""), stats)
            if any(c.score != 0.0 for c in chunks):
                stats.rows_with_scores += 1
            if any(c.metadata for c in chunks):
                stats.rows_with_metadata += 1

            f = _coerce_float(row.get(C["faithfulness_score"]))
            if f is not None:
                stats.rows_with_faithfulness += 1
            faithfulness.append(f)

            created.append(_parse_date(row.get(C["request_created_on"], ""), stats))
            interactions.append(Interaction(
                interaction_id=str(row.get(C["transaction_id"], "") or f"row_{stats.rows}"),
                query=row.get(C["request"], "") or "",
                answer=row.get(C["response"], "") or "",
                chunks=chunks,
                user_attributes=_parse_user_attributes(row.get(C.get("user_attributes", ""), "")),
            ))

    # Faithfulness reported 0-100 rather than 0-1.
    present = [f for f in faithfulness if f is not None]
    if present and max(present) > 1.0:
        stats.faithfulness_rescaled = True
        faithfulness = [None if f is None else f / 100.0 for f in faithfulness]

    return interactions, faithfulness, created, stats


def main() -> int:
    interactions, faithfulness, created, stats = load_rows(INPUT_CSV)
    if not interactions:
        raise SystemExit(f"{INPUT_CSV}: no rows")

    # Chronological order. You said sorting is on your side; doing it here too
    # is cheap, and drift is meaningless without it.
    order = sorted(range(len(interactions)),
                   key=lambda i: (created[i] is None,
                                  created[i] or datetime.min))
    interactions = [interactions[i] for i in order]
    faithfulness = [faithfulness[i] for i in order]
    created = [created[i] for i in order]

    judges = [from_faithfulness(lambda _ix, v=f: v) if f is not None else None
              for f in faithfulness]

    # Pass 1: baseline over the whole file.
    baseline = Baseline(window=len(interactions) + 1, min_n=1)
    all_signals = []
    for ix, judge in zip(interactions, judges):
        sig = extract_signals(ix, top_k=TOP_K)
        if judge is not None:
            try:
                sig.values["unsupported_ratio"] = judge(ix)
            except Exception:
                pass
        all_signals.append(sig)
        baseline.add(sig)

    # Pass 2: score every row against that fixed baseline.
    import policy_monitor
    policy_monitor.FLAG_SIGMA = FLAG_SIGMA

    rows_out = []
    risks = []
    for ix, judge, ts in zip(interactions, judges, created):
        risk = score_risk(ix, baseline=baseline, judge=judge, top_k=TOP_K)
        risks.append(risk)
        v = risk.signals.values
        rows_out.append({
            "transaction_id": ix.interaction_id,
            "request_created_on": ts.isoformat() if ts else "",
            "risk_score": round(risk.score, 1),
            "flags": "|".join(risk.flags),
            "n_flags": len(risk.flags),
            "clarifying_question": risk.clarifying_question or "",
            "branch_dims": json.dumps(risk.signals.branch_dims) if risk.signals.branch_dims else "",
            "notes": " ; ".join(risk.notes),
            **{f"sig_{k}": round(v.get(k, 0.0), 4) for k in SIGNAL_NAMES},
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # ---- drift: most recent CURRENT_WINDOW rows vs everything before --------
    n = len(all_signals)
    cut = max(1, n - CURRENT_WINDOW)
    hist, curr = all_signals[:cut], all_signals[cut:]
    alerts = []
    if len(hist) >= 30 and len(curr) >= 30:
        for name in SIGNAL_NAMES:
            h = [s.values[name] for s in hist if name in s.values]
            c = [s.values[name] for s in curr if name in s.values]
            if len(h) < 30 or len(c) < 30:
                continue
            sd = statistics.pstdev(h)
            if sd <= 1e-9:
                continue
            mu = statistics.fmean(h)
            cm = statistics.fmean(c)
            sigma = (cm - mu) / (sd / math.sqrt(len(c)))
            if abs(sigma) >= ALERT_SIGMA:
                alerts.append((name, mu, cm, sigma))
        alerts.sort(key=lambda a: -abs(a[3]))

    # ---- summary ------------------------------------------------------------
    flag_counts: Dict[str, int] = {}
    for r in risks:
        for fl in set(r.flags):
            flag_counts[fl] = flag_counts.get(fl, 0) + 1

    summary: List[Dict[str, Any]] = []

    def add(kind, metric, value, detail=""):
        summary.append({"kind": kind, "metric": metric, "value": value, "detail": detail})

    add("coverage", "rows", stats.rows)
    add("coverage", "rows_with_chunk_scores", stats.rows_with_scores,
        "THIN_COVERAGE / DIFFUSE_MATCH need these")
    add("coverage", "rows_with_chunk_metadata", stats.rows_with_metadata,
        "BRANCH_UNRESOLVED / STALE_SOURCE / SENSITIVE need these")
    add("coverage", "rows_with_faithfulness", stats.rows_with_faithfulness)
    add("coverage", "rows_with_empty_chunks", stats.empty_chunks)
    add("coverage", "unparsed_dates", stats.unparsed_dates)
    add("coverage", "distance_converted_to_similarity", stats.distance_converted)
    add("coverage", "faithfulness_rescaled_from_0_100", int(stats.faithfulness_rescaled))
    add("coverage", "rows_parsed_as_labelled_text", stats.labelled_text_rows,
        "**File Name:** / **Document Title:** form")
    add("coverage", "chunks_per_row_mean",
        round(statistics.fmean([len(i.chunks) for i in interactions]), 2),
        "if this is 1.0 the chunk delimiter was not recognised")

    for fl, cnt in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
        add("flag_rate", fl, round(cnt / len(risks), 4), f"{cnt} of {len(risks)}")

    scores = [r.score for r in risks]
    add("risk", "mean", round(statistics.fmean(scores), 2))
    add("risk", "p50", round(statistics.median(scores), 2))
    add("risk", "p90", round(sorted(scores)[int(0.9 * (len(scores) - 1))], 2))
    add("risk", "clean_rate", round(sum(1 for r in risks if not r.flags) / len(risks), 4),
        "interactions with no flag at all")

    for name, mu, cm, sigma in alerts:
        add("drift_alert", name, round(cm, 4),
            f"baseline {mu:.4f} -> current {cm:.4f} ({sigma:+.1f} sigma, "
            f"last {len(curr)} rows)")
    if not alerts:
        add("drift_alert", "none", 0,
            f"no signal moved more than {ALERT_SIGMA} sigma"
            if len(hist) >= 30 and len(curr) >= 30
            else f"need >=30 rows each side; have {len(hist)} and {len(curr)}")

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "metric", "value", "detail"])
        w.writeheader()
        w.writerows(summary)

    # Parse preview: verify the chunk splitting before trusting any numbers.
    if PREVIEW_ROWS:
        print("=" * 72)
        print("PARSE PREVIEW - check this before trusting the output")
        print("=" * 72)
        for ix in interactions[:PREVIEW_ROWS]:
            print(f"\n[{ix.interaction_id}] parsed {len(ix.chunks)} chunk(s)")
            for c in ix.chunks[:4]:
                t, r, o = __import__("policy_monitor")._table_integrity(c.text)
                print(f"   doc_id={c.doc_id!r} score={c.score} "
                      f"meta={c.metadata} tables={t} ragged={r} orphan={o}")
                print(f"     text[:90]={c.text[:90]!r}")
        print("\n" + "=" * 72 + "\n")

    print(f"read    {stats.rows} rows from {INPUT_CSV}")
    print(f"wrote   {OUTPUT_CSV}  ({len(rows_out)} rows)")
    print(f"wrote   {SUMMARY_CSV} ({len(summary)} rows)")
    if stats.rows_with_scores == 0:
        print("NOTE    no chunk scores found -> retrieval-quality flags disabled")
    if stats.rows_with_metadata == 0:
        print("NOTE    no chunk metadata found -> branch/staleness flags disabled")
    if alerts:
        print(f"ALERTS  {len(alerts)}: " + ", ".join(a[0] for a in alerts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
