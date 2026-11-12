"""
run_monitor
===========

Batch scoring of policy-RAG interactions from a pandas DataFrame.

Usage:

    import pandas as pd
    from run_monitor import run

    scores, summary = run(my_dataframe)

`scores` is one row per interaction, aligned to your input and carrying
`source_index` so you can join it straight back onto the original frame.
`summary` is long-format diagnostics: parse coverage, flag rates, risk
percentiles and drift alerts.

Edit the CONFIG block for column names and thresholds. Set WRITE_SCORES_CSV /
WRITE_SUMMARY_CSV to a path if you also want files; leave them None to stay
in memory.

Expected columns (rename in COLUMNS if yours differ):
    transaction_id, request, response, document_chunks,
    faithfulness_score, request_created_on
    user_attributes  -- optional

Type tolerance, because a DataFrame is messier than a CSV:
    - missing values arrive as NaN/None/NaT, not "" -- all coerced
    - `document_chunks` may already be a list or dict (not a string) if the
      frame came from JSON or an API response; used directly when so
    - `request_created_on` may already be Timestamp/datetime64 or a string
    - the index may be non-sequential or non-unique; never relied on for order

`document_chunks` as text is parsed in this order:
    1. already a list/dict        -> used as-is
    2. JSON or a Python repr      -> parsed
    3. labelled text             -> split on each `**File Name:**` marker,
                                     taking doc_id from the filename and
                                     document_title into metadata
    4. last resort               -> split on a blank-line run or a rule.
                                     NOTE '|||' is deliberately not a
                                     separator: pipe tables would be shredded.

Signals that cannot be computed are disabled rather than guessed at, and the
summary's `coverage` rows tell you which were live:
    no per-chunk scores -> THIN_COVERAGE / DIFFUSE_MATCH cannot fire
    no chunk metadata   -> BRANCH_UNRESOLVED falls back to filename variants;
                           STALE_SOURCE / SENSITIVE cannot fire

Two passes, deliberately: the baseline is built over the WHOLE frame so every
row is flagged against the same distribution. A streaming warm-up would
under-flag early rows purely because they arrived first, which is an artefact.
Drift is then measured in time order: the most recent CURRENT_WINDOW rows
against everything before them.
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from policy_monitor import (
    Baseline, Chunk, Interaction, SIGNAL_NAMES,
    _table_integrity, extract_signals, from_faithfulness, score_risk,
)

# ===========================================================================
# CONFIG - edit these
# ===========================================================================

COLUMNS = {
    "transaction_id": "transaction_id",
    "request": "request",
    "response": "response",
    "document_chunks": "document_chunks",
    "faithfulness_score": "faithfulness_score",
    "request_created_on": "request_created_on",
    # Optional. A dict or JSON string of what you already know about the asker,
    # e.g. {"country": "IN"}. Present attributes resolve branches instead of
    # being flagged as unknown. Drop the key entirely if you have no such column.
    "user_attributes": "user_attributes",
}

TOP_K = 5              # chunks considered per interaction
CURRENT_WINDOW = 200   # most-recent rows compared against the rest for drift
ALERT_SIGMA = 2.5      # standard errors before a drift alert fires
FLAG_SIGMA = 1.5       # standard deviations before a per-row flag fires
PREVIEW_ROWS = 3       # print how the first N rows parsed; 0 to silence
VERBOSE = True         # print the run summary to stdout

WRITE_SCORES_CSV: Optional[str] = None     # e.g. "risk_scores.csv"
WRITE_SUMMARY_CSV: Optional[str] = None    # e.g. "risk_summary.csv"

# ===========================================================================


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
    """What the parser found. Reported, never guessed around."""

    def __init__(self) -> None:
        self.rows = 0
        self.rows_with_scores = 0
        self.rows_with_metadata = 0
        self.rows_with_faithfulness = 0
        self.distance_converted = 0
        self.unparsed_dates = 0
        self.empty_chunks = 0
        self.faithfulness_rescaled = False
        self.labelled_text_rows = 0
        self.native_object_rows = 0
        self.chunk_parse_failures = 0


# ---------------------------------------------------------------------------
# pandas-safe coercion
# ---------------------------------------------------------------------------

def _is_missing(v: Any) -> bool:
    """NaN / None / NaT / pd.NA, without tripping on lists or dicts."""
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_text(v: Any) -> str:
    return "" if _is_missing(v) else str(v)


def _coerce_float(v: Any) -> Optional[float]:
    if _is_missing(v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _parse_datetime(v: Any, stats: ParseStats) -> Optional[datetime]:
    """Accept Timestamp/datetime64/datetime/date/str."""
    if _is_missing(v):
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "year") and hasattr(v, "month"):      # datetime.date
        try:
            return datetime(v.year, v.month, getattr(v, "day", 1))
        except Exception:
            pass
    raw = str(v).strip()
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
    try:
        ts = pd.to_datetime(raw, errors="coerce")
        if not pd.isna(ts):
            return ts.to_pydatetime()
    except Exception:
        pass
    stats.unparsed_dates += 1
    return None


def _first(d: Dict[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and not _is_missing(d[k]) and d[k] != "":
            return d[k]
    return None


# ---------------------------------------------------------------------------
# chunk parsing
# ---------------------------------------------------------------------------

def _chunks_from_objects(items: Sequence[Any], stats: ParseStats) -> List[Chunk]:
    chunks: List[Chunk] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            chunks.append(Chunk(text=item, score=0.0, doc_id=f"chunk_{i}", metadata={}))
            continue
        if not isinstance(item, dict):
            chunks.append(Chunk(text=_as_text(item), score=0.0,
                                doc_id=f"chunk_{i}", metadata={}))
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
        if isinstance(meta, dict):
            meta = {str(k): str(v) for k, v in meta.items() if not _is_missing(v) and v != ""}
        else:
            known = set(_TEXT_KEYS + _SCORE_KEYS + _DISTANCE_KEYS + _DOCID_KEYS + _META_KEYS)
            meta = {k: str(v) for k, v in item.items()
                    if k not in known and isinstance(v, (str, int, float, bool))}

        chunks.append(Chunk(
            text=_as_text(text),
            score=score if score is not None else 0.0,
            doc_id=_as_text(doc_id) or f"chunk_{i}",
            metadata=meta,
        ))
    return chunks


def _chunks_from_labelled_text(raw: str, stats: ParseStats) -> Optional[List[Chunk]]:
    """The `**File Name:** / **Document Title:**` form."""
    marks = list(_FILE_NAME.finditer(raw))
    if not marks:
        return None
    out: List[Chunk] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        block = raw[m.start():end]
        fname = (m.group(1) or "").strip()
        tm = _DOC_TITLE.search(block)
        title = (tm.group(1) or "").strip() if tm else ""
        body = _DOC_TITLE.sub("", _FILE_NAME.sub("", block)).strip()
        meta: Dict[str, str] = {}
        if fname:
            meta["file_name"] = fname
        if title:
            meta["document_title"] = title
        out.append(Chunk(text=body, score=0.0,
                         doc_id=fname or title or f"chunk_{i}", metadata=meta))
    stats.labelled_text_rows += 1
    return out


def parse_chunks(value: Any, stats: ParseStats) -> List[Chunk]:
    # 1. Already structured in the frame.
    if isinstance(value, (list, tuple)):
        stats.native_object_rows += 1
        if len(value) == 0:
            stats.empty_chunks += 1
            return []
        return _chunks_from_objects(list(value), stats)
    if isinstance(value, dict):
        stats.native_object_rows += 1
        inner = _first(value, ("chunks", "documents", "results", "contexts"))
        return _chunks_from_objects(list(inner) if isinstance(inner, list) else [value], stats)

    raw = _as_text(value).strip()
    if not raw:
        stats.empty_chunks += 1
        return []

    # 2. JSON / Python repr.
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            inner = _first(parsed, ("chunks", "documents", "results", "contexts"))
            parsed = list(inner) if isinstance(inner, list) else [parsed]
        if isinstance(parsed, (list, tuple)):
            return _chunks_from_objects(list(parsed), stats)
        break

    # 3. Labelled text.
    labelled = _chunks_from_labelled_text(raw, stats)
    if labelled is not None:
        return labelled

    # 4. Last resort. '|||' is NOT used: it would shred pipe tables.
    for sep in ("\n\n\n", "\n---\n", "\n___\n"):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            break
    else:
        parts = [raw]
    return [Chunk(text=p, score=0.0, doc_id=f"chunk_{i}", metadata={})
            for i, p in enumerate(parts)]


def _parse_user_attributes(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if not _is_missing(v) and v != ""}
    raw = _as_text(value).strip()
    if not raw:
        return {}
    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(raw)
            if isinstance(v, dict):
                return {str(k): str(val) for k, val in v.items()
                        if not _is_missing(val) and val != ""}
        except Exception:
            continue
    return {}


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run(df: "pd.DataFrame",
        columns: Optional[Dict[str, str]] = None,
        top_k: int = TOP_K,
        current_window: int = CURRENT_WINDOW,
        alert_sigma: float = ALERT_SIGMA,
        flag_sigma: float = FLAG_SIGMA,
        preview_rows: int = PREVIEW_ROWS,
        verbose: bool = VERBOSE,
        ) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Score every row of `df`. Returns (scores_df, summary_df)."""
    C = dict(COLUMNS if columns is None else columns)

    required = [k for k in C if k != "user_attributes"]
    missing = [C[k] for k in required if C[k] not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing column(s): {missing}. "
            f"Present: {list(df.columns)}. Fix the COLUMNS map or pass columns=."
        )
    ua_col = C.get("user_attributes")
    has_ua = bool(ua_col) and ua_col in df.columns

    stats = ParseStats()
    interactions: List[Interaction] = []
    faithfulness: List[Optional[float]] = []
    created: List[Optional[datetime]] = []
    source_index: List[Any] = []

    for pos in range(len(df)):
        row = df.iloc[pos]
        stats.rows += 1
        chunks = parse_chunks(row[C["document_chunks"]], stats)
        if any(c.score != 0.0 for c in chunks):
            stats.rows_with_scores += 1
        if any(c.metadata for c in chunks):
            stats.rows_with_metadata += 1

        f = _coerce_float(row[C["faithfulness_score"]])
        if f is not None:
            stats.rows_with_faithfulness += 1
        faithfulness.append(f)
        created.append(_parse_datetime(row[C["request_created_on"]], stats))
        source_index.append(df.index[pos])

        tid = _as_text(row[C["transaction_id"]]) or f"row_{pos}"
        interactions.append(Interaction(
            interaction_id=tid,
            query=_as_text(row[C["request"]]),
            answer=_as_text(row[C["response"]]),
            chunks=chunks,
            user_attributes=_parse_user_attributes(row[ua_col]) if has_ua else {},
        ))

    if not interactions:
        empty = pd.DataFrame()
        return empty, empty

    # Faithfulness reported 0-100 rather than 0-1.
    present = [x for x in faithfulness if x is not None]
    if present and max(present) > 1.0:
        stats.faithfulness_rescaled = True
        faithfulness = [None if x is None else x / 100.0 for x in faithfulness]

    # Chronological order; drift is meaningless without it. Rows with an
    # unparseable timestamp go last, keeping their relative order.
    order = sorted(range(len(interactions)),
                   key=lambda i: (created[i] is None, created[i] or datetime.min, i))
    interactions = [interactions[i] for i in order]
    faithfulness = [faithfulness[i] for i in order]
    created = [created[i] for i in order]
    source_index = [source_index[i] for i in order]

    judges = [from_faithfulness(lambda _ix, v=x: v) if x is not None else None
              for x in faithfulness]

    # Pass 1: baseline over the whole frame.
    baseline = Baseline(window=len(interactions) + 1, min_n=1)
    all_signals = []
    for ix, judge in zip(interactions, judges):
        sig = extract_signals(ix, top_k=top_k)
        if judge is not None:
            try:
                sig.values["unsupported_ratio"] = judge(ix)
            except Exception:
                pass
        all_signals.append(sig)
        baseline.add(sig)

    # Pass 2: score every row against that fixed baseline.
    import policy_monitor
    policy_monitor.FLAG_SIGMA = flag_sigma

    records: List[Dict[str, Any]] = []
    risks = []
    for ix, judge, ts, src in zip(interactions, judges, created, source_index):
        risk = score_risk(ix, baseline=baseline, judge=judge, top_k=top_k)
        risks.append(risk)
        v = risk.signals.values
        records.append({
            "source_index": src,
            "transaction_id": ix.interaction_id,
            "request_created_on": ts,
            "risk_score": round(risk.score, 1),
            "flags": "|".join(risk.flags),
            "n_flags": len(risk.flags),
            "clarifying_question": risk.clarifying_question or "",
            "branch_dims": json.dumps(risk.signals.branch_dims) if risk.signals.branch_dims else "",
            "notes": " ; ".join(risk.notes),
            **{f"sig_{k}": round(v.get(k, 0.0), 4) for k in SIGNAL_NAMES},
        })
    scores_df = pd.DataFrame.from_records(records)

    # ---- drift: most recent current_window rows vs everything before --------
    n = len(all_signals)
    cut = max(1, n - current_window)
    hist, curr = all_signals[:cut], all_signals[cut:]
    alerts: List[Tuple[str, float, float, float]] = []
    if len(hist) >= 30 and len(curr) >= 30:
        for name in SIGNAL_NAMES:
            h = [s.values[name] for s in hist if name in s.values]
            c = [s.values[name] for s in curr if name in s.values]
            if len(h) < 30 or len(c) < 30:
                continue
            sd = statistics.pstdev(h)
            if sd <= 1e-9:
                continue
            mu, cm = statistics.fmean(h), statistics.fmean(c)
            sigma = (cm - mu) / (sd / math.sqrt(len(c)))
            if abs(sigma) >= alert_sigma:
                alerts.append((name, mu, cm, sigma))
        alerts.sort(key=lambda a: -abs(a[3]))

    # ---- summary ------------------------------------------------------------
    summary: List[Dict[str, Any]] = []

    def add(kind: str, metric: str, value: Any, detail: str = "") -> None:
        summary.append({"kind": kind, "metric": metric, "value": value, "detail": detail})

    add("coverage", "rows", stats.rows)
    add("coverage", "rows_with_chunk_scores", stats.rows_with_scores,
        "THIN_COVERAGE / DIFFUSE_MATCH need these")
    add("coverage", "rows_with_chunk_metadata", stats.rows_with_metadata,
        "STALE_SOURCE / SENSITIVE need these; BRANCH_UNRESOLVED prefers them")
    add("coverage", "rows_with_faithfulness", stats.rows_with_faithfulness)
    add("coverage", "rows_with_empty_chunks", stats.empty_chunks)
    add("coverage", "rows_parsed_as_native_objects", stats.native_object_rows,
        "cell already held a list/dict")
    add("coverage", "rows_parsed_as_labelled_text", stats.labelled_text_rows,
        "**File Name:** / **Document Title:** form")
    add("coverage", "chunks_per_row_mean",
        round(statistics.fmean([len(i.chunks) for i in interactions]), 2),
        "if this is 1.0 the chunk delimiter was not recognised")
    add("coverage", "unparsed_dates", stats.unparsed_dates)
    add("coverage", "distance_converted_to_similarity", stats.distance_converted)
    add("coverage", "faithfulness_rescaled_from_0_100", int(stats.faithfulness_rescaled))

    flag_counts: Dict[str, int] = {}
    for r in risks:
        for fl in set(r.flags):
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    for fl, cnt in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
        add("flag_rate", fl, round(cnt / len(risks), 4), f"{cnt} of {len(risks)}")

    sc = [r.score for r in risks]
    add("risk", "mean", round(statistics.fmean(sc), 2))
    add("risk", "p50", round(statistics.median(sc), 2))
    add("risk", "p90", round(sorted(sc)[int(0.9 * (len(sc) - 1))], 2))
    add("risk", "clean_rate", round(sum(1 for r in risks if not r.flags) / len(risks), 4),
        "interactions with no flag at all")

    for name, mu, cm, sigma in alerts:
        add("drift_alert", name, round(cm, 4),
            f"baseline {mu:.4f} -> current {cm:.4f} ({sigma:+.1f} sigma, "
            f"last {len(curr)} of {n} rows)")
    if not alerts:
        add("drift_alert", "none", 0,
            f"no signal moved more than {alert_sigma} sigma"
            if len(hist) >= 30 and len(curr) >= 30
            else f"need >=30 rows each side; have {len(hist)} and {len(curr)}")

    summary_df = pd.DataFrame.from_records(summary)

    if WRITE_SCORES_CSV:
        scores_df.to_csv(WRITE_SCORES_CSV, index=False)
    if WRITE_SUMMARY_CSV:
        summary_df.to_csv(WRITE_SUMMARY_CSV, index=False)

    if preview_rows:
        print("=" * 72)
        print("PARSE PREVIEW - check this before trusting the output")
        print("=" * 72)
        for ix in interactions[:preview_rows]:
            print(f"\n[{ix.interaction_id}] parsed {len(ix.chunks)} chunk(s)")
            for c in ix.chunks[:4]:
                t, rg, orp = _table_integrity(c.text)
                print(f"   doc_id={c.doc_id!r} score={c.score} meta={c.metadata}")
                print(f"     tables={t} ragged={rg} orphan={orp} "
                      f"text[:80]={c.text[:80]!r}")
        print("\n" + "=" * 72)

    if verbose:
        print(f"scored  {len(scores_df)} rows")
        if stats.rows_with_scores == 0:
            print("NOTE    no chunk scores found -> retrieval-quality flags disabled")
        if stats.rows_with_metadata == 0:
            print("NOTE    no chunk metadata found -> staleness/sensitivity flags disabled")
        if alerts:
            print(f"ALERTS  {len(alerts)}: " + ", ".join(a[0] for a in alerts))
        top = sorted(flag_counts.items(), key=lambda kv: -kv[1])[:5]
        if top:
            print("TOP     " + ", ".join(f"{k} {v / len(risks):.0%}" for k, v in top))

    return scores_df, summary_df


# ---------------------------------------------------------------------------

def _demo_frame() -> "pd.DataFrame":
    good = ("**File Name:** leave_policy_in.pdf\n"
            "**Document Title:** India Leave Policy\n\n"
            "| Grade | Days | Carryover |\n|-------|------|-----------|\n"
            "| L1-L3 | 18   | 30        |\n\n"
            "**File Name:** leave_policy_de.pdf\n"
            "**Document Title:** Germany Leave Policy\n\n"
            "Employees accrue 28 days. Carryover must be used by 31 March.")
    ragged = ("**File Name:** expense_policy.pdf\n**Document Title:** Expense Policy\n\n"
              "| City | Breakfast | Lunch | Dinner |\n|------|----|----|----|\n"
              "| Mumbai | 300 | 600 | 900 |\n| Delhi | 350 | 650 |\n")
    orphan = ("**File Name:** notice_period.pdf\n**Document Title:** Notice Period\n\n"
              "| L1-L3 | 30 days |\n| L4-L6 | 60 days |\n")
    native = [{"text": "Probation lasts six months.", "score": 0.88,
               "doc_id": "probation.pdf", "metadata": {"country": "IN"}}]

    rows = [
        ("tx-1", "Can I carry over unused leave?",
         "Unused leave may be carried over up to 30 days.", good, 0.96, "2026-11-01 09:00:00", None),
        ("tx-2", "What is the meal allowance in Delhi?",
         "In Delhi the dinner allowance is 650.", ragged, 0.94, "2026-11-01 10:00:00", None),
        ("tx-3", "What notice period applies to me?",
         "Your notice period is 30 days.", orphan, 0.97, "2026-11-01 11:00:00", None),
        ("tx-4", "How long is probation?",
         "Probation lasts six months.", native, 0.99, pd.Timestamp("2026-11-01 12:00"), {"country": "IN"}),
        ("tx-5", "What about sabbaticals?",
         "You may want to check with your People Partner.", None, float("nan"), None, None),
    ]
    return pd.DataFrame(rows, columns=[
        "transaction_id", "request", "response", "document_chunks",
        "faithfulness_score", "request_created_on", "user_attributes"])


if __name__ == "__main__":
    scores, summary = run(_demo_frame())
    print("\n--- scores ---")
    print(scores[["transaction_id", "risk_score", "flags", "clarifying_question"]]
          .to_string(index=False))
    print("\n--- summary (coverage) ---")
    print(summary[summary["kind"] == "coverage"][["metric", "value"]].to_string(index=False))
