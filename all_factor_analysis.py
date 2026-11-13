# -------------------------------------------------------------------------
# Dataiku — quality factor analysis
#
# Answers one question: for each factor the user would complain about,
# what drove it?
#
# Self-contained. No LLM calls. No imports beyond dataiku/pandas.
# -------------------------------------------------------------------------
import json
import math
import re
import textwrap

import dataiku
import pandas as pd
from dataiku import spark as dkuspark
from pyspark.sql import SQLContext
from pyspark import SparkContext

# ---- CONFIG -------------------------------------------------------------
AUDIT_DATASET  = "audit_output"      # Dataiku dataset: flag_* columns
TRACES_SOURCE  = "policy_traces"     # Dataiku dataset OR hive db.table
TRACES_IS_HIVE = False               # True -> read TRACES_SOURCE as a Hive table
OUTPUT_DATASET = None                # e.g. "audit_factors"; None = don't write
LONG_DATASET   = None                # e.g. "audit_factors_long" for Tableau

sc  = SparkContext.getOrCreate()
sql = SQLContext(sc)


def read_spark(name, is_hive=False):
    if is_hive:
        return sql.table(name)
    return dkuspark.get_dataframe(sql, dataiku.Dataset(name))


# 644 rows is small. Row-wise Python on pandas beats a Spark UDF here.
# If this ever runs over millions of rows, push factor_row into a UDF
# instead of collecting.
audit  = read_spark(AUDIT_DATASET).toPandas()
traces = read_spark(TRACES_SOURCE, TRACES_IS_HIVE).toPandas()

print(f"audit  {audit.shape}")
print(f"traces {traces.shape}")

# Bring the chunks across; the audit output drops them because they are large
if "document_chunks" in traces.columns and "document_chunks" not in audit.columns:
    audit["transaction_id"] = audit["transaction_id"].astype(str)
    lut = (traces.assign(transaction_id=traces["transaction_id"].astype(str))
                 .drop_duplicates("transaction_id")
                 .set_index("transaction_id")["document_chunks"])
    audit["document_chunks"] = audit["transaction_id"].map(lut)
    print(f"chunks attached to {audit['document_chunks'].notna().sum()}/{len(audit)} rows")


# ---- SCOPE OF THE ANALYSIS ---------------------------------------------
# CHUNKS_ONLY: drop rows with no source chunks. The chunk-based flags could
#   never have been judged on those, so they quietly depress every rate.
# THEMED_ONLY: keep only rows the audit gave a theme. See the warning below
#   before turning this on — it excludes the generation defects.
CHUNKS_ONLY = True
THEMED_ONLY = True

_start = len(audit)

# Snapshot before filtering. THEMED_ONLY removes every expected-behaviour
# row by definition, so the expected share must be measured here or it
# reads as zero.
audit_full = audit.copy()

if CHUNKS_ONLY and "document_chunks" in audit.columns:
    _blank = audit["document_chunks"].astype(str).str.strip().isin(
        ["", "[]", "null", "None", "nan"])
    has = audit["document_chunks"].notna() & (~_blank)
    dropped = int((~has).sum())
    audit = audit[has].copy()
    print(f"\nchunks filter : kept {len(audit)}, dropped {dropped} "
          f"with no chunks")
elif CHUNKS_ONLY:
    print("\nwarning: no document_chunks column - chunks filter skipped")

# Flags that produce NO theme, because themes_for covers routing and
# coverage defects but not generation defects. A themed-only filter
# removes these rows even though they carry the strongest evidence.
THEMELESS_FLAGS = ["ANSWER_CONTRADICTS_CHUNKS", "ANSWER_MISREADS_TABLE",
                   "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY", "CHUNKS_CONFLICT"]

if THEMED_ONLY:
    _theme_col = next((c for c in audit.columns
                       if c.lower() in ("primary_theme", "themes")), None)
    if _theme_col:
        keep = (audit[_theme_col].notna()
                & (audit[_theme_col].astype(str).str.strip()
                   .isin(["", "nan", "None"]) == False))
        losing = audit[~keep]

        # how many of the dropped rows carried a strong generation defect?
        lost_strong = {}
        for fl in THEMELESS_FLAGS:
            c = next((c for c in losing.columns
                      if c.lower() == f"flag_{fl}".lower()), None)
            if c is not None:
                n = int((losing[c].astype(str).str.strip().str.upper()
                         == "TRUE").sum())
                if n:
                    lost_strong[fl] = n

        audit = audit[keep].copy()
        print(f"theme filter  : kept {len(audit)}, dropped "
              f"{int((~keep).sum())} with no theme")
        if lost_strong:
            print("\n  WARNING: the theme filter just removed rows carrying "
                  "strong evidence:")
            for fl, n in sorted(lost_strong.items(), key=lambda x: -x[1]):
                print(f"    {fl:38s} {n} rows")
            print("  These are generation defects. themes_for has no theme "
                  "for them,\n  so filtering on themes excludes them. Set "
                  "THEMED_ONLY = False to keep them.")
    else:
        print("warning: no primary_theme/themes column - theme filter skipped")

print(f"\nanalysing {len(audit)} of {_start} interactions")


# ---- FACTOR DEFINITIONS -------------------------------------------------
TRUE, FALSE = "TRUE", "FALSE"

# Hive and Spark lowercase column names, so flag_ANSWER_X arrives as
# flag_answer_x. Look columns up case-insensitively, or every flag reads as
# missing and every factor comes out zero.
COLMAP = {c.lower(): c for c in audit.columns}


def _col(name):
    return COLMAP.get(str(name).lower())


def _val(v):
    """Normalise TRUE/FALSE however the store wrote them."""
    if isinstance(v, bool):
        return TRUE if v else FALSE
    s = str(v).strip().upper()
    if s in ("TRUE", "T", "1", "1.0", "Y", "YES"):
        return TRUE
    if s in ("FALSE", "F", "0", "0.0", "N", "NO"):
        return FALSE
    return s


def flagval(r, flag):
    c = _col(f"flag_{flag}")
    return _val(r.get(c)) if c else None

FACTORS = {
    "Incorrect":       "The answer is wrong",
    "Incomplete":      "The answer is missing important information",
    "Irrelevant":      "The answer does not address the question",
    "Inconsistent":    "The answer contradicts other information",
    "Unsubstantiated": "The answer is not supported by references",
}

# Which flags drive each factor. INVERTED flags are a defect when FALSE.
DRIVERS = {
    "Incorrect": ["ANSWER_CONTRADICTS_CHUNKS", "ANSWER_MISREADS_TABLE",
                  "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY"],
    "Incomplete": ["ANSWER_COVERS_ALL_PARTS", "ANSWER_OMITS_CONDITION_OR_EXCEPTION",
                   "CHUNKS_PARTIAL_COVERAGE", "AUDIT_INPUT_LIMITED"],
    "Irrelevant": ["ANSWER_ADDRESSES_QUESTION", "GUARDRAIL_FALSE_BLOCK",
                   "ACTION_GUARDRAIL_FALSE_BLOCK", "CLARIFICATION_UNNECESSARY"],
    "Inconsistent": ["ANSWERED_ELSEWHERE", "CHUNKS_CONFLICT",
                     "CHUNKS_WRONG_APPLICABILITY", "HISTORY_HAS_NEEDED_INFO"],
    "Unsubstantiated": ["ANSWER_HAS_UNSUPPORTED_CLAIM", "ANSWER_NUMBERS_UNSUPPORTED",
                        "ANSWERED_WITH_SILENT_ASSUMPTION"],
}
INVERTED = {"ANSWER_ADDRESSES_QUESTION", "ANSWER_COVERS_ALL_PARTS"}

# Factors that only make sense when the agent actually answered
ANSWER_FACTORS = {"Incorrect", "Incomplete", "Unsubstantiated"}

# Flags that had to quote evidence on both sides to survive the audit's own
# checks. A factor resting on one of these is defensible in review.
STRONG = {"ANSWER_CONTRADICTS_CHUNKS", "ANSWER_MISREADS_TABLE",
          "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY", "GUARDRAIL_FALSE_BLOCK",
          "ACTION_GUARDRAIL_FALSE_BLOCK", "ANSWERED_ELSEWHERE",
          "ANSWER_ADDRESSES_QUESTION", "ANSWER_COVERS_ALL_PARTS",
          "CHUNKS_CONFLICT"}


# ---- EVIDENCE LOCATION --------------------------------------------------
# Finding where a judge's quote came from is a PROVENANCE problem, not a
# retrieval one: we want the exact words, not the topically-similar chunk.
# Lexical matching is the right tool, and it works because the prompt asks
# the judge to quote rather than summarise. The score says which case you
# are in — near 1.0 verbatim, under 0.3 paraphrased or fabricated.

_WORD = re.compile(r"[A-Za-z][\w'-]+|\d+(?:[.,]\d+)?")
_NUM  = re.compile(r"\d+(?:[.,]\d+)?")
_STOPW = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with",
    "from", "which", "their", "there", "have", "has", "had", "not", "but",
    "any", "all", "may", "must", "shall", "will", "can", "should", "would",
    "policy", "policies", "document", "documents", "standard", "standards",
}

# Flags whose evidence points at chunk text worth locating
CHUNK_FLAGS = {"ANSWER_CONTRADICTS_CHUNKS", "ANSWER_MISREADS_TABLE",
               "ANSWER_HAS_UNSUPPORTED_CLAIM",
               "ANSWER_OMITS_CONDITION_OR_EXCEPTION",
               "CHUNKS_WRONG_APPLICABILITY", "CHUNKS_CONFLICT",
               "CHUNKS_PARTIAL_COVERAGE", "AUDIT_INPUT_LIMITED"}


def _toks(s):
    return {w.lower() for w in _WORD.findall(str(s))
            if w.lower() not in _STOPW and len(w) > 2}


def _sentences(text):
    """Split on sentence ends and newlines, keeping table rows whole."""
    out = []
    for line in str(text).split("\n"):
        if line.strip().startswith("|") or line.count("|") >= 2:
            out.append(line.strip())
            continue
        for part in re.split(r"(?<=[.;:])\s+", line):
            if part.strip():
                out.append(part.strip())
    return out


def parse_chunks(raw):
    if raw is None or str(raw).strip() in ("", "[]", "null", "None", "nan"):
        return []
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return [{"doc_title": "?", "text": str(raw)}]
    if not isinstance(obj, list):
        obj = [obj]
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append({"doc_title": str(item.get("doc_title")
                                         or item.get("title") or "?"),
                        "text": str(item.get("text")
                                    or item.get("content") or "")})
        else:
            txt = str(item)
            m = re.search(r"\*{0,2}Document Title:?\*{0,2}\s*(.+)", txt)
            title = re.sub(r"\*+", "", m.group(1)).strip() if m else "?"
            out.append({"doc_title": title, "text": txt})
    return out


def split_evidence(ev):
    """Return (chunk_side, answer_side) from 'chunk: X || answer: Y'."""
    ev = str(ev or "")
    if "||" in ev:
        a, b = ev.split("||", 1)
    elif re.search(r"\banswer\s*:", ev, re.I):
        a, b = re.split(r"\banswer\s*:", ev, maxsplit=1, flags=re.I)
        b = "answer: " + b
    else:
        return ev.strip(), ""
    cut = lambda t, w: re.sub(rf"^\s*{w}\s*:\s*", "", t.strip(), flags=re.I)
    return cut(a, r"(chunk|row|source)"), cut(b, r"(answer|response)")


def locate(quote, chunks, context=1, top=2):
    """Where did this quote come from? Best matches first.

    Scoring is token overlap, with two lexical sharpeners:
      - a number shared with the quote is near-decisive in policy text
      - rare terms count more than common ones (IDF over these chunks)
    """
    qt = _toks(quote)
    if not qt or not chunks:
        return []

    sent_lists = [_sentences(c["text"]) for c in chunks]
    # document frequency across sentences, for rare-term weighting
    dfreq, nsent = {}, sum(len(x) for x in sent_lists) or 1
    for sl in sent_lists:
        for sent in sl:
            for t in _toks(sent):
                dfreq[t] = dfreq.get(t, 0) + 1
    weight = {t: math.log(1 + nsent / (1 + dfreq.get(t, 0))) for t in qt}
    total_w = sum(weight.values()) or 1.0
    qnums = set(_NUM.findall(str(quote)))

    hits = []
    for ci, sents in enumerate(sent_lists):
        for si, sent in enumerate(sents):
            shared = qt & _toks(sent)
            if not shared:
                continue
            score = sum(weight[t] for t in shared) / total_w
            # numbers are high-precision anchors
            if qnums and qnums & set(_NUM.findall(sent)):
                score = min(1.0, score + 0.25)
            lo, hi = max(0, si - context), min(len(sents), si + context + 1)
            hits.append({"chunk": ci + 1,
                         "doc_title": chunks[ci]["doc_title"],
                         "score": round(min(score, 1.0), 2),
                         "sentence": sent,
                         "context": " ".join(sents[lo:hi])})
    hits.sort(key=lambda h: -h["score"])
    return hits[:top]


def row_evidence(r):
    """The evidence_json dict for one row, tolerating a mangled string."""
    raw = r.get(_col("evidence_json") or "evidence_json")
    if not raw or str(raw).strip() in ("nan", "None", ""):
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def driver_evidence(r, flag, ev=None):
    """Readable evidence for ONE driver flag, with four fallbacks.

    1. the judge's own quote
    2. a generated explanation for DERIVED flags, which have no quote
    3. a generated explanation for CODE flags
    4. the raw flag value, so nothing comes back blank
    """
    ev = row_evidence(r) if ev is None else ev
    q = ev.get(flag)
    if q and str(q).strip() not in ("", "nan", "stub"):
        return str(q)

    exp = r.get(_col("flag_EXPECTED_ROUTE") or "") or r.get("EXPECTED_ROUTE")
    cat = r.get(_col("scope_category") or "scope_category")
    route = r.get(_col("route") or "route")

    if flag == "GUARDRAIL_FALSE_BLOCK":
        why = [w for w, f in (
            ("same question answered elsewhere", "ANSWERED_ELSEWHERE"),
            ("the corpus holds the answer", "CORPUS_HAS_ANSWER"),
            ("on-topic chunks were retrieved", "CHUNKS_ON_TOPIC"),
            ("a corpus title matched", "CORPUS_TITLE_MATCHED"))
            if flagval(r, f) == TRUE]
        return (f"refused as out of scope though scope={cat}; "
                f"corroborated by: "
                + ("; ".join(why) if why else "nothing (suspected only)"))
    if flag == "ACTION_GUARDRAIL_FALSE_BLOCK":
        return (f"refused as a material action, but the request asks about "
                f"the process rather than to perform it (scope={cat})")
    if flag == "CLARIFICATION_UNNECESSARY":
        clear = r.get(_col("clarify_sources_clear") or "")
        return (f"clarified although already clear from: {clear}"
                if clear else "clarified although the request was answerable")
    if flag == "ANSWER_NUMBERS_UNSUPPORTED":
        return "a number in the answer does not appear in the chunks"
    if flag == "ANSWERED_ELSEWHERE":
        return "the same question was answered in another interaction"

    v = flagval(r, flag)
    return f"flag is {v}" + (" (no evidence recorded)" if v == TRUE else "")


def factor_row(r):
    """Factors for one row, as {factor: [driver flags]}.

    Only populated when the audit did NOT clear the row as expected
    behaviour — a row judged correct cannot also be a user complaint.
    """
    exp_c = _col("expected_behavior")
    if exp_c and _val(r.get(exp_c)) == TRUE:
        return {}

    # Three factors presuppose that an answer exists. On a clarified or
    # refused row there is no answer to be wrong, incomplete or
    # unsupported — and on a clarified row, limited inputs are what
    # JUSTIFY clarifying, so using them to tag a defect inverts the
    # meaning. Irrelevant and Inconsistent stay route-agnostic.
    route = str(r.get(_col("route") or "route") or "").strip().lower()
    answered = route == "answered"

    def fires(flag):
        return flagval(r, flag) == (FALSE if flag in INVERTED else TRUE)

    out = {}
    for factor, flags in DRIVERS.items():
        if factor in ANSWER_FACTORS and not answered:
            continue
        hits = [f for f in flags if fires(f)]
        # an acknowledged corpus gap is honest behaviour, not Incomplete
        if (factor == "Incomplete" and "AUDIT_INPUT_LIMITED" in hits
                and flagval(r, "ANSWER_ACKNOWLEDGES_LIMITATION") == TRUE):
            hits.remove("AUDIT_INPUT_LIMITED")
        if hits:
            out[factor] = hits
    return out


# ---- DIAGNOSTIC ---------------------------------------------------------
# Everything coming out zero almost always means the flag columns were not
# found, not that the agent was perfect. Check before trusting any table.
all_drivers = sorted({f for fl in DRIVERS.values() for f in fl})
found   = [f for f in all_drivers if _col(f"flag_{f}")]
missing = [f for f in all_drivers if not _col(f"flag_{f}")]

print("\n" + "=" * 78)
print("DIAGNOSTIC")
print("=" * 78)
print(f"flag_* columns in dataset : "
      f"{sum(1 for c in audit.columns if c.lower().startswith('flag_'))}")
print(f"driver flags found        : {len(found)}/{len(all_drivers)}")
if missing:
    print(f"driver flags MISSING      : {', '.join(missing)}")
if not found:
    print("\n>> No driver flags found, so every factor will be zero.")
    print("   Columns actually present:")
    print("  ", list(audit.columns)[:25])
    raise SystemExit("flag columns not found - fix the column names first")

for f in found[:6]:
    vc = audit[_col(f"flag_{f}")].map(_val).value_counts().to_dict()
    print(f"   {f:36s} {vc}")

exp_c = _col("expected_behavior")
if exp_c:
    print(f"\nexpected_behavior values  : "
          f"{audit[exp_c].map(_val).value_counts().to_dict()}")
else:
    print("\nnote: no expected_behavior column, so no row is cleared and "
          "factor counts will run higher than the audit intended")

rows = audit.to_dict("records")
facts = [factor_row(r) for r in rows]

if not any(facts):
    print("\n>> Flags were found but no factor fired on any row.")
    print("   Either this slice genuinely has no defects, or every row is "
          "marked expected_behavior. The values above tell you which.")

audit["factors"] = ["; ".join(f) for f in facts]
audit["factor_drivers"] = [
    " | ".join(f"{k}: {', '.join(v)}" for k, v in f.items()) for f in facts]
audit["factor_count"] = [len(f) for f in facts]

# why_flagged covers only review-queue reasons. factor_why covers every
# factor driver, so a row whose factors never raised a review still
# explains itself.
audit["factor_why"] = [
    " | ".join(f"{fa}: " + "; ".join(
        f"{fl} -> {driver_evidence(r, fl)}" for fl in flags)
        for fa, flags in f.items()) if f else ""
    for r, f in zip(rows, facts)]
for fa in FACTORS:
    audit[f"is_{fa}"] = [fa in f for f in facts]


# ---- 1. HOW MANY OF EACH ------------------------------------------------
total = max(len(audit), 1)
summary = pd.DataFrame([{
    "Factor": fa,
    "What the user experienced": desc,
    "Interactions": int(audit[f"is_{fa}"].sum()),
    "Share": f"{100 * audit[f'is_{fa}'].sum() / total:.1f}%",
} for fa, desc in FACTORS.items()])

print("\n" + "=" * 78)
print("WHAT THE USER EXPERIENCED")
print("=" * 78)
print(summary.to_string(index=False))


# ---- 2. THE CAUSE BEHIND EACH FACTOR ------------------------------------
# The driver flag IS the cause, stated at the level a fix happens.
drivers = []
for f in facts:
    for factor, flags in f.items():
        for fl in flags:
            drivers.append({
                "Factor": factor,
                "Driven by": fl,
                "Evidence strength": "strong" if fl in STRONG else "weak",
            })

print("\n" + "=" * 78)
print("WHAT DROVE EACH FACTOR")
print("=" * 78)
if drivers:
    dd = (pd.DataFrame(drivers)
            .groupby(["Factor", "Driven by", "Evidence strength"])
            .size().reset_index(name="Interactions")
            .sort_values(["Factor", "Interactions"], ascending=[True, False]))
    print(dd.to_string(index=False))
else:
    print("(no factors fired)")


# ---- 3. FACTOR x PIPELINE CAUSE -----------------------------------------
# Only if the audit output carries a cause column.
cause_col = next((c for c in ("all_causes", "cause") if c in audit.columns), None)
if cause_col:
    pairs = []
    for r, f in zip(rows, facts):
        causes = [c.strip() for c in str(r.get(cause_col) or "").split(";")
                  if c.strip()] or ["(none assigned)"]
        for factor in f:
            for c in causes:
                pairs.append({"Factor": factor, "Pipeline cause": c})
    if pairs:
        print("\n" + "=" * 78)
        print("FACTOR  x  PIPELINE CAUSE")
        print("=" * 78)
        print(pd.crosstab(pd.DataFrame(pairs)["Factor"],
                          pd.DataFrame(pairs)["Pipeline cause"],
                          margins=True, margins_name="TOTAL").to_string())


# ---- 4. DRILL INTO ONE FACTOR -------------------------------------------
def show(factor, n=5, strong_only=True, locate_chunks=True):
    """Print the interactions behind one factor, with the chunk located."""
    sub = audit[audit[f"is_{factor}"]]
    if strong_only:
        keep = [i for i, f in zip(sub.index, [factor_row(r) for r in
                                              sub.to_dict("records")])
                if any(fl in STRONG for fl in f.get(factor, []))]
        sub = sub.loc[keep]
    print(f"\n{factor.upper()} — {len(sub)} interactions"
          f"{' (strong evidence only)' if strong_only else ''}\n")

    for _, row in sub.head(n).iterrows():
        r = row.to_dict()
        print("-" * 78)
        print(f"{r['transaction_id']}   route={r.get('route')}")
        print(f"  Q: {str(r.get('request'))[:150]}")
        print(f"  A: {str(r.get('response'))[:200]}")

        ev     = row_evidence(r)
        chunks = parse_chunks(r.get("document_chunks")) if locate_chunks else []
        drivers = factor_row(r).get(factor, [])

        for fl in drivers:
            print(f"\n  [{fl}]")
            text = driver_evidence(r, fl, ev)
            cside, aside = split_evidence(ev.get(fl, ""))
            if cside:
                print(f"    judge quoted (chunk) : {cside[:160]}")
            if aside:
                print(f"    judge quoted (answer): {aside[:160]}")
            if not cside and not aside:
                print(f"    {text[:200]}")

            if chunks and fl in CHUNK_FLAGS and cside:
                hits = locate(cside, chunks)
                if not hits or hits[0]["score"] < 0.3:
                    print("    >> NOT FOUND in the chunks. The quote was "
                          "paraphrased or invented — treat as unverified.")
                else:
                    h = hits[0]
                    tag = "" if h["score"] >= 0.8 else "  (partial match)"
                    print(f"    FOUND chunk [{h['chunk']}] "
                          f"{h['doc_title']}  match {h['score']}{tag}")
                    for line in textwrap.wrap(h["context"], 88):
                        print(f"       {line}")


for fa in FACTORS:
    if audit[f"is_{fa}"].any():
        show(fa, n=2)


# ---- 5. WEEK ON WEEK ----------------------------------------------------
DATE_COL      = "request_created_on"
MIN_WEEK_ROWS = 20   # below this a weekly rate is too noisy to read
NOISE_COUNT   = 2    # a move of this many interactions or fewer is noise


def _arrow(d):
    if d is None or pd.isna(d):
        return "  "
    return "up" if d > 0 else ("dn" if d < 0 else "--")


if DATE_COL in audit.columns:
    audit["_ts"] = pd.to_datetime(audit[DATE_COL], errors="coerce")
    bad = audit["_ts"].isna().sum()
    if bad:
        print(f"\nnote: {bad} rows have an unparseable {DATE_COL}, "
              f"excluded from the weekly view")
    wk = audit[audit["_ts"].notna()].copy()
    # week starting Monday
    wk["week"] = wk["_ts"].dt.to_period("W-SUN").dt.start_time.dt.date

    weeks = sorted(wk["week"].unique())
    vol   = wk.groupby("week").size()

    print("\n" + "=" * 78)
    print("WEEK ON WEEK — factor rate (% of that week's interactions)")
    print("=" * 78)

    table = pd.DataFrame(index=weeks)
    table["Interactions"] = [int(vol.get(w, 0)) for w in weeks]
    for fa in FACTORS:
        counts = wk[wk[f"is_{fa}"]].groupby("week").size()
        table[fa] = [
            round(100 * counts.get(w, 0) / max(vol.get(w, 0), 1), 1)
            for w in weeks]
    table["thin?"] = ["yes" if v < MIN_WEEK_ROWS else ""
                      for v in table["Interactions"]]
    print(table.to_string())
    if (table["Interactions"] < MIN_WEEK_ROWS).any():
        print(f"\n'thin' weeks have under {MIN_WEEK_ROWS} interactions — "
              f"their rates swing on one or two rows. Do not read a trend "
              f"into them.")

    # ---- change vs the previous week, in counts AND points ----
    print("\n" + "=" * 78)
    print("CHANGE VS PREVIOUS WEEK")
    print("=" * 78)
    moves = []
    for fa in FACTORS:
        counts = wk[wk[f"is_{fa}"]].groupby("week").size()
        for prev, cur in zip(weeks, weeks[1:]):
            c0, c1 = int(counts.get(prev, 0)), int(counts.get(cur, 0))
            r0 = 100 * c0 / max(vol.get(prev, 0), 1)
            r1 = 100 * c1 / max(vol.get(cur, 0), 1)
            moves.append({
                "Week": cur, "Factor": fa,
                "Interactions": c1, "Was": c0,
                "Change": c1 - c0,
                "Rate %": round(r1, 1),
                "Point change": round(r1 - r0, 1),
                "Dir": _arrow(c1 - c0),
                "Signal?": ("noise" if abs(c1 - c0) <= NOISE_COUNT
                            else "worth a look"),
            })
    if moves:
        md = pd.DataFrame(moves)
        md = md[(md["Interactions"] > 0) | (md["Was"] > 0)]
        print(md.to_string(index=False))
        print(f"\na move of {NOISE_COUNT} interactions or fewer is marked "
              f"noise — at this volume it is not distinguishable from "
              f"random variation.")

    # ---- which driver flag is actually moving ----
    drv_rows = []
    for r, f in zip(wk.to_dict("records"), [factor_row(r) for r in
                                            wk.to_dict("records")]):
        for factor, flags in f.items():
            for fl in flags:
                drv_rows.append({"week": r["week"], "Factor": factor,
                                 "Driven by": fl})
    if drv_rows:
        print("\n" + "=" * 78)
        print("WHICH DRIVER IS MOVING (counts by week)")
        print("=" * 78)
        dv = pd.DataFrame(drv_rows)
        piv = pd.crosstab([dv["Factor"], dv["Driven by"]], dv["week"])
        piv["total"] = piv.sum(axis=1)
        print(piv.sort_values("total", ascending=False).to_string())
        print("\nA factor that looks flat can hide one driver rising while "
              "another falls. This is where that shows up.")
else:
    print(f"\n(no {DATE_COL} column — skipping the weekly view)")


# ---- 6. CHARTS (preview of the Tableau build) ---------------------------
# Same six views you would build in Tableau. Render inline in the Dataiku
# notebook, or set CHARTS_HTML to save one shareable file.
CHARTS_HTML = None          # e.g. "/tmp/audit_charts.html"

# How to render inline.
#   "html" - IPython.display.HTML. Works WITHOUT nbformat. Default.
#   "show" - plotly fig.show(). Needs nbformat >= 4.2.0 installed.
#   "none" - render nothing inline; use CHARTS_HTML to save a file instead.
CHART_RENDER = "html"

PALETTE = {"strong": "#1f6f4a", "weak": "#c9a227",
           "expected": "#2f7d55", "defect": "#b8562f",
           "review": "#b8562f", "sample": "#4a7ba7", "cleared": "#8aa79b"}


def _render(figs):
    """Render inline without requiring nbformat.

    fig.show() goes through plotly's mime-type renderer, which raises
    "mime type rendering requires nbformat>=4.2.0" when that package is
    absent. Handing the HTML straight to IPython skips that path.
    """
    if CHART_RENDER == "show":
        for _, fig in figs:
            fig.show()
        return
    try:
        from IPython.display import HTML, display
    except ImportError:
        print("no IPython available - set CHARTS_HTML to save a file instead")
        return
    for i, (name, fig) in enumerate(figs):
        # load plotly.js once, from CDN, on the first chart only
        html = fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if i == 0 else False)
        display(HTML(html))


def build_charts(show=True, html_path=None):
    try:
        import plotly.express as px
    except ImportError:
        print("\nplotly not installed - skipping charts.")
        print("  pip install plotly")
        print("  (px.bar and px.imshow are all this uses; both ship with "
              "the base plotly package)")
        return []

    figs = []
    nfull = max(len(audit_full), 1)

    # --- 1. Triage split: how much was EXPECTED ---------------------
    def _count(col):
        if col not in audit_full.columns:
            return 0
        return int(audit_full[col].map(_val).eq(TRUE).sum())

    n_exp    = _count("expected_behavior")
    n_rev    = _count("needs_review")
    n_samp   = _count("in_sample_queue")
    n_other  = max(nfull - n_exp - n_rev - n_samp, 0)

    split = pd.DataFrame([
        ("Expected behaviour", n_exp,   "expected"),
        ("Flagged for review", n_rev,   "review"),
        ("Control sample",     n_samp,  "sample"),
        ("Cleared, unsampled", n_other, "cleared"),
    ], columns=["Outcome", "Interactions", "key"])
    split["Share"] = (100 * split["Interactions"] / nfull).round(1)

    f1 = px.bar(split, x="Interactions", y="Outcome", orientation="h",
                text=split.apply(
                    lambda r: f"{r['Interactions']}  ({r['Share']}%)", axis=1),
                color="key", color_discrete_map=PALETTE,
                title=f"Triage split - {nfull} interactions "
                      f"({100 * n_exp / nfull:.0f}% expected behaviour)")
    f1.update_layout(showlegend=False, yaxis_title="", height=320)
    figs.append(("triage_split", f1))

    # --- 2. Route x expected vs defect ------------------------------
    rc = _col("route") or "route"
    ec = _col("expected_behavior")
    if rc in audit_full.columns and ec:
        rx = audit_full.copy()
        rx["Outcome"] = rx[ec].map(
            lambda v: "expected" if _val(v) == TRUE else "defect")
        g = (rx.groupby([rc, "Outcome"]).size()
               .reset_index(name="Interactions"))
        f2 = px.bar(g, x="Interactions", y=rc, color="Outcome",
                    orientation="h", barmode="stack",
                    color_discrete_map=PALETTE,
                    title="Where the agent is reliable, by route")
        f2.update_layout(yaxis_title="", height=320)
        figs.append(("route_outcome", f2))

    # --- 3. Factor x driver, coloured by evidence strength ----------
    drv = []
    for r, f in zip(rows, facts):
        for factor, flags in f.items():
            for fl in flags:
                drv.append({"Factor": factor, "Driver": fl,
                            "Strength": "strong" if fl in STRONG else "weak",
                            "transaction_id": r.get("transaction_id")})
    if drv:
        dd = pd.DataFrame(drv)
        g = (dd.groupby(["Factor", "Driver", "Strength"])["transaction_id"]
               .nunique().reset_index(name="Interactions"))
        f3 = px.bar(g, x="Interactions", y="Factor", color="Driver",
                    orientation="h", barmode="stack", text="Interactions",
                    title="What drove each factor")
        f3.update_layout(yaxis_title="", height=420)
        figs.append(("factor_driver", f3))

        # --- 4. Evidence strength per factor ------------------------
        g2 = (dd.groupby(["Factor", "Strength"])["transaction_id"]
                .nunique().reset_index(name="Interactions"))
        f4 = px.bar(g2, x="Interactions", y="Factor", color="Strength",
                    orientation="h", barmode="stack", text="Interactions",
                    color_discrete_map=PALETTE,
                    title="Evidence strength - what can be defended "
                          "in review")
        f4.update_layout(yaxis_title="", height=360)
        figs.append(("evidence_strength", f4))

    # --- 5. Factor x cause heatmap ----------------------------------
    cc = _col("all_causes") or _col("cause")
    if drv and cc:
        pairs = []
        for r, f in zip(rows, facts):
            causes = [c.strip() for c in str(r.get(cc) or "").split(";")
                      if c.strip()] or ["(none assigned)"]
            for factor in f:
                for c in causes:
                    pairs.append({"Factor": factor, "Cause": c})
        if pairs:
            pv = pd.crosstab(pd.DataFrame(pairs)["Factor"],
                             pd.DataFrame(pairs)["Cause"])
            f5 = px.imshow(pv, text_auto=True, aspect="auto",
                           color_continuous_scale="Teal",
                           title="Factor x root cause")
            f5.update_layout(height=380)
            figs.append(("factor_cause", f5))

    # --- 6. Owner workload ------------------------------------------
    oc = _col("cause_owners") or _col("cause_owner")
    if oc and oc in audit.columns:
        own = []
        for r in rows:
            for o in {x.strip() for x in str(r.get(oc) or "").split(";")
                      if x.strip()}:
                own.append({"Owner": o,
                            "transaction_id": r.get("transaction_id")})
        if own:
            g3 = (pd.DataFrame(own).groupby("Owner")["transaction_id"]
                    .nunique().reset_index(name="Interactions")
                    .sort_values("Interactions"))
            f6 = px.bar(g3, x="Interactions", y="Owner", orientation="h",
                        text="Interactions",
                        title="Who fixes what")
            f6.update_layout(yaxis_title="", height=340)
            figs.append(("owner_workload", f6))

    # --- 7. Weekly volume: BARS, not lines --------------------------
    # At this volume a line implies a trend that is not distinguishable
    # from noise. Bars with counts on them are the honest form.
    if DATE_COL in audit.columns and drv:
        w = audit.copy()
        w["_ts"] = pd.to_datetime(w[DATE_COL], errors="coerce")
        w = w[w["_ts"].notna()]
        if len(w):
            w["week"] = w["_ts"].dt.to_period("W-SUN").dt.start_time.dt.date
            wk = []
            for fa in FACTORS:
                if f"is_{fa}" in w.columns:
                    for week, n in w[w[f"is_{fa}"]].groupby("week").size().items():
                        wk.append({"Week": str(week), "Factor": fa,
                                   "Interactions": int(n)})
            if wk:
                f7 = px.bar(pd.DataFrame(wk), x="Week", y="Interactions",
                            color="Factor", barmode="group", text="Interactions",
                            title="Weekly counts (bars, not a line - at this "
                                  "volume a trend line would mislead)")
                f7.update_layout(height=360)
                figs.append(("weekly", f7))

    if show and CHART_RENDER != "none":
        _render(figs)
    if html_path:
        with open(html_path, "w") as fh:
            fh.write("<html><body style='font-family:sans-serif'>")
            for name, fig in figs:
                fh.write(fig.to_html(full_html=False,
                                     include_plotlyjs="cdn"))
            fh.write("</body></html>")
        print(f"\ncharts written to {html_path}")
    return figs


_figs = build_charts(show=True, html_path=CHARTS_HTML)


# ---- 7. TABLEAU EXTRACTS ------------------------------------------------
# quality_factors / all_causes are multi-value strings, which Tableau
# cannot count. Build a LONG table: one row per (interaction, factor,
# driver). In Tableau always use COUNTD(transaction_id), never SUM(1) —
# an interaction carrying two factors appears twice and SUM double-counts.
long_rows = []
for r, f in zip(rows, facts):
    causes = [c.strip() for c in
              str(r.get(_col("all_causes") or "all_causes") or "").split(";")
              if c.strip()] or ["(none assigned)"]
    for factor, flags in f.items():
        for fl in flags:
            long_rows.append({
                "transaction_id": r.get("transaction_id"),
                "week": (pd.to_datetime(r.get(DATE_COL), errors="coerce")
                           .to_period("W-SUN").start_time.date()
                         if r.get(DATE_COL) else None),
                "route": r.get(_col("route") or "route"),
                "theme": r.get(_col("primary_theme") or "primary_theme"),
                "factor": factor,
                "driver": fl,
                "evidence_strength": "strong" if fl in STRONG else "weak",
                "cause": causes[0],
                "all_causes": "; ".join(causes),
                "owner": r.get(_col("cause_owners") or "cause_owners"),
                "evidence": driver_evidence(r, fl)[:400],
            })

long_df = pd.DataFrame(long_rows)
print("\n" + "=" * 78)
print("TABLEAU EXTRACT")
print("=" * 78)
print(f"long table: {len(long_df)} rows "
      f"({long_df['transaction_id'].nunique() if len(long_df) else 0} "
      f"distinct interactions)")
print("In Tableau use COUNTD(transaction_id), never SUM(1).")
if len(long_df):
    print(long_df.head(5).to_string(index=False))

if LONG_DATASET and len(long_df):
    dataiku.Dataset(LONG_DATASET).write_with_schema(long_df)
    print(f"wrote {LONG_DATASET}")


# ---- 8. WRITE BACK ------------------------------------------------------
if OUTPUT_DATASET:
    keep = [c for c in audit.columns if not c.startswith("flag_")
            and c != "document_chunks"]
    dataiku.Dataset(OUTPUT_DATASET).write_with_schema(audit[keep])
    print(f"\nwrote {OUTPUT_DATASET} ({len(audit)} rows)")
