# -------------------------------------------------------------------------
# Dataiku — quality factor analysis
#
# Answers one question: for each factor the user would complain about,
# what drove it?
#
# Self-contained. No LLM calls. No imports beyond dataiku/pandas.
# -------------------------------------------------------------------------
import ast
import json
import math
import re
import textwrap

import dataiku
import numpy as np
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
# Chunk fields shown above a located hit. Everything else is body text,
# which the matched extract already covers.
SHOW_META = {"file name", "document title", "geography scope", "section",
             "effective date", "version", "owner", "applicability"}

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


# **Key:**value  — the colon sits INSIDE the asterisks, and a value runs
# until the next key marker or the end of the chunk. So values may span
# lines and contain commas, colons and pipe tables.
KV_RE = re.compile(r"\*\*\s*([^*:\n]{1,80}?)\s*:\s*\*\*")

# Chunks are separated by commas, but values contain commas too, so a
# naive split corrupts them. Split on the repeated LEADING key instead —
# that marker only ever starts a chunk. Set this if auto-detection picks
# the wrong key.
CHUNK_SPLIT_KEY = None      # e.g. "File Name"


def parse_chunk_text(text):
    """One chunk -> ordered [(key, value)]."""
    text = str(text)
    out, last_key, last_end = [], None, 0
    for m in KV_RE.finditer(text):
        seg = text[last_end:m.start()].strip()
        if last_key is not None:
            out.append((last_key, seg))
        elif seg:
            out.append(("(preamble)", seg))
        last_key, last_end = m.group(1).strip(), m.end()
    tail = text[last_end:].strip()
    if last_key is not None:
        out.append((last_key, tail))
    elif tail:
        out.append(("(body)", tail))
    return out


def split_chunks(raw, split_key=None):
    """document_chunks -> list of chunk strings."""
    if raw is None or str(raw).strip() in ("", "[]", "null", "None", "nan"):
        return []
    s = str(raw).strip()

    for loader in (json.loads, ast.literal_eval):
        try:
            v = loader(s)
            if isinstance(v, list):
                return [x if isinstance(x, str) else json.dumps(x) for x in v]
            if isinstance(v, (str, dict)):
                return [v if isinstance(v, str) else json.dumps(v)]
        except Exception:
            pass

    key = split_key or CHUNK_SPLIT_KEY
    if not key:
        m = KV_RE.search(s)
        key = m.group(1).strip() if m else None
    if not key:
        return [s]

    marker = re.compile(r"\*\*\s*" + re.escape(key) + r"\s*:\s*\*\*")
    starts = [m.start() for m in marker.finditer(s)]
    if len(starts) <= 1:
        return [s]
    bounds = starts + [len(s)]
    out = []
    for a, b in zip(bounds, bounds[1:]):
        piece = s[a:b].strip().rstrip(",").strip()
        if piece:
            out.append(piece)
    return out


def parse_chunks(raw):
    """document_chunks -> [{doc_title, kv, text}], one entry per chunk."""
    out = []
    for text in split_chunks(raw):
        kv = parse_chunk_text(text)
        d  = {k.lower(): v for k, v in kv}
        title = (next((v for k, v in d.items() if "document title" in k), None)
                 or next((v for k, v in d.items() if "file name" in k), None)
                 or "?")
        # Search the parsed VALUES, not the raw text: otherwise the
        # **Key:** markers end up inside the matched extract, and with no
        # newlines between pairs the whole chunk reads as one sentence.
        sents = []
        for k, v in kv:
            for sent in _sentences(v):
                sents.append((k, sent))
        out.append({"doc_title": re.sub(r"\*+", "", str(title)).strip()[:80],
                    "kv": kv, "text": text, "sentences": sents})
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

    sent_lists = [c.get("sentences") or [("", x) for x in
                                          _sentences(c["text"])]
                  for c in chunks]
    # document frequency across sentences, for rare-term weighting
    dfreq, nsent = {}, sum(len(x) for x in sent_lists) or 1
    for sl in sent_lists:
        for _, sent in sl:
            for t in _toks(sent):
                dfreq[t] = dfreq.get(t, 0) + 1
    weight = {t: math.log(1 + nsent / (1 + dfreq.get(t, 0))) for t in qt}
    total_w = sum(weight.values()) or 1.0
    qnums = set(_NUM.findall(str(quote)))

    hits = []
    for ci, sents in enumerate(sent_lists):
        for si, (field, sent) in enumerate(sents):
            shared = qt & _toks(sent)
            if not shared:
                continue
            score = sum(weight[t] for t in shared) / total_w
            # numbers are high-precision anchors
            if qnums and qnums & set(_NUM.findall(sent)):
                score = min(1.0, score + 0.25)
            # keep the context window inside the SAME field, or a hit in
            # Content picks up the tail of Geography Scope
            lo, hi = si, si + 1
            while lo > 0 and si - lo < context and sents[lo - 1][0] == field:
                lo -= 1
            while (hi < len(sents) and hi - si <= context
                   and sents[hi][0] == field):
                hi += 1
            hits.append({"chunk": ci + 1,
                         "doc_title": chunks[ci]["doc_title"],
                         "field": field,
                         "score": round(min(score, 1.0), 2),
                         "sentence": sent,
                         "context": " ".join(x for _, x in sents[lo:hi])})
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
                    print("    >> NOT FOUND in any chunk. The quote was "
                          "paraphrased or invented — treat as unverified.")
                else:
                    h   = hits[0]
                    ci  = h["chunk"] - 1
                    tag = "" if h["score"] >= 0.8 else "  (partial match)"
                    print(f"    FOUND in chunk [{h['chunk']}] of "
                          f"{len(chunks)}   match {h['score']}{tag}")
                    # ONLY this chunk's fields — never another chunk's
                    for k, v in chunks[ci]["kv"]:
                        if k.lower() in SHOW_META:
                            print(f"      {k}: {str(v)[:90]}")
                    fld = h.get("field") or "content"
                    print(f"      matched in field '{fld}':")
                    for line in textwrap.wrap(h["context"], 84):
                        print(f"        {line}")


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
# matplotlib, not plotly: static images need no JavaScript, no CDN and no
# nbformat. A blocked CDN is the usual reason plotly renders blank inside a
# corporate network.
#
# Palette is the validated categorical set (worst adjacent CVD dE 9.1,
# normal-vision dE 19.6). Its mid-lightness hues sit under 3:1 against the
# surface, so every bar carries a direct value label.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CHART_DIR = None      # e.g. "/tmp/audit_charts" to also save PNGs
CHART_DPI = 110

# categorical: fixed order, never cycled
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK      = "#0b0b0b"      # primary text
INK_SOFT = "#52514e"      # secondary text
GRID     = "#e4e3df"
SURFACE  = "#fcfcfb"
OK_C     = "#1baf7a"      # status: expected / good
BAD_C    = "#e34948"      # status: defect
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT,
    "text.color": INK, "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 14,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.autolayout": False,
})


def _finish(fig, ax, name, xlabel=None, grid_axis="x", grid=True):
    """Recessive grid, soft spines, save, show.

    grid=False for heatmaps: a grid drawn over image cells reads as
    hairlines through the data.
    """
    if grid:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SOFT)
    fig.tight_layout()
    if CHART_DIR:
        import os
        os.makedirs(CHART_DIR, exist_ok=True)
        fig.savefig(f"{CHART_DIR}/{name}.png", dpi=CHART_DPI,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.show()


def _label_barh(ax, bars, values, fmt="{:,.0f}"):
    """Direct value labels. Obligatory: these hues fall under 3:1."""
    for b, v in zip(bars, values):
        if v <= 0:
            continue
        ax.text(b.get_width() + max(values) * 0.015,
                b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=9, color=INK)


def _driver_frame():
    out = []
    for r, f in zip(rows, facts):
        for factor, flags in f.items():
            for fl in flags:
                out.append({"Factor": factor, "Driver": fl,
                            "Strength": "strong" if fl in STRONG else "weak",
                            "tid": r.get("transaction_id")})
    return pd.DataFrame(out)


def build_charts():
    nfull = max(len(audit_full), 1)

    def mask(col):
        c = _col(col)
        return (audit_full[c].map(_val).eq(TRUE) if c
                else pd.Series(False, index=audit_full.index))

    m_rev, m_samp = mask("needs_review"), mask("in_sample_queue")
    m_exp = mask("expected_behavior")

    # expected_behavior may arrive blank after a round-trip through the
    # store. The pipeline sets themes = [] whenever a row is expected, so
    # a BLANK THEME is the surviving signal — not "wasn't flagged".
    # The two differ: a themed row that stayed below the review bar is
    # not expected behaviour, it is an unraised defect.
    exp_derived = not m_exp.any()
    if exp_derived:
        ec = _col("expected_behavior")
        vals = (dict(audit_full[ec].map(_val).value_counts()) if ec
                else "column absent")
        tc = _col("primary_theme") or _col("themes")
        if tc:
            no_theme = (audit_full[tc].isna()
                        | audit_full[tc].astype(str).str.strip()
                          .isin(["", "nan", "None"]))
        else:
            no_theme = pd.Series(True, index=audit_full.index)
        m_exp = no_theme & ~m_rev
        print(f"\nnote: expected_behavior carries no TRUE values ({vals}).")
        print(f"      Deriving it from a blank theme + not flagged, which "
              f"is how the pipeline writes an expected row.")
        print(f"      derived cleared: {int(m_exp.sum())} of {nfull}")

    # four mutually exclusive buckets that sum to the total
    n_rev  = int(m_rev.sum())
    n_samp = int((m_samp & ~m_rev).sum())
    n_exp  = int((m_exp & ~m_rev & ~m_samp).sum())
    n_rest = max(nfull - n_exp - n_rev - n_samp, 0)

    exp_word = ("expected behaviour (derived from a blank theme)"
                if exp_derived else "confirmed expected behaviour")

    # --- 1. Headline: how much was expected -------------------------
    fig, ax = plt.subplots(figsize=(9, 1.6))
    ax.axis("off")
    ax.text(0, .72, f"{100 * n_exp / nfull:.0f}%", fontsize=46,
            fontweight="bold", color=OK_C, va="center")
    ax.text(.22, .80, f"of interactions were {exp_word}",
            fontsize=13, color=INK, va="center")
    ax.text(.22, .46, f"{nfull:,} audited   ·   {n_rev:,} flagged for review "
                      f"({100 * n_rev / nfull:.0f}%)   ·   {n_samp:,} control "
                      f"sample", fontsize=11, color=INK_SOFT, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _finish(fig, ax, "01_headline", grid_axis="y")

    # --- 2. Triage split --------------------------------------------
    lab = ["Expected behaviour" + (" (derived)" if exp_derived else ""),
           "Flagged for review", "Control sample",
           "Themed, below review bar"]
    val = [n_exp, n_rev, n_samp, n_rest]
    col = [OK_C, BAD_C, CAT[0], CAT[3]]
    # the 4th bucket is a real finding when non-zero: a theme was assigned
    # but nothing met the bar to raise it. Drawn only when it exists.
    keep = [i for i, v in enumerate(val) if v > 0 or i < 3]
    lab, val, col = ([lab[i] for i in keep], [val[i] for i in keep],
                     [col[i] for i in keep])
    fig, ax = plt.subplots(figsize=(9, 3.1))
    bars = ax.barh(lab, val, color=col, height=.62, zorder=3)
    _label_barh(ax, bars, val)
    for b, v in zip(bars, val):
        ax.text(b.get_width() * 0.5, b.get_y() + b.get_height() / 2,
                f"{100 * v / nfull:.0f}%", va="center", ha="center",
                fontsize=9, color="white", fontweight="bold") if v / nfull > .08 else None
    ax.invert_yaxis()
    ax.set_title(f"Triage split — {nfull:,} interactions")
    _finish(fig, ax, "02_triage", "interactions")

    # --- 3. Route x expected vs defect -------------------------------
    rc = _col("route") or "route"
    if rc in audit_full.columns:
        t = audit_full.copy()
        t["ok"] = m_exp.values     # same definition as the headline
        g = t.groupby(rc)["ok"].agg(["sum", "count"])
        g["defect"] = g["count"] - g["sum"]
        g = g.sort_values("count")
        fig, ax = plt.subplots(figsize=(9, max(3.2, .62 * len(g) + 2.1)))
        b1 = ax.barh(g.index, g["sum"], color=OK_C, height=.6,
                     label="expected", zorder=3,
                     edgecolor=SURFACE, linewidth=1.6)
        b2 = ax.barh(g.index, g["defect"], left=g["sum"], color=BAD_C,
                     height=.6, label="defect", zorder=3,
                     edgecolor=SURFACE, linewidth=1.6)
        for y, (s, c) in enumerate(zip(g["sum"], g["count"])):
            ax.text(c + c * .02, y, f"{100 * s / c:.0f}% ok",
                    va="center", fontsize=9, color=INK)
        ax.legend(frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(.5, -.22), fontsize=9)
        ax.set_title("Where the agent is reliable, by route")
        _finish(fig, ax, "03_route", "interactions")

    dd = _driver_frame()

    # --- 4. Factor counts -------------------------------------------
    fc = [(fa, int(audit[f"is_{fa}"].sum())) for fa in FACTORS
          if f"is_{fa}" in audit.columns]
    if fc:
        fc.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(9, 3.4))
        bars = ax.barh([a for a, _ in fc], [b for _, b in fc],
                       color=CAT[0], height=.6, zorder=3)
        _label_barh(ax, bars, [b for _, b in fc])
        ax.set_title(f"What the user experienced  "
                     f"(of {len(audit):,} analysed)")
        _finish(fig, ax, "04_factors", "interactions")

    # --- 5. Factor x driver -----------------------------------------
    if len(dd):
        piv = (dd.groupby(["Factor", "Driver"])["tid"].nunique()
                 .unstack(fill_value=0))
        piv = piv.loc[piv.sum(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(10, max(3, .66 * len(piv) + 1.8)))
        left = np.zeros(len(piv))
        for i, drv in enumerate(piv.columns):
            v = piv[drv].values
            ax.barh(piv.index, v, left=left, height=.6,
                    color=CAT[i % len(CAT)], label=drv, zorder=3,
                    edgecolor=SURFACE, linewidth=1.6)
            for y, (x0, x1) in enumerate(zip(left, v)):
                if x1 > 0:
                    ax.text(x0 + x1 / 2, y, f"{int(x1)}", va="center",
                            ha="center", fontsize=8, color="white",
                            fontweight="bold")
            left += v
        ax.legend(frameon=False, fontsize=8, ncol=2,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_title("What drove each factor")
        _finish(fig, ax, "05_factor_driver", "interactions")

        # --- 6. Evidence strength ----------------------------------
        sp = (dd.groupby(["Factor", "Strength"])["tid"].nunique()
                .unstack(fill_value=0))
        for c in ("strong", "weak"):
            if c not in sp.columns:
                sp[c] = 0
        sp = sp.loc[sp.sum(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(9, max(3.4, .62 * len(sp) + 2.2)))
        ax.barh(sp.index, sp["strong"], color="#1d5fa8", height=.6,
                label="strong — defensible in review", zorder=3,
                edgecolor=SURFACE, linewidth=1.6)
        ax.barh(sp.index, sp["weak"], left=sp["strong"], color="#9dc3ec",
                height=.6, label="weak — sample only", zorder=3,
                edgecolor=SURFACE, linewidth=1.6)
        for y, (s, w) in enumerate(zip(sp["strong"], sp["weak"])):
            if s: ax.text(s / 2, y, int(s), va="center", ha="center",
                          fontsize=8, color="white", fontweight="bold")
            if w: ax.text(s + w / 2, y, int(w), va="center", ha="center",
                          fontsize=8, color=INK)
        ax.legend(frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(.5, -.20), fontsize=9)
        ax.set_title("Evidence strength by factor")
        _finish(fig, ax, "06_strength", "interactions")

        # --- 7. Factor co-occurrence -------------------------------
        fl = [fa for fa in FACTORS if f"is_{fa}" in audit.columns]
        if len(fl) > 1:
            m = np.zeros((len(fl), len(fl)), int)
            for _, r in audit.iterrows():
                on = [i for i, fa in enumerate(fl) if r.get(f"is_{fa}")]
                for i in on:
                    for j in on:
                        m[i, j] += 1
            if m.sum():
                fig, ax = plt.subplots(figsize=(6.4, 5.4))
                im = ax.imshow(m, cmap=SEQ)
                ax.set_xticks(range(len(fl)), fl, rotation=35, ha="right")
                ax.set_yticks(range(len(fl)), fl)
                hi = m.max() or 1
                for i in range(len(fl)):
                    for j in range(len(fl)):
                        if m[i, j]:
                            ax.text(j, i, m[i, j], ha="center", va="center",
                                    fontsize=9,
                                    color="white" if m[i, j] > hi * .55 else INK)
                ax.set_title("Which factors occur together\n(diagonal = total)")
                fig.colorbar(im, ax=ax, shrink=.75, label="interactions")
                _finish(fig, ax, "07_cooccurrence", grid=False)

    # --- 8. Factor x cause heatmap -----------------------------------
    cc = _col("all_causes") or _col("cause")
    if len(dd) and cc:
        pr = []
        for r, f in zip(rows, facts):
            cs = [c.strip() for c in str(r.get(cc) or "").split(";")
                  if c.strip()] or ["(none assigned)"]
            for fa in f:
                for c in cs:
                    pr.append({"Factor": fa, "Cause": c})
        if pr:
            pv = pd.crosstab(pd.DataFrame(pr)["Factor"],
                             pd.DataFrame(pr)["Cause"])
            fig, ax = plt.subplots(
                figsize=(max(7, 1.5 + 1.25 * pv.shape[1]),
                         max(3, 1.4 + .62 * pv.shape[0])))
            im = ax.imshow(pv.values, cmap=SEQ, aspect="auto")
            ax.set_xticks(range(pv.shape[1]),
                          [c.replace("_", " ") for c in pv.columns],
                          rotation=32, ha="right", fontsize=9)
            ax.set_yticks(range(pv.shape[0]), pv.index, fontsize=9)
            hi = pv.values.max() or 1
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    v = pv.values[i, j]
                    if v:
                        ax.text(j, i, v, ha="center", va="center", fontsize=9,
                                color="white" if v > hi * .55 else INK)
            ax.set_title("Factor × root cause")
            ax.grid(False)
            fig.tight_layout()
            if CHART_DIR:
                import os
                os.makedirs(CHART_DIR, exist_ok=True)
                fig.savefig(f"{CHART_DIR}/08_factor_cause.png", dpi=CHART_DPI,
                            facecolor=SURFACE, bbox_inches="tight")
            plt.show()

    # --- 9. Owner workload -------------------------------------------
    oc = _col("cause_owners") or _col("cause_owner")
    if oc and oc in audit.columns:
        ow = {}
        for r in rows:
            for o in {x.strip() for x in str(r.get(oc) or "").split(";")
                      if x.strip()}:
                ow[o] = ow.get(o, 0) + 1
        if ow:
            k = sorted(ow, key=ow.get)
            fig, ax = plt.subplots(figsize=(9, max(2.6, .55 * len(k) + 1.4)))
            bars = ax.barh(k, [ow[x] for x in k], color=CAT[6],
                           height=.6, zorder=3)
            _label_barh(ax, bars, [ow[x] for x in k])
            ax.set_title("Who fixes what")
            _finish(fig, ax, "09_owners", "interactions")

    # --- 10. Weekly: bars, never a line ------------------------------
    # At this volume a trend line implies a signal that is not
    # distinguishable from noise. Bars with counts are the honest form.
    if DATE_COL in audit.columns and len(dd):
        w = audit.copy()
        w["_ts"] = pd.to_datetime(w[DATE_COL], errors="coerce")
        w = w[w["_ts"].notna()]
        if len(w):
            w["week"] = w["_ts"].dt.to_period("W-SUN").dt.start_time.dt.date
            wks = sorted(w["week"].unique())
            act = [fa for fa in FACTORS
                   if f"is_{fa}" in w.columns and w[f"is_{fa}"].any()]
            if wks and act:
                fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(wks) + 3), 4.3))
                bw = .8 / len(act)
                xs = np.arange(len(wks))
                for i, fa in enumerate(act):
                    v = [int(w[(w["week"] == wk) & w[f"is_{fa}"]].shape[0])
                         for wk in wks]
                    ax.bar(xs + i * bw - .4 + bw / 2, v, bw * .88,
                           color=CAT[i % len(CAT)], label=fa, zorder=3)
                    for x, y in zip(xs + i * bw - .4 + bw / 2, v):
                        if y:
                            ax.text(x, y + .1, y, ha="center", fontsize=8,
                                    color=INK)
                vol = [int((w["week"] == wk).sum()) for wk in wks]
                ax.set_xticks(xs, [f"{wk}\n(n={n})" for wk, n in zip(wks, vol)],
                              fontsize=8)
                ax.legend(frameon=False, ncol=min(len(act), 5), fontsize=9,
                          loc="upper center", bbox_to_anchor=(.5, -.16))
                ax.set_title("Weekly counts — bars, not a line: at this "
                             "volume a trend line would mislead")
                _finish(fig, ax, "10_weekly", grid_axis="y")


try:
    build_charts()
except Exception as _e:
    print(f"\ncharts skipped: {type(_e).__name__}: {_e}")
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
