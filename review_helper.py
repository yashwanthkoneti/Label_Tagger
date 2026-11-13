# Databricks notebook source
# Locate a flag's evidence inside the chunks — no LLM calls, no rerun.
#
# The judge quotes a fragment of a chunk. It does not say WHICH chunk or
# where, so verifying a contradiction means reading everything by eye.
# This finds the quote, prints the surrounding sentences, puts the answer
# side next to it, and highlights the values that actually differ.
#
#   from review_helper import review, review_all, attach_chunks
#
#   # The audit output does NOT carry document_chunks — they are huge and
#   # would bloat the CSV. Merge them back from the input traces first:
#   result = attach_chunks(result, INPUT_CSV)
#
#   review(result, "TXN1002")          # one row
#   review_all(result, limit=10)       # the whole review queue
#
# Or pass the traces each time and let it merge on the fly:
#   review(result, "TXN1002", traces=INPUT_CSV)
#
# It also catches a false positive you cannot see by reading: if the judge's
# chunk-side quote does not appear in the chunks at all, it was paraphrased
# or invented, and the finding should not stand.

import json
import re
import textwrap

TRUE, FALSE = "TRUE", "FALSE"

# Flags whose evidence points at chunk text worth locating
LOCATABLE = [
    "ANSWER_CONTRADICTS_CHUNKS",
    "ANSWER_MISREADS_TABLE",
    "ANSWER_HAS_UNSUPPORTED_CLAIM",
    "ANSWER_OMITS_CONDITION_OR_EXCEPTION",
    "CHUNKS_WRONG_APPLICABILITY",
    "CHUNKS_CONFLICT",
]

_WORD = re.compile(r"[A-Za-z][\w'-]+|\d+(?:[.,]\d+)?")
_NUM  = re.compile(r"\d+(?:[.,]\d+)?")
_STOP = {
    "the", "and", "for", "are", "was", "were", "this", "that", "with",
    "from", "which", "their", "there", "have", "has", "had", "not", "but",
    "any", "all", "may", "must", "shall", "will", "can", "should", "would",
    "policy", "policies", "document", "documents", "standard", "standards",
}


def _tokens(s):
    return {w.lower() for w in _WORD.findall(str(s))
            if w.lower() not in _STOP and len(w) > 2}


def _sentences(text):
    """Split on sentence ends and newlines; keep table rows whole."""
    out = []
    for line in str(text).split("\n"):
        if line.strip().startswith("|") or line.count("|") >= 2:
            out.append(line.strip())          # a table row is one unit
            continue
        for s in re.split(r"(?<=[.;:])\s+", line):
            if s.strip():
                out.append(s.strip())
    return out


def _parse_chunks(raw):
    """Same shape as the pipeline's parser, kept standalone on purpose."""
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
            txt   = str(item)
            title = "?"
            m = re.search(r"\*{0,2}Document Title:?\*{0,2}\s*(.+)", txt)
            if m:
                title = re.sub(r"\*+", "", m.group(1)).strip()
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
    strip = lambda s, w: re.sub(rf"^\s*{w}\s*:\s*", "", s.strip(), flags=re.I)
    return strip(a, r"(chunk|row|source)"), strip(b, r"(answer|response)")


def locate(quote, chunks, context=1, top=2):
    """Find where a quote lives. Returns list of hits, best first.

    Each hit: chunk index, doc title, the matched sentence, the sentences
    around it, and a 0-1 score for how much of the quote it covers.
    """
    qt = _tokens(quote)
    if not qt:
        return []
    hits = []
    for ci, ch in enumerate(chunks):
        sents = _sentences(ch["text"])
        for si, s in enumerate(sents):
            overlap = len(qt & _tokens(s))
            if not overlap:
                continue
            score = overlap / len(qt)           # how much of the quote is here
            lo, hi = max(0, si - context), min(len(sents), si + context + 1)
            hits.append({
                "chunk": ci + 1,
                "doc_title": ch["doc_title"],
                "score": round(score, 2),
                "sentence": s,
                "context": " ".join(sents[lo:hi]),
            })
    hits.sort(key=lambda h: -h["score"])
    return hits[:top]


def differing_numbers(a, b):
    """Numbers present on one side but not the other — usually the conflict."""
    na, nb = set(_NUM.findall(str(a))), set(_NUM.findall(str(b)))
    return sorted(na - nb), sorted(nb - na)


def _wrap(s, width=96, indent="      "):
    return textwrap.fill(str(s), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def attach_chunks(df, traces):
    """Merge document_chunks back onto the audit output.

    The audit output deliberately omits the chunks — they are large and the
    CSV would be unusable. They live in the input traces, keyed by
    transaction_id, so joining them back is all that is needed.

    traces: path to the input CSV, or a DataFrame of it.
    """
    import pandas as pd

    if "document_chunks" in df.columns:
        return df

    src = pd.read_csv(traces) if isinstance(traces, str) else traces
    need = {"transaction_id", "document_chunks"}
    missing = need - set(src.columns)
    if missing:
        raise ValueError(f"traces is missing {missing}")

    out = df.copy()
    out["transaction_id"] = out["transaction_id"].astype(str)
    lookup = (src.assign(transaction_id=src["transaction_id"].astype(str))
                 .drop_duplicates("transaction_id")
                 .set_index("transaction_id")["document_chunks"])
    out["document_chunks"] = out["transaction_id"].map(lookup)

    matched = out["document_chunks"].notna().sum()
    if matched < len(out):
        print(f"note: chunks found for {matched}/{len(out)} rows "
              f"(the rest had no chunks, or the ids do not line up)")
    return out


def _ensure_chunks(df, traces):
    """Return df with chunks, or raise with a clear instruction."""
    if "document_chunks" in df.columns:
        return df
    if traces is not None:
        return attach_chunks(df, traces)
    raise ValueError(
        "This dataframe has no 'document_chunks' column.\n"
        "The audit output omits chunks on purpose (they are large).\n"
        "Merge them back from the input traces first:\n"
        "    result = attach_chunks(result, INPUT_CSV)\n"
        "or pass them per call:\n"
        "    review(result, '<txn>', traces=INPUT_CSV)")


def review(df, transaction_id, context=1, traces=None):
    """Print a review card for one interaction."""
    df = _ensure_chunks(df, traces)
    rows = df[df["transaction_id"].astype(str) == str(transaction_id)]
    if not len(rows):
        print(f"no row with transaction_id {transaction_id}")
        return
    r = rows.iloc[0]

    chunks = _parse_chunks(r.get("document_chunks"))
    try:
        ev_all = json.loads(r.get("evidence_json") or "{}")
    except Exception:
        ev_all = {}

    print("=" * 100)
    print(f"{r['transaction_id']}   route={r.get('route')}   "
          f"tier={r.get('tier')}   factors={r.get('quality_factors') or '-'}")
    print("-" * 100)
    print("REQUEST")
    print(_wrap(r.get("request")))
    print("\nANSWER")
    print(_wrap(r.get("response")))
    print(f"\nCHUNKS: {len(chunks)} retrieved, "
          f"{sum(len(c['text']) for c in chunks):,} chars total")
    for i, c in enumerate(chunks):
        print(f"   [{i+1}] {c['doc_title']}  ({len(c['text']):,} chars)")

    raised = [k for k in LOCATABLE if str(r.get(f"flag_{k}")) == TRUE]
    if not raised:
        print("\n(no chunk-based flag raised on this row)")
        print("=" * 100)
        return

    for flag in raised:
        ev = ev_all.get(flag, "")
        chunk_side, answer_side = split_evidence(ev)
        print("\n" + "-" * 100)
        print(f"FLAG  {flag}")
        print(f"  judge quoted (chunk side) : {chunk_side or '(none)'}")
        print(f"  judge quoted (answer side): {answer_side or '(none)'}")

        hits = locate(chunk_side, chunks, context=context)
        if not hits:
            print("\n  >> NOT FOUND in any chunk.")
            print("     The quote was paraphrased or invented. Treat this "
                  "finding as unverified.")
        else:
            best = hits[0]
            if best["score"] < 0.4:
                print(f"\n  >> WEAK MATCH ({best['score']}). The judge's quote "
                      f"only loosely resembles the chunk text.")
            print(f"\n  FOUND IN chunk [{best['chunk']}] "
                  f"{best['doc_title']}   (match {best['score']})")
            print(_wrap(best["context"], indent="      "))
            for h in hits[1:]:
                print(f"\n  also in chunk [{h['chunk']}] {h['doc_title']} "
                      f"(match {h['score']})")
                print(_wrap(h["sentence"], indent="      "))

        only_chunk, only_answer = differing_numbers(
            best["context"] if hits else chunk_side, r.get("response"))
        if only_chunk or only_answer:
            print(f"\n  NUMBERS  in chunk only : {only_chunk or '-'}")
            print(f"           in answer only: {only_answer or '-'}")
            print("           (if these are the same quantity, that is "
                  "the contradiction)")

    print("=" * 100)


def review_all(df, limit=None, only_flag=None, context=1, traces=None):
    """Review every flagged row, or just those carrying one flag."""
    df = _ensure_chunks(df, traces)
    q = df[df["needs_review"]] if "needs_review" in df else df
    if only_flag:
        q = q[q[f"flag_{only_flag}"].astype(str) == TRUE]
    if limit:
        q = q.head(limit)
    print(f"{len(q)} interactions to review\n")
    for tid in q["transaction_id"]:
        review(df, tid, context=context)   # chunks already attached
        print()


def unverified_findings(df, traces=None):
    """Rows where the judge's chunk quote cannot be found in the chunks.

    These are false positives you cannot spot by reading the answer — the
    judge quoted something the source does not contain.
    """
    df = _ensure_chunks(df, traces)
    out = []
    q = df[df["needs_review"]] if "needs_review" in df else df
    for r in q.to_dict("records"):
        chunks = _parse_chunks(r.get("document_chunks"))
        if not chunks:
            continue
        try:
            ev_all = json.loads(r.get("evidence_json") or "{}")
        except Exception:
            continue
        for flag in LOCATABLE:
            if str(r.get(f"flag_{flag}")) != TRUE:
                continue
            chunk_side, _ = split_evidence(ev_all.get(flag, ""))
            hits = locate(chunk_side, chunks, top=1)
            score = hits[0]["score"] if hits else 0.0
            if score < 0.4:
                out.append({
                    "transaction_id": r.get("transaction_id"),
                    "flag": flag,
                    "match_score": score,
                    "judge_quote": chunk_side[:120],
                    "verdict": ("quote not in chunks" if score == 0
                                else "weak match"),
                })
    import pandas as pd
    return pd.DataFrame(out)
