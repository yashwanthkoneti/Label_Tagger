# Databricks notebook source
# User-perceived quality factors, derived from existing flags — NO LLM calls.
#
# The pipeline themes describe what the SYSTEM did wrong (a routing defect,
# a retrieval gap). These five factors describe what the USER experienced.
# Same rows, different lens — and the language a feedback form or a business
# stakeholder actually uses.
#
#   result = pd.read_csv(OUTPUT_CSV)
#   result = add_quality_factors(result)
#   print(factor_report(result).to_string(index=False))
#
# A row can carry several factors. They are not mutually exclusive.
#
# STRENGTH: "strong" means the factor rests on at least one flag that had to
# quote evidence on both sides to survive the code-level evidence gate.
# "weak" means it rests only on flags known to over-fire. Weak factors are
# reported but should not drive a ticket.

import pandas as pd

TRUE, FALSE, UNKNOWN, NA = "TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"

FACTORS = ["Incorrect", "Incomplete", "Irrelevant",
           "Inconsistent", "Unsubstantiated"]

FACTOR_MEANING = {
    "Incorrect":       "The answer is wrong",
    "Incomplete":      "The answer is missing important information",
    "Irrelevant":      "The answer does not address the question",
    "Inconsistent":    "The answer contradicts other information or earlier responses",
    "Unsubstantiated": "The answer is not supported by appropriate references",
}

# Flags that survived an evidence gate requiring a two-sided quote.
# A factor resting on one of these is defensible in review.
STRONG_FLAGS = {
    "ANSWER_CONTRADICTS_CHUNKS",
    "ANSWER_MISREADS_TABLE",
    "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY",
    "GUARDRAIL_FALSE_BLOCK",
    "ACTION_GUARDRAIL_FALSE_BLOCK",
    "ANSWERED_ELSEWHERE",
    "ANSWER_ADDRESSES_QUESTION",
    "ANSWER_COVERS_ALL_PARTS",
    "CHUNKS_CONFLICT",
}


def factors_for(r):
    """Every quality factor that holds for this row.

    Returns a list of (factor, strength, [driver flags]).
    """
    def f(name):
        return r.get(f"flag_{name}")

    out = []

    # These two read the opposite way: FALSE is the defect, TRUE is healthy.
    # They must be excluded from the generic TRUE check, or a clean row
    # scores as defective on both.
    INVERTED = {"ANSWER_ADDRESSES_QUESTION", "ANSWER_COVERS_ALL_PARTS"}

    def add(factor, drivers):
        hits = []
        for d in drivers:
            want = FALSE if d in INVERTED else TRUE
            if f(d) == want:
                hits.append(d)
        hits = sorted(set(hits))
        if hits:
            strength = ("strong" if any(h in STRONG_FLAGS for h in hits)
                        else "weak")
            out.append((factor, strength, hits))

    # ---- Incorrect: a statement that is demonstrably wrong ------------
    # A personal eligibility ruling belongs here: the agent decided
    # something about the user it was not entitled to decide, so the
    # answer may simply be wrong for this person.
    add("Incorrect", [
        "ANSWER_CONTRADICTS_CHUNKS",
        "ANSWER_MISREADS_TABLE",
        "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY",
    ])

    # ---- Incomplete: something material is missing --------------------
    # AUDIT_INPUT_LIMITED only counts when the agent did NOT say so.
    # An answer that honestly flags the gap is not an incomplete failure.
    incomplete = ["ANSWER_COVERS_ALL_PARTS",
                  "ANSWER_OMITS_CONDITION_OR_EXCEPTION",
                  "CHUNKS_PARTIAL_COVERAGE"]
    if (f("AUDIT_INPUT_LIMITED") == TRUE
            and f("ANSWER_ACKNOWLEDGES_LIMITATION") != TRUE):
        incomplete.append("AUDIT_INPUT_LIMITED")
    add("Incomplete", incomplete)

    # ---- Irrelevant: the user did not get their question answered -----
    # A refusal of an answerable question reads as irrelevant from the
    # user's seat, whatever the system's reason was. Same for getting a
    # clarifying question back when the request was already clear.
    add("Irrelevant", [
        "ANSWER_ADDRESSES_QUESTION",
        "GUARDRAIL_FALSE_BLOCK",
        "ACTION_GUARDRAIL_FALSE_BLOCK",
        "CLARIFICATION_UNNECESSARY",
    ])

    # ---- Inconsistent: conflicts with other information ---------------
    # ANSWERED_ELSEWHERE is the cleanest signal here: the same question
    # was answered in one interaction and refused in another.
    # HISTORY_HAS_NEEDED_INFO means the response ignored what the
    # conversation had already established.
    add("Inconsistent", [
        "ANSWERED_ELSEWHERE",
        "CHUNKS_CONFLICT",
        "CHUNKS_WRONG_APPLICABILITY",
        "HISTORY_HAS_NEEDED_INFO",
    ])

    # ---- Unsubstantiated: stated without support ----------------------
    add("Unsubstantiated", [
        "ANSWER_HAS_UNSUPPORTED_CLAIM",
        "ANSWER_NUMBERS_UNSUPPORTED",
        "ANSWERED_WITH_SILENT_ASSUMPTION",
    ])

    return [(fa, st, dr) for fa in FACTORS
            for (f2, st, dr) in out if f2 == fa]


def add_quality_factors(df):
    """Add quality_factors / _strong / factor_count / factor_evidence."""
    rows = df.to_dict("records")
    allf = [factors_for(r) for r in rows]

    df = df.copy()
    df["quality_factors"] = ["; ".join(fa for fa, _, _ in fs) for fs in allf]
    df["quality_factors_strong"] = [
        "; ".join(fa for fa, st, _ in fs if st == "strong") for fs in allf]
    df["factor_count"] = [len(fs) for fs in allf]
    df["factor_evidence"] = [
        " | ".join(f"{fa} ({st}): {', '.join(dr)}" for fa, st, dr in fs)
        for fs in allf]
    df["primary_factor"] = [
        next((fa for fa, st, _ in fs if st == "strong"), None)
        or (fs[0][0] if fs else None) for fs in allf]
    return df


def factor_report(df):
    """One row per factor, with how many rest on strong evidence."""
    rows = df.to_dict("records")
    counts, strong = {fa: 0 for fa in FACTORS}, {fa: 0 for fa in FACTORS}
    for r in rows:
        for fa, st, _ in factors_for(r):
            counts[fa] += 1
            if st == "strong":
                strong[fa] += 1
    n = max(len(df), 1)
    return pd.DataFrame([{
        "Factor": fa,
        "What the user experienced": FACTOR_MEANING[fa],
        "Interactions": counts[fa],
        "Share": f"{100 * counts[fa] / n:.1f}%",
        "Of which strong evidence": strong[fa],
    } for fa in FACTORS])


def factor_by_cause(df, cause_col="all_causes"):
    """Factor x root cause: what the user felt vs what the system did.

    Needs expand_root_causes.expand_causes() run first for all_causes;
    falls back to the single 'cause' column.
    """
    col = cause_col if cause_col in df.columns else "cause"
    pairs = []
    for r in df.to_dict("records"):
        causes = [c.strip() for c in str(r.get(col) or "").split(";")
                  if c.strip()]
        for fa, _, _ in factors_for(r):
            for c in causes or ["(no cause assigned)"]:
                pairs.append({"Factor": fa, "Cause": c})
    if not pairs:
        return pd.DataFrame()
    p = pd.DataFrame(pairs)
    return pd.crosstab(p["Factor"], p["Cause"], margins=True,
                       margins_name="TOTAL")


def factor_rows(df, factor, strong_only=True):
    """The interactions behind one factor, for review."""
    keep = []
    for i, r in enumerate(df.to_dict("records")):
        for fa, st, dr in factors_for(r):
            if fa == factor and (not strong_only or st == "strong"):
                keep.append(i)
                break
    cols = [c for c in ["transaction_id", "request", "response", "route",
                        "quality_factors", "factor_evidence"]
            if c in df.columns]
    return df.iloc[keep][cols]
