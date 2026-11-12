# Databricks notebook source
# Recompute root causes from an existing audit run — NO LLM calls.
#
# The pipeline assigns one primary cause using "earliest failing stage wins".
# That is right for routing a ticket, but it hides the other things that were
# also wrong on the same row. This re-derives EVERY cause that holds, from the
# flag_* columns already in the output.
#
# Also applies the clarification fix (clarification causes were unreachable
# behind the guardrail block) without needing a rerun.
#
#   result = pd.read_csv(OUTPUT_CSV)
#   result = expand_causes(result)

import pandas as pd

TRUE, FALSE, UNKNOWN, NA = "TRUE", "FALSE", "UNKNOWN", "NOT_APPLICABLE"

STAGE_ORDER = ["request_understanding", "context_use", "retrieval",
               "decision", "generation"]

CAUSE_OWNER = {
    "intent_misread_action_detector":      "intent detection",
    "intent_misread":                      "intent detection / prompt",
    "ambiguity_missed":                    "prompt / routing",
    "clarified_on_stale_context":          "conversation memory",
    "context_continuity_failure":          "conversation memory",
    "retrieval_coverage_gap":              "corpus owners",
    "retrieval_precision_miss":            "retriever / ranking",
    "retrieval_applicability_miss":        "retriever / metadata",
    "retrieval_partial_coverage":          "corpus owners",
    "retrieval_conflict":                  "corpus owners",
    "acknowledged_limitation_correctly":   "corpus owners",
    "guardrail_blocked_answerable":        "guardrail / scope definition",
    "guardrail_refused_policy_question":   "guardrail / scope definition",
    "scope_definition_too_narrow":         "guardrail / scope definition",
    "refused_answerable_generally":        "guardrail / prompt",
    "clarified_though_chunks_resolved_it": "routing prompt",
    "clarified_though_request_was_clear":  "routing prompt",
    "no_next_step_in_template":            "static response design",
    "prompt_or_routing":                   "prompt / routing",
    "generation_or_communication":         "response prompt",
    "answer_misread_table":                "response prompt",
    "answer_contradicts_source":           "response prompt",
    "answer_incomplete":                   "response prompt / retriever",
}


def all_causes_for(r):
    """Every cause that holds for this row, not just the earliest.

    r is a dict-like row with flag_* keys. Returns a list of
    (stage, cause) pairs in stage order.
    """
    def f(name):
        return r.get(f"flag_{name}")

    themes = str(r.get("themes") or "")
    route  = str(r.get("route") or "")
    cat    = r.get("scope_category")
    out    = []

    # ---- 1. request understanding ----------------------------------
    if (f("ACTION_GUARDRAIL_FALSE_BLOCK") == TRUE
            and f("REQUEST_IS_PROCEDURAL") == TRUE
            and f("REQUEST_IS_AMBIGUOUS") != TRUE):
        out.append(("request_understanding", "intent_misread_action_detector"))
    if "intent_misinterpretation" in themes:
        out.append(("request_understanding", "intent_misread"))
    if "ambiguous_user_context" in themes:
        out.append(("request_understanding", "ambiguity_missed"))
    if (route == "clarified" and f("REQUEST_IS_TOPIC_CHANGE") == TRUE):
        out.append(("request_understanding", "clarified_on_stale_context"))

    # ---- 2. context use --------------------------------------------
    if (f("HISTORY_HAS_NEEDED_INFO") == TRUE
            and route in ("clarified", "refused_scope", "refused_action")):
        out.append(("context_use", "context_continuity_failure"))

    # ---- 3. retrieval ----------------------------------------------
    # Only counts when chunks were actually retrieved and something about
    # them was wrong. A clarification is never caused by retrieval alone —
    # it fires because a source was CLEAR — so that case is excluded.
    chunks_bad = (
        f("RETRIEVAL_RETURNED_CHUNKS") == TRUE
        and (f("AUDIT_INPUT_LIMITED") == TRUE
             or f("CHUNKS_ON_TOPIC") == FALSE
             or f("CHUNKS_PARTIAL_COVERAGE") == TRUE))
    clarification_only = (
        "unnecessary_clarification" in themes
        and not any(t in themes for t in ("partial_answer",
                                          "intent_misinterpretation",
                                          "excessive_guardrail")))
    if chunks_bad and not clarification_only:
        if f("CHUNKS_WRONG_APPLICABILITY") == TRUE:
            out.append(("retrieval", "retrieval_applicability_miss"))
        if f("CHUNKS_CONFLICT") == TRUE:
            out.append(("retrieval", "retrieval_conflict"))
        if f("CHUNKS_PARTIAL_COVERAGE") == TRUE:
            out.append(("retrieval", "retrieval_partial_coverage"))
        if f("CORPUS_HAS_ANSWER") == TRUE:
            out.append(("retrieval", "retrieval_precision_miss"))
        elif f("CORPUS_HAS_ANSWER") == FALSE:
            out.append(("retrieval", "retrieval_coverage_gap"))
    if (f("AUDIT_INPUT_LIMITED") == TRUE
            and f("ANSWER_ACKNOWLEDGES_LIMITATION") == TRUE):
        out.append(("retrieval", "acknowledged_limitation_correctly"))

    # ---- 4. decision ------------------------------------------------
    if "unnecessary_clarification" in themes:
        if f("CHUNKS_RESOLVE_AMBIGUITY") == TRUE:
            out.append(("decision", "clarified_though_chunks_resolved_it"))
        elif f("REQUEST_IS_TOPIC_CHANGE") != TRUE:
            out.append(("decision", "clarified_though_request_was_clear"))

    if "excessive_guardrail" in themes:
        if f("CORPUS_HAS_ANSWER") == TRUE:
            out.append(("decision", "guardrail_blocked_answerable"))
        if f("REQUEST_IS_DOCUMENT_LOCATION") == TRUE:
            out.append(("decision", "guardrail_refused_policy_question"))
        if cat == "adjacent":
            out.append(("decision", "scope_definition_too_narrow"))
        if cat == "in_scope_user_specific":
            out.append(("decision", "refused_answerable_generally"))
        if not any(c for _, c in out if c.startswith(
                ("guardrail_", "scope_definition", "refused_answerable"))):
            out.append(("decision", "prompt_or_routing"))

    if "incomplete_next_step_guidance" in themes:
        out.append(("decision", "no_next_step_in_template"))

    # ---- 5. generation ----------------------------------------------
    if f("ANSWER_MISREADS_TABLE") == TRUE:
        out.append(("generation", "answer_misread_table"))
    if f("ANSWER_CONTRADICTS_CHUNKS") == TRUE:
        out.append(("generation", "answer_contradicts_source"))
    if f("ANSWER_COVERS_ALL_PARTS") == FALSE:
        out.append(("generation", "answer_incomplete"))
    if "communication_quality" in themes and not out:
        out.append(("generation", "generation_or_communication"))

    # de-duplicate, keep stage order
    seen, ordered = set(), []
    for stage in STAGE_ORDER:
        for s, c in out:
            if s == stage and c not in seen:
                seen.add(c)
                ordered.append((s, c))
    return ordered


def expand_causes(df):
    """Add all_causes / cause_stages / cause_owners / cause_count columns.

    primary_cause is the earliest stage, matching the pipeline's own choice,
    so existing views keep working.
    """
    rows = df.to_dict("records")
    allc = [all_causes_for(r) for r in rows]

    df = df.copy()
    df["all_causes"]    = ["; ".join(c for _, c in cs) for cs in allc]
    df["cause_stages"]  = ["; ".join(sorted({s for s, _ in cs},
                                            key=STAGE_ORDER.index))
                           for cs in allc]
    df["cause_owners"]  = ["; ".join(sorted({CAUSE_OWNER.get(c, "manual review")
                                             for _, c in cs}))
                           for cs in allc]
    df["cause_count"]   = [len(cs) for cs in allc]
    df["primary_cause"] = [cs[0][1] if cs else None for cs in allc]
    df["primary_stage"] = [cs[0][0] if cs else None for cs in allc]

    # did the pipeline's single cause differ from the recomputed primary?
    if "cause" in df.columns:
        df["cause_changed"] = (df["cause"].fillna("")
                               != df["primary_cause"].fillna(""))
    return df


def cause_report(df):
    """Counts by cause, counted once per occurrence rather than per row."""
    exploded = (df["all_causes"].fillna("").str.split("; ")
                  .explode().str.strip())
    exploded = exploded[exploded != ""]
    rep = exploded.value_counts().rename_axis("Cause").reset_index(
        name="Interactions")
    rep["Stage"] = rep["Cause"].map(
        lambda c: next((s for s, cc in
                        [(s, cc) for row in df.to_dict("records")
                         for s, cc in all_causes_for(row)] if cc == c), ""))
    rep["Owner"] = rep["Cause"].map(
        lambda c: CAUSE_OWNER.get(c, "manual review"))
    return rep[["Cause", "Stage", "Owner", "Interactions"]]


def multi_cause_rows(df, min_causes=2):
    """Rows where more than one thing went wrong — the interesting ones."""
    cols = [c for c in ["transaction_id", "request", "route", "themes",
                        "cause_count", "all_causes", "cause_owners"]
            if c in df.columns]
    return (df[df["cause_count"] >= min_causes][cols]
            .sort_values("cause_count", ascending=False))
