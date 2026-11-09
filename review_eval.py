# Databricks notebook source
# Build a condensed, shareable workbook from the audit output.
#
# Run after run_audit(), or load a finished run:
#     result = pd.read_csv(OUTPUT_CSV)
#
# Produces one .xlsx with six sheets, each for a different reader.
# Nobody has to look at the 60-column raw output.

import pandas as pd

SHARE_PATH = OUTPUT_CSV.replace(".csv", "_shareable.xlsx")

TOTAL = len(result)
pct   = lambda n: round(100 * n / max(TOTAL, 1), 1)


def col(name, default=""):
    """Tolerate a column being absent so this never crashes mid-presentation."""
    return result[name] if name in result.columns else pd.Series(
        [default] * len(result), index=result.index)


# --------------------------------------------------------------------------
# SHEET 1 — Summary. The numbers you say out loud.
# --------------------------------------------------------------------------
summary = pd.DataFrame([
    ("Interactions audited",        TOTAL,                                    ""),
    ("Flagged for review",          int(col("needs_review").sum()),
     f"{pct(int(col('needs_review').sum()))}%"),
    ("Sampled as control",          int(col("in_sample_queue").sum()),
     f"{pct(int(col('in_sample_queue').sum()))}%"),
    ("Confirmed expected behaviour", int(col("expected_behavior").sum()),
     f"{pct(int(col('expected_behavior').sum()))}%"),
    ("Audit error rate",            "",
     f"{100 * (col('errors').fillna('') != '').mean():.1f}%"),
    ("", "", ""),
    ("Read me", "Flagged rows carry the evidence that triggered them. "
                "Rates are directional until the judge is calibrated "
                "against human labels.", ""),
], columns=["Metric", "Count", "Share"])


# --------------------------------------------------------------------------
# SHEET 2 — Review queue. The cases worth a human look.
# --------------------------------------------------------------------------
REVIEW_COLS = [
    ("transaction_id", "Transaction"),
    ("request",        "User request"),
    ("route",          "What the agent did"),
    ("themes",         "Symptom"),
    ("cause",          "Root cause"),
    ("cause_owner",    "Owning component"),
    ("why_flagged",    "Evidence"),
]
rq = result[result["needs_review"]] if "needs_review" in result else result.head(0)
review = rq[[c for c, _ in REVIEW_COLS if c in rq.columns]].copy()
review.columns = [n for c, n in REVIEW_COLS if c in rq.columns]
review = review.sort_values("Owning component" if "Owning component"
                            in review.columns else review.columns[0])


# --------------------------------------------------------------------------
# SHEET 3 — By component. Who fixes what.
# --------------------------------------------------------------------------
themed = result[col("themes").fillna("") != ""].copy()
if len(themed):
    themed["Symptom"] = themed["themes"].str.split("; ").str[0]
    by_component = (themed.groupby("cause_owner")
        .agg(Interactions=("transaction_id", "count"),
             Symptoms=("Symptom", lambda s: ", ".join(sorted(set(s)))),
             Example=("request", "first"))
        .sort_values("Interactions", ascending=False)
        .reset_index()
        .rename(columns={"cause_owner": "Owning component"}))
    by_component["Share of all"] = by_component["Interactions"].map(
        lambda n: f"{pct(n)}%")
else:
    by_component = pd.DataFrame(
        columns=["Owning component", "Interactions", "Symptoms", "Example"])


# --------------------------------------------------------------------------
# SHEET 4 — Symptom x cause. The analysis table.
# --------------------------------------------------------------------------
if len(themed):
    symptom_cause = pd.crosstab(themed["Symptom"], themed["cause"],
                                margins=True, margins_name="TOTAL")
else:
    symptom_cause = pd.DataFrame()


# --------------------------------------------------------------------------
# SHEET 5 — Corpus backlog. Documents retrieval could not find.
# --------------------------------------------------------------------------
md = result[col("missing_document_titles").fillna("") != ""]
if len(md):
    corpus_backlog = (md["missing_document_titles"].str.split("; ")
        .explode().value_counts()
        .rename_axis("Candidate missing document")
        .reset_index(name="Times needed"))
    corpus_backlog["Caveat"] = ("matched on document title, not content "
                                "— verify before acting")
else:
    corpus_backlog = pd.DataFrame(
        columns=["Candidate missing document", "Times needed", "Caveat"])


# --------------------------------------------------------------------------
# SHEET 6 — Confirmed correct. Proof the audit is not flagging everything.
# --------------------------------------------------------------------------
exp = result[col("expected_behavior") == True]
if len(exp):
    confirmed = (exp["expected_reason"].value_counts()
        .rename_axis("Why it was correct")
        .reset_index(name="Interactions"))
    confirmed["Share of all"] = confirmed["Interactions"].map(
        lambda n: f"{pct(n)}%")
else:
    confirmed = pd.DataFrame(columns=["Why it was correct", "Interactions"])


# --------------------------------------------------------------------------
# WRITE
# --------------------------------------------------------------------------
sheets = {
    "Summary":         summary,
    "Review queue":    review,
    "By component":    by_component,
    "Symptom x cause": symptom_cause,
    "Corpus backlog":  corpus_backlog,
    "Confirmed correct": confirmed,
}

with pd.ExcelWriter(SHARE_PATH, engine="xlsxwriter") as xl:
    for name, frame in sheets.items():
        keep_index = name == "Symptom x cause"
        frame.to_excel(xl, sheet_name=name[:31], index=keep_index)
        ws = xl.sheets[name[:31]]
        # readable column widths; long text wraps rather than spilling
        wrap = xl.book.add_format({"text_wrap": True, "valign": "top"})
        for i, c in enumerate(frame.columns):
            longest = max([len(str(c))] +
                          [len(str(v)) for v in frame[c].head(200)])
            ws.set_column(i + (1 if keep_index else 0),
                          i + (1 if keep_index else 0),
                          min(max(longest + 2, 12), 60), wrap)
        ws.freeze_panes(1, 0)

print(f"wrote {SHARE_PATH}")
for name, frame in sheets.items():
    print(f"  {name:<20s} {len(frame):>4} rows")
