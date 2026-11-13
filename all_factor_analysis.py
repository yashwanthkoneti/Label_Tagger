# -------------------------------------------------------------------------
# Dataiku — quality factor analysis
#
# Answers one question: for each factor the user would complain about,
# what drove it?
#
# Self-contained. No LLM calls. No imports beyond dataiku/pandas.
# -------------------------------------------------------------------------
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


# ---- FACTOR DEFINITIONS -------------------------------------------------
TRUE, FALSE = "TRUE", "FALSE"

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

# Flags that had to quote evidence on both sides to survive the audit's own
# checks. A factor resting on one of these is defensible in review.
STRONG = {"ANSWER_CONTRADICTS_CHUNKS", "ANSWER_MISREADS_TABLE",
          "ANSWER_ASSERTS_PERSONAL_ELIGIBILITY", "GUARDRAIL_FALSE_BLOCK",
          "ACTION_GUARDRAIL_FALSE_BLOCK", "ANSWERED_ELSEWHERE",
          "ANSWER_ADDRESSES_QUESTION", "ANSWER_COVERS_ALL_PARTS",
          "CHUNKS_CONFLICT"}


def factor_row(r):
    """Factors for one row, as {factor: [driver flags]}.

    Only populated when the audit did NOT clear the row as expected
    behaviour — a row judged correct cannot also be a user complaint.
    """
    if r.get("expected_behavior") in (True, "True", "true"):
        return {}

    def fires(flag):
        v = r.get(f"flag_{flag}")
        return v == (FALSE if flag in INVERTED else TRUE)

    out = {}
    for factor, flags in DRIVERS.items():
        hits = [f for f in flags if fires(f)]
        # an acknowledged corpus gap is honest behaviour, not Incomplete
        if (factor == "Incomplete" and "AUDIT_INPUT_LIMITED" in hits
                and r.get("flag_ANSWER_ACKNOWLEDGES_LIMITATION") == TRUE):
            hits.remove("AUDIT_INPUT_LIMITED")
        if hits:
            out[factor] = hits
    return out


rows = audit.to_dict("records")
facts = [factor_row(r) for r in rows]

audit["factors"] = ["; ".join(f) for f in facts]
audit["factor_drivers"] = [
    " | ".join(f"{k}: {', '.join(v)}" for k, v in f.items()) for f in facts]
audit["factor_count"] = [len(f) for f in facts]
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
def show(factor, n=5, strong_only=True):
    """Print the interactions behind one factor."""
    sub = audit[audit[f"is_{factor}"]]
    if strong_only:
        keep = [i for i, f in zip(sub.index, [factor_row(r) for r in
                                              sub.to_dict("records")])
                if any(fl in STRONG for fl in f.get(factor, []))]
        sub = sub.loc[keep]
    print(f"\n{factor.upper()} — {len(sub)} interactions"
          f"{' (strong evidence only)' if strong_only else ''}\n")
    for _, r in sub.head(n).iterrows():
        print("-" * 78)
        print(f"{r['transaction_id']}   route={r.get('route')}")
        print(f"  Q: {str(r.get('request'))[:150]}")
        print(f"  A: {str(r.get('response'))[:200]}")
        print(f"  driven by: {r['factor_drivers']}")
        if r.get("why_flagged"):
            print(f"  evidence : {str(r['why_flagged'])[:220]}")


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


# ---- 5. WRITE BACK ------------------------------------------------------
if OUTPUT_DATASET:
    keep = [c for c in audit.columns if not c.startswith("flag_")
            and c != "document_chunks"]
    dataiku.Dataset(OUTPUT_DATASET).write_with_schema(audit[keep])
    print(f"\nwrote {OUTPUT_DATASET} ({len(audit)} rows)")
