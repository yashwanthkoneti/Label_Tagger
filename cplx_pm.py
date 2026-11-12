"""
policy_monitor
==============

Post-generation complexity / risk monitoring for a policy RAG system, for the
case where you have NO graded outcomes and labelling is not practical.

What this does and does not claim
---------------------------------
It does NOT produce a validated difficulty score. Without ground truth you
cannot show that any number here predicts failure, and this file will not
pretend otherwise. What it produces is:

  1. Reference-free SIGNALS computed from the retrieved chunks and the
     generated answer. Each one is a mechanical property of the interaction
     (how concentrated retrieval was, whether the answer is traceable to the
     chunks, whether the policy branched on an attribute the user never gave).
     No signal needs a label to compute, and none needs a keyword list of your
     business vocabulary.

  2. FLAGS: named, actionable conditions, each pointing at a specific
     remediation. These are what you act on per-interaction.

  3. A composite `risk` score. Treat it as an ORDINAL for ranking a review
     queue, not as a probability and not as a difficulty measurement. It is
     unvalidated by construction.

  4. DRIFT ALERTS. This is the part that is trustworthy without labels. You
     cannot verify "this answer was wrong", but you can verify "thin-retrieval
     interactions tripled since last week", which is actionable on its own and
     usually means a stale index or a changed policy.

Why thresholds are not hardcoded
--------------------------------
Retrieval scores are not comparable across retrievers (cosine vs BM25 vs a
reranker), and drift over time as the corpus changes. Every signal is therefore
z-scored against a rolling baseline built from your own traffic. The system
calibrates against itself. The only constant you must supply is how many
standard deviations counts as an alert.

Why there is a lexicon here when I warned against them
------------------------------------------------------
The hedging detector matches against the ANSWER, not the query. The answer is
produced by your own model in your own prompt's register, so its vocabulary is
narrow and predictable. User queries are open-vocabulary and were the thing a
lexicon could never cover. Keep this distinction: do not extend these patterns
to match user input.

Getting ground truth anyway
---------------------------
You said labelling is not practical as a project. It is still obtainable as a
by-product:
  - `review_queue()` surfaces the N riskiest interactions per period. If anyone
    reviews even 10 a week, that is 500 labels a year for free, and they are
    concentrated exactly where the signal is uncertain.
  - Implicit negatives cost nothing to log: the user rephrases the same question
    within a few minutes, the user thumbs-down, or the user opens an HR ticket
    shortly after asking. In a policy bot the last one is a strong failure
    label you do not have to create.
Once you have a few hundred, `correlate_with_outcomes()` tells you which of
these signals actually predicts failure in YOUR system, and you can drop the
rest.

Stdlib only. Python 3.9+.

    python policy_monitor.py --selftest
    python policy_monitor.py --demo
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Chunk", "Interaction", "Signals", "Risk", "Monitor", "Flag",
    "extract_signals", "score_risk", "correlate_with_outcomes",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One retrieved passage.

    metadata is the important field. Whatever keys your policy documents carry
    -- country, entity, contract_type, band, effective_date, policy_area,
    sensitivity -- put them here. The branch detector learns which attributes
    matter from this metadata rather than from any hardcoded list, so it works
    with your schema without being told what your schema is.
    """
    text: str
    score: float
    doc_id: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Interaction:
    query: str
    chunks: List[Chunk]
    answer: str
    # Attributes already known about the asker (from HRIS/SSO). Any metadata key
    # present here is considered resolved and will not be flagged as a missing
    # branch. Supplying these converts most CLARIFY cases into clean answers.
    user_attributes: Dict[str, str] = field(default_factory=dict)
    citations: Optional[List[str]] = None   # doc_ids the answer cited, if tracked
    interaction_id: str = ""


class Flag:
    THIN_COVERAGE = "THIN_COVERAGE"              # corpus may not cover this
    DIFFUSE_MATCH = "DIFFUSE_MATCH"              # no clear winning passage
    MULTI_DOC = "MULTI_DOC"                      # answer spans several policies
    BRANCH_UNRESOLVED = "BRANCH_UNRESOLVED"      # policy varies on an unknown attribute
    STALE_SOURCE = "STALE_SOURCE"                # retrieved doc superseded/expired
    LOW_GROUNDEDNESS = "LOW_GROUNDEDNESS"        # answer not traceable to chunks
    UNCITED = "UNCITED"                          # claims without citations
    HEDGED = "HEDGED"                            # model signalling its own doubt
    SENSITIVE = "SENSITIVE"                      # corpus marks this human-only
    COMPOUND_QUERY = "COMPOUND_QUERY"            # several distinct asks in one
    TABLE_DAMAGED = "TABLE_DAMAGED"              # pipe table shifted or cut by chunking


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

SIGNAL_NAMES = [
    "top_score", "score_margin", "score_decay", "support_mass",
    "doc_entropy", "n_docs", "branch_unresolved_count",
    "unsupported_ratio", "citation_gap", "hedge_ratio",
    "query_parts", "answer_len", "stale_ratio", "sensitive_hit",
    "table_ragged", "table_orphan",
]

# Applies to the ANSWER only. See module docstring.
_HEDGE = re.compile(
    r"\b(?:i (?:do not|don't) have|not specified|does not specify|unclear|"
    r"unable to (?:find|determine)|no information|not covered|"
    r"you (?:may|should|might) (?:want to )?(?:check|contact|consult|reach out)|"
    r"please (?:contact|consult|verify|confirm)|"
    r"(?:it|this) (?:may|might|could) (?:vary|depend)|generally speaking|"
    r"typically|in most cases|i (?:am|'m) not (?:sure|certain)|"
    r"cannot confirm|recommend (?:you )?(?:contacting|reaching out|speaking))\b",
    re.IGNORECASE,
)

_STOP = frozenset("""
a an the of to in for on at by with from as is are was were be been being and or
but if then than that this these those it its it's you your we our they their he
she his her i me my do does did not no yes can could will would shall should may
might must have has had having there here what when where which who whom how why
all any both each few more most other some such only own same so too very s t
""".split())

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
# Structural, not lexical: counts distinct asks. Interrogative stems and
# question marks are closed-class grammar, not business vocabulary, so this
# generalises in a way the content lexicons of the earlier assessor did not.
_WH = re.compile(r"\b(?:what|how|when|where|why|who|whom|which|whether)\b", re.IGNORECASE)
# "... and should I ...", "... or do we ..." -- a second clause with its own subject
_MODAL_JOIN = re.compile(
    r"\b(?:and|or)\s+(?:can|could|should|would|will|do|does|did|is|are|was|were|must|may|might|am)\s+"
    r"(?:i|we|you|they|it|he|she|the|my|our|their)\b", re.IGNORECASE)


def _count_asks(query: str) -> int:
    """How many distinct questions are packed into one query."""
    wh = len(_WH.findall(query))
    qm = query.count("?")
    return max(1, max(wh, qm) + len(_MODAL_JOIN.findall(query)))


def _content_words(text: str) -> Counter:
    return Counter(w.lower() for w in _WORD.findall(text)
                   if w.lower() not in _STOP and len(w) > 2)


def _entropy(weights: Sequence[float]) -> float:
    """Normalised Shannon entropy in [0,1]. 0 = all mass on one item."""
    total = sum(w for w in weights if w > 0)
    if total <= 0 or len(weights) < 2:
        return 0.0
    ps = [w / total for w in weights if w > 0]
    if len(ps) < 2:
        return 0.0
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps))


def _groundedness(answer: str, chunks: Sequence[Chunk],
                  min_overlap: float = 0.35) -> Tuple[float, List[str]]:
    """Fraction of answer sentences poorly supported by the retrieved text.

    This is a deliberately cheap lexical proxy for attribution. It is biased:
    a correctly paraphrased sentence can look unsupported, and a fluent
    hallucination that reuses corpus vocabulary can look supported. Do not read
    an absolute value off it. It is informative as a TREND, and as a ranking
    signal for the review queue. For a real attribution check, pass a `judge`
    to score_risk() and use an NLI or LLM entailment model instead.
    """
    corpus = _content_words(" ".join(c.text for c in chunks))
    if not corpus:
        return 1.0, []
    bad: List[str] = []
    sents = [s.strip() for s in _SENT.split(answer) if len(s.strip()) > 25]
    if not sents:
        return 0.0, []
    for s in sents:
        cw = _content_words(s)
        if not cw:
            continue
        covered = sum(n for w, n in cw.items() if w in corpus)
        total = sum(cw.values())
        if total and covered / total < min_overlap:
            bad.append(s[:120])
    return len(bad) / len(sents), bad


_PIPE_SEP = re.compile(r"^[\s|:\-+=]+$")


def _table_integrity(text: str) -> Tuple[int, int, int]:
    """Check pipe-delimited tables for structural damage from chunking.

    Returns (n_tables, n_ragged, n_orphan).

    A well-formed pipe table has the same number of delimiters on its header
    and on every data row. Two things break that, and both are chunking
    defects rather than content problems:

      ragged  - rows disagree on column count, so cells have shifted and any
                value the model reads off the table is attached to the wrong
                heading. This is the quiet one: the answer looks specific and
                authoritative and the number is simply from the wrong column.

      orphan  - a header with its separator and no data rows beneath it, or
                data rows with no header anywhere in the chunk. The chunk was
                cut through the middle of a table. Rows without a header are
                the dangerous half: "18 | 30 | 31 March" with no column names
                is uninterpretable, and the model will interpret it anyway.

    Policy corpora are full of tables - entitlement by grade, expense caps by
    city, notice period by tenure - so for a policy bot this is a live failure
    mode, and it is invisible to faithfulness: a claim read off a shifted table
    IS entailed by the retrieved text.
    """
    lines = text.split("\n")
    n_tables = n_ragged = n_orphan = 0
    i = 0
    while i < len(lines):
        if lines[i].count("|") < 2:
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].count("|") >= 2:
            j += 1
        run = lines[i:j]
        i = j
        if len(run) < 2:
            continue
        n_tables += 1
        sep_at = [k for k, ln in enumerate(run) if _PIPE_SEP.match(ln) and "-" in ln]
        content = [ln for k, ln in enumerate(run) if k not in sep_at]
        counts = {ln.count("|") for ln in content}
        if len(counts) > 1:
            n_ragged += 1
        if sep_at:
            # header + separator present; are there any rows after the separator?
            if max(sep_at) >= len(run) - 1:
                n_orphan += 1
        else:
            # no separator anywhere: rows whose header was cut away
            n_orphan += 1
    return n_tables, n_ragged, n_orphan


#   Identity fields: they name WHICH document a chunk came from, so they differ
#   between any two chunks by definition. Treating them as branch attributes
#   makes BRANCH_UNRESOLVED fire on every multi-document answer and produces
#   nonsense clarifiers ("which file name applies to you?"). Document identity
#   is already covered by n_docs / doc_entropy / MULTI_DOC.
BRANCH_FROM_FILENAME = "document_variant"

IDENTITY_KEYS = ("chunk_id", "section", "page", "file_name", "filename",
                 "document_title", "title", "doc_id", "document_id", "source",
                 "uri", "url", "id", "index", "offset", "score")


def _filename_branch(doc_ids: Sequence[str]) -> Dict[str, List[str]]:
    """Recover a branch dimension from systematic file naming.

    Where metadata is absent but filenames are conventional, the varying token
    is the branching attribute:

        leave_policy_in.pdf  +  leave_policy_de.pdf   ->  {in, de}

    Two or more documents whose name tokens match in every position but one
    are variants of the same policy, and that position's values are the thing
    the asker needs to be pinned to. Purely structural - no vocabulary list,
    no assumption about what the token means.

    Heuristic, and reported under a key named to say so. It fires only on an
    exact one-position difference, so unrelated documents retrieved together
    will not trip it.
    """
    stems = []
    for d in doc_ids:
        base = str(d).rsplit("/", 1)[-1]
        base = base.rsplit(".", 1)[0] if "." in base else base
        toks = [t for t in re.split(r"[_\-.\s]+", base.lower()) if t]
        if toks:
            stems.append(toks)
    by_len: Dict[int, List[List[str]]] = {}
    for t in stems:
        by_len.setdefault(len(t), []).append(t)
    for _n, group in sorted(by_len.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 2:
            continue
        for p in range(len(group[0])):
            others_match = all(
                all(g[q] == group[0][q] for q in range(len(g)) if q != p)
                for g in group)
            values = sorted({g[p] for g in group})
            if others_match and len(values) > 1:
                return {BRANCH_FROM_FILENAME: values}
    return {}


def _branch_dimensions(chunks: Sequence[Chunk],
                       user_attributes: Dict[str, str],
                       ignore_keys: Sequence[str] = IDENTITY_KEYS,
                       infer_from_filenames: bool = True,
                       ) -> Dict[str, List[str]]:
    """Metadata keys on which the retrieved policy text DISAGREES, and which the
    asker has not pinned down.

    This is the highest-value signal in a policy bot and it needs no lexicon.
    If a question pulls chunks tagged country=IN, country=DE and country=US, then
    country is a branching dimension for this question. If the asker's attributes
    do not fix it, the answer is being generated without knowing which rule
    applies -- the characteristic way a policy bot is confidently wrong.
    """
    if not chunks:
        return {}
    by_key: Dict[str, set] = {}
    for c in chunks:
        for k, v in (c.metadata or {}).items():
            if k in ignore_keys or v in (None, ""):
                continue
            by_key.setdefault(k, set()).add(str(v))
    out: Dict[str, List[str]] = {}
    for k, vals in by_key.items():
        if len(vals) > 1 and k not in user_attributes:
            out[k] = sorted(vals)

    # Only fall back to filenames when real metadata gave us nothing. Explicit
    # metadata is always preferable to an inference from naming convention.
    if not out and infer_from_filenames and len(chunks) > 1:
        known = {str(v).strip().lower() for v in user_attributes.values()}
        guess = _filename_branch([c.doc_id for c in chunks])
        for k, vals in guess.items():
            # Case-insensitive: a filename token 'in' is resolved by a user
            # attribute of 'IN'.
            if not any(v.strip().lower() in known for v in vals):
                out[k] = vals
    return out


@dataclass
class Signals:
    values: Dict[str, float]
    branch_dims: Dict[str, List[str]] = field(default_factory=dict)
    unsupported_examples: List[str] = field(default_factory=list)

    def __getitem__(self, k: str) -> float:
        return self.values[k]


def extract_signals(ix: Interaction, top_k: int = 5) -> Signals:
    chunks = sorted(ix.chunks, key=lambda c: c.score, reverse=True)[:top_k]
    scores = [c.score for c in chunks]

    s0 = scores[0] if scores else 0.0
    s1 = scores[1] if len(scores) > 1 else 0.0
    sk = scores[-1] if scores else 0.0
    eps = 1e-9

    doc_weight: Dict[str, float] = {}
    for c in chunks:
        doc_weight[c.doc_id] = doc_weight.get(c.doc_id, 0.0) + max(c.score, 0.0)

    branch = _branch_dimensions(chunks, ix.user_attributes)
    unsupported, examples = _groundedness(ix.answer, chunks)

    hedges = len(_HEDGE.findall(ix.answer))
    sents = max(1, len([s for s in _SENT.split(ix.answer) if s.strip()]))

    if ix.citations is not None:
        cited = set(ix.citations)
        retrieved = {c.doc_id for c in chunks}
        citation_gap = 0.0 if cited else 1.0
        if cited:
            # fraction of cited docs that were not actually retrieved
            citation_gap = len(cited - retrieved) / max(1, len(cited))
    else:
        citation_gap = 0.0

    stale = sum(1 for c in chunks
                if str(c.metadata.get("status", "")).lower() in ("superseded", "expired", "archived"))
    sensitive = 1.0 if any(
        str(c.metadata.get("sensitivity", "")).lower() in ("high", "human_only", "restricted")
        for c in chunks) else 0.0

    parts = _count_asks(ix.query)

    n_tab = rag = orph = 0
    for c in chunks:
        t, r, o = _table_integrity(c.text)
        n_tab += t
        rag += r
        orph += o

    values = {
        "top_score": s0,
        "score_margin": (s0 - s1) / (abs(s0) + eps),
        "score_decay": (s0 - sk) / (abs(s0) + eps),
        "support_mass": sum(scores),
        "doc_entropy": _entropy(list(doc_weight.values())),
        "n_docs": float(len(doc_weight)),
        "branch_unresolved_count": float(len(branch)),
        "unsupported_ratio": unsupported,
        "citation_gap": citation_gap,
        "hedge_ratio": hedges / sents,
        "query_parts": float(parts),
        "answer_len": float(len(_WORD.findall(ix.answer))),
        "stale_ratio": stale / max(1, len(chunks)),
        "sensitive_hit": sensitive,
        "table_ragged": float(rag),
        "table_orphan": float(orph),
    }
    return Signals(values=values, branch_dims=branch, unsupported_examples=examples)


# ---------------------------------------------------------------------------
# Per-interaction risk
# ---------------------------------------------------------------------------

@dataclass
class Risk:
    score: float                      # ordinal, 0-100, UNVALIDATED
    flags: List[str]
    signals: Signals
    notes: List[str]
    clarifying_question: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "flags": self.flags,
            "notes": self.notes,
            "clarifying_question": self.clarifying_question,
            "signals": {k: round(v, 4) for k, v in self.signals.values.items()},
            "branch_dims": self.signals.branch_dims,
        }


# How many baseline standard deviations away counts as "unusual" for flagging.
FLAG_SIGMA = 1.5

# Weights for the composite ordinal. Hand-set, because with no labels there is
# nothing to fit them against. Revisit via correlate_with_outcomes() once the
# review queue has produced outcomes.
RISK_WEIGHTS = {
    Flag.THIN_COVERAGE: 1.0,
    Flag.DIFFUSE_MATCH: 0.6,
    Flag.MULTI_DOC: 0.5,
    Flag.BRANCH_UNRESOLVED: 1.2,
    Flag.STALE_SOURCE: 1.0,
    Flag.LOW_GROUNDEDNESS: 1.3,
    Flag.UNCITED: 0.4,
    Flag.HEDGED: 0.7,
    Flag.SENSITIVE: 0.9,
    Flag.COMPOUND_QUERY: 0.5,
    Flag.TABLE_DAMAGED: 1.1,
}

Judge = Callable[[Interaction], float]   # returns unsupported_ratio in [0,1]


def from_faithfulness(fn: Callable[[Interaction], float]) -> Judge:
    """Adapt a RAGAS-style faithfulness score into a Judge.

    Faithfulness is reported as 1.0 = every claim entailed by the context;
    this module's `unsupported_ratio` runs the other way, and the flag fires on
    HIGH values. Inverting by hand is an easy way to silently invert your
    alerting, so do it here:

        mon.observe(ix, judge=from_faithfulness(my_ragas_faithfulness))

    IMPORTANT, and the reason faithfulness alone is not enough for a policy
    bot: faithfulness measures fidelity to the RETRIEVED CONTEXT, not to the
    policy that actually governs the asker. An answer that quotes the German
    leave policy verbatim to an employee in India scores a perfect 1.0 and is
    still wrong. That failure is invisible to faithfulness and is precisely
    what BRANCH_UNRESOLVED and STALE_SOURCE detect. The signals are orthogonal;
    use both, and never treat a high faithfulness score as evidence the answer
    was correct.
    """
    def _judge(ix: Interaction) -> float:
        return 1.0 - max(0.0, min(1.0, float(fn(ix))))
    return _judge


def _phrase_clarifier(dims: Dict[str, List[str]]) -> Optional[str]:
    """Build the clarifying question straight from corpus metadata.

    Entirely corpus-driven: the options are the values that actually occur in
    the retrieved policies, so this stays correct as policies are added.
    """
    if not dims:
        return None
    k, vals = sorted(dims.items(), key=lambda kv: -len(kv[1]))[0]
    shown = vals[:4]
    opts = ", ".join(shown[:-1]) + f" or {shown[-1]}" if len(shown) > 1 else shown[0]
    if k == BRANCH_FROM_FILENAME:
        return f"Which version of this policy applies to you - {opts}?"
    return f"Which {k.replace('_', ' ')} applies to you - {opts}?"


def score_risk(ix: Interaction,
               baseline: Optional["Baseline"] = None,
               judge: Optional[Judge] = None,
               top_k: int = 5) -> Risk:
    """Flag and rank one interaction.

    baseline: rolling statistics from your own traffic. Without it, the
        distribution-relative flags (THIN_COVERAGE, DIFFUSE_MATCH) cannot fire,
        because there is no retriever-independent notion of a 'low' score. Feed
        a Monitor at least a few hundred interactions before trusting those.
    judge: optional callable returning a real attribution score, replacing the
        lexical groundedness proxy. Exceptions are swallowed.
    """
    sig = extract_signals(ix, top_k=top_k)

    if judge is not None:
        try:
            sig.values["unsupported_ratio"] = max(0.0, min(1.0, float(judge(ix))))
        except Exception:
            pass

    flags: List[str] = []
    notes: List[str] = []
    v = sig.values

    if baseline is not None and baseline.ready:
        if baseline.z("top_score", v["top_score"]) < -FLAG_SIGMA:
            flags.append(Flag.THIN_COVERAGE)
            notes.append("top retrieval score unusually low for this corpus; "
                         "the policy set may not cover this question")
        if baseline.z("score_margin", v["score_margin"]) < -FLAG_SIGMA:
            flags.append(Flag.DIFFUSE_MATCH)
            notes.append("no clearly winning passage; retrieval was ambiguous")
        if baseline.z("unsupported_ratio", v["unsupported_ratio"]) > FLAG_SIGMA:
            flags.append(Flag.LOW_GROUNDEDNESS)
        if baseline.z("hedge_ratio", v["hedge_ratio"]) > FLAG_SIGMA:
            flags.append(Flag.HEDGED)
            notes.append("answer hedges more than usual; the model is signalling doubt")
    else:
        notes.append("no baseline yet: distribution-relative flags disabled")
        if v["unsupported_ratio"] > 0.5:
            flags.append(Flag.LOW_GROUNDEDNESS)
        if v["hedge_ratio"] > 0.25:
            flags.append(Flag.HEDGED)

    if v["branch_unresolved_count"] > 0:
        flags.append(Flag.BRANCH_UNRESOLVED)
        notes.append("retrieved policies differ on: "
                     + ", ".join(sig.branch_dims.keys())
                     + "; the asker's attributes do not resolve which applies"
                     + ("; variant inferred from file naming - add explicit "
                        "chunk metadata to make this reliable"
                        if BRANCH_FROM_FILENAME in sig.branch_dims else ""))
    if v["n_docs"] >= 3 and v["doc_entropy"] > 0.75:
        flags.append(Flag.MULTI_DOC)
        notes.append("answer draws on several policies with no dominant source")
    if v["stale_ratio"] > 0:
        flags.append(Flag.STALE_SOURCE)
        notes.append("a retrieved document is marked superseded/expired")
    if v["sensitive_hit"]:
        flags.append(Flag.SENSITIVE)
        notes.append("corpus marks this area as human-only")
    if v["citation_gap"] > 0:
        flags.append(Flag.UNCITED)
    if v["query_parts"] >= 2:
        flags.append(Flag.COMPOUND_QUERY)
    if v["table_ragged"] > 0 or v["table_orphan"] > 0:
        flags.append(Flag.TABLE_DAMAGED)
        bits = []
        if v["table_ragged"]:
            bits.append(f"{int(v['table_ragged'])} table(s) with inconsistent column "
                        f"counts: values may be under the wrong heading")
        if v["table_orphan"]:
            bits.append(f"{int(v['table_orphan'])} table(s) cut by chunking "
                        f"(header without rows, or rows without a header)")
        notes.append("; ".join(bits))
    if Flag.LOW_GROUNDEDNESS in flags and sig.unsupported_examples:
        notes.append("unsupported: " + sig.unsupported_examples[0])

    raw = sum(RISK_WEIGHTS.get(fl, 0.4) for fl in flags)
    score = 100.0 * (1.0 - math.exp(-raw / 2.2))    # saturating, never caps out

    return Risk(
        score=score,
        flags=flags,
        signals=sig,
        notes=notes,
        clarifying_question=_phrase_clarifier(sig.branch_dims),
    )


# ---------------------------------------------------------------------------
# Rolling baseline + drift detection
# ---------------------------------------------------------------------------

class Baseline:
    """Mean/stdev per signal over a rolling window of recent traffic."""

    def __init__(self, window: int = 2000, min_n: int = 100):
        self.window = window
        self.min_n = min_n
        self._buf: Dict[str, deque] = {n: deque(maxlen=window) for n in SIGNAL_NAMES}

    def add(self, sig: Signals) -> None:
        for n in SIGNAL_NAMES:
            if n in sig.values:
                self._buf[n].append(sig.values[n])

    @property
    def ready(self) -> bool:
        return len(self._buf["top_score"]) >= self.min_n

    def stats(self, name: str) -> Tuple[float, float]:
        xs = list(self._buf[name])
        if len(xs) < 2:
            return 0.0, 0.0
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs)
        return mu, sd

    def z(self, name: str, value: float) -> float:
        mu, sd = self.stats(name)
        if sd <= 1e-9:
            return 0.0
        return (value - mu) / sd


@dataclass
class Alert:
    signal: str
    baseline_mean: float
    current_mean: float
    sigma: float
    direction: str

    def __str__(self) -> str:
        return (f"{self.signal}: {self.baseline_mean:.3f} -> {self.current_mean:.3f} "
                f"({self.sigma:+.1f} sigma, {self.direction})")


class Monitor:
    """Accumulates interactions, ranks a review queue, and detects drift.

    Drift detection is the part that does not need labels. A step change in
    `top_score` or `branch_unresolved_count` is real and diagnosable regardless
    of whether you can grade any individual answer.
    """

    def __init__(self, window: int = 2000, current: int = 200,
                 min_n: int = 100, alert_sigma: float = 2.0):
        self.baseline = Baseline(window=window, min_n=min_n)
        self.current: deque = deque(maxlen=current)
        self.alert_sigma = alert_sigma
        self._queue: List[Tuple[float, str, Risk]] = []

    def observe(self, ix: Interaction, judge: Optional[Judge] = None) -> Risk:
        risk = score_risk(ix, baseline=self.baseline, judge=judge)
        self.current.append(risk.signals)
        self.baseline.add(risk.signals)
        self._queue.append((risk.score, ix.interaction_id or ix.query[:60], risk))
        self._queue.sort(key=lambda r: -r[0])
        del self._queue[500:]
        return risk

    def alerts(self) -> List[Alert]:
        """Signals whose recent window has moved against the longer baseline."""
        out: List[Alert] = []
        if not self.baseline.ready or len(self.current) < 30:
            return out
        for name in SIGNAL_NAMES:
            cur = [s.values[name] for s in self.current if name in s.values]
            if len(cur) < 30:
                continue
            mu, sd = self.baseline.stats(name)
            if sd <= 1e-9:
                continue
            cm = statistics.fmean(cur)
            # standard error of the current-window mean
            sigma = (cm - mu) / (sd / math.sqrt(len(cur)))
            if abs(sigma) >= self.alert_sigma:
                out.append(Alert(name, mu, cm, sigma,
                                 "up" if sigma > 0 else "down"))
        return sorted(out, key=lambda a: -abs(a.sigma))

    def review_queue(self, n: int = 20) -> List[Tuple[float, str, Risk]]:
        """The riskiest recent interactions. Reviewing these is how you get
        ground truth without running a labelling project."""
        return self._queue[:n]

    def flag_rates(self) -> Dict[str, float]:
        tot = max(1, len(self._queue))
        c: Counter = Counter()
        for _s, _k, r in self._queue:
            c.update(set(r.flags))
        return {k: v / tot for k, v in c.most_common()}


# ---------------------------------------------------------------------------
# Later, once the review queue has produced outcomes
# ---------------------------------------------------------------------------

def correlate_with_outcomes(signals: Sequence[Signals],
                            failed: Sequence[bool]) -> List[Tuple[str, float]]:
    """Point-biserial correlation of each signal with observed failure.

    Run this once you have a few hundred graded interactions, from the review
    queue or from implicit negatives. It tells you which of these signals
    actually predicts failure in YOUR system so you can drop the rest and
    reweight RISK_WEIGHTS against evidence instead of my guesses.
    """
    out: List[Tuple[str, float]] = []
    n = len(signals)
    if n != len(failed) or n < 10:
        return out
    for name in SIGNAL_NAMES:
        xs = [s.values.get(name, 0.0) for s in signals]
        sd = statistics.pstdev(xs)
        if sd <= 1e-9:
            continue
        mu = statistics.fmean(xs)
        pos = [x for x, f in zip(xs, failed) if f]
        neg = [x for x, f in zip(xs, failed) if not f]
        if len(pos) < 3 or len(neg) < 3:
            continue
        p = len(pos) / n
        r = (statistics.fmean(pos) - statistics.fmean(neg)) / sd * math.sqrt(p * (1 - p))
        out.append((name, r))
    return sorted(out, key=lambda kv: -abs(kv[1]))


# ---------------------------------------------------------------------------
# Demo / tests
# ---------------------------------------------------------------------------

def _demo_interactions() -> List[Interaction]:
    leave_in = Chunk("Employees in India accrue 18 days of paid leave per year. "
                     "Unused leave may be carried over up to a maximum of 30 days.",
                     0.81, "leave-in", {"country": "IN", "policy_area": "leave",
                                        "status": "current"})
    leave_de = Chunk("Employees in Germany accrue 28 days of paid leave. Carryover "
                     "must be used by 31 March of the following year.",
                     0.78, "leave-de", {"country": "DE", "policy_area": "leave",
                                        "status": "current"})
    leave_us = Chunk("US employees follow an unlimited discretionary time off "
                     "policy; there is no carryover.",
                     0.74, "leave-us", {"country": "US", "policy_area": "leave",
                                        "status": "current"})
    expense = Chunk("Client entertainment is capped at 75 USD per head. Overages "
                    "require VP approval before the expense is incurred.",
                    0.88, "expense", {"policy_area": "expense", "status": "current"})
    conduct = Chunk("Reports of harassment are handled confidentially by People "
                    "Operations. Do not route these through automated channels.",
                    0.83, "conduct", {"policy_area": "conduct", "sensitivity": "human_only",
                                      "status": "current"})
    old_travel = Chunk("Business travel requires a Form T-12 submitted two weeks "
                       "in advance.", 0.69, "travel-2019",
                       {"policy_area": "travel", "status": "superseded"})
    weak = Chunk("The company observes ten public holidays per calendar year.",
                 0.21, "holidays", {"policy_area": "leave", "status": "current"})

    return [
        Interaction(
            interaction_id="ix-1",
            query="Can I carry over unused leave into next year?",
            chunks=[leave_in, leave_de, leave_us],
            answer="Unused leave may be carried over up to a maximum of 30 days.",
        ),
        Interaction(
            interaction_id="ix-2",
            query="Can I carry over unused leave into next year?",
            chunks=[leave_in, leave_de, leave_us],
            answer="In India you accrue 18 days per year and may carry over up to 30 days.",
            user_attributes={"country": "IN"},
        ),
        Interaction(
            interaction_id="ix-3",
            query="My client dinner went over the per-head limit. What do I do?",
            chunks=[expense],
            answer="Client entertainment is capped at 75 USD per head. Overages "
                   "require VP approval before the expense is incurred.",
        ),
        Interaction(
            interaction_id="ix-4",
            query="How do I report harassment by my manager?",
            chunks=[conduct],
            answer="Reports of harassment are handled confidentially by People Operations.",
        ),
        Interaction(
            interaction_id="ix-5",
            query="What approvals do I need for business travel?",
            chunks=[old_travel],
            answer="You need to submit a Form T-12 two weeks in advance.",
        ),
        Interaction(
            interaction_id="ix-6",
            query="What is the sabbatical policy and how does it affect my equity vesting?",
            chunks=[weak],
            answer="Sabbaticals are generally available after five years of service "
                   "and vesting typically continues throughout, though you may want "
                   "to check with your People Partner to confirm the details.",
        ),
    ]


def _demo() -> int:
    mon = Monitor(min_n=3, alert_sigma=2.0)
    print("Per-interaction assessment\n" + "=" * 72)
    for ix in _demo_interactions():
        r = mon.observe(ix)
        print(f"\n[{ix.interaction_id}] {ix.query}")
        print(f"  risk {r.score:5.1f}  flags: {', '.join(r.flags) or 'none'}")
        for n in r.notes:
            print(f"    - {n}")
        if r.clarifying_question:
            print(f"    ASK: {r.clarifying_question}")
    print("\n\nFlag rates\n" + "=" * 72)
    for k, v in mon.flag_rates().items():
        print(f"  {k:<22} {v:5.1%}")
    print("\nReview queue (top 3)\n" + "=" * 72)
    for s, key, _r in mon.review_queue(3):
        print(f"  {s:5.1f}  {key}")
    return 0


def _selftest() -> int:
    checks: List[Tuple[str, bool]] = []
    ixs = _demo_interactions()

    r1 = score_risk(ixs[0])   # no country known, 3 countries retrieved
    checks.append(("unresolved branch detected", Flag.BRANCH_UNRESOLVED in r1.flags))
    checks.append(("clarifier generated from metadata",
                   r1.clarifying_question is not None and "country" in r1.clarifying_question))
    checks.append(("clarifier lists real corpus values",
                   all(v in r1.clarifying_question for v in ("DE", "IN", "US"))))

    r2 = score_risk(ixs[1])   # same query, country known
    checks.append(("known attribute resolves branch",
                   Flag.BRANCH_UNRESOLVED not in r2.flags))
    checks.append(("knowing the attribute lowers risk", r2.score < r1.score))

    r4 = score_risk(ixs[3])
    checks.append(("sensitivity from metadata", Flag.SENSITIVE in r4.flags))

    r5 = score_risk(ixs[4])
    checks.append(("stale source flagged", Flag.STALE_SOURCE in r5.flags))

    r6 = score_risk(ixs[5])
    checks.append(("hedging detected", Flag.HEDGED in r6.flags))
    checks.append(("compound query detected", Flag.COMPOUND_QUERY in r6.flags))
    checks.append(("ungrounded answer flagged", Flag.LOW_GROUNDEDNESS in r6.flags))

    # grounded answer should not trip groundedness
    r3 = score_risk(ixs[2])
    checks.append(("verbatim answer is grounded", Flag.LOW_GROUNDEDNESS not in r3.flags))

    # no baseline -> distribution flags suppressed, no crash
    checks.append(("no-baseline note present",
                   any("no baseline" in n for n in r3.notes)))

    # drift detection
    mon = Monitor(window=500, current=40, min_n=40, alert_sigma=2.0)
    good = Chunk("Policy text about leave entitlements and carryover rules.", 0.85,
                 "d1", {"policy_area": "leave", "status": "current"})
    for i in range(200):
        mon.observe(Interaction(query="q", chunks=[good],
                                answer="Policy text about leave entitlements.",
                                interaction_id=f"g{i}"))
    pre = mon.alerts()
    bad = Chunk("Policy text about leave entitlements and carryover rules.", 0.30,
                "d1", {"policy_area": "leave", "status": "current"})
    for i in range(60):
        mon.observe(Interaction(query="q", chunks=[bad],
                                answer="Policy text about leave entitlements.",
                                interaction_id=f"b{i}"))
    post = mon.alerts()
    checks.append(("quiet when stable", len(pre) == 0))
    checks.append(("drift detected on score collapse",
                   any(a.signal == "top_score" and a.direction == "down" for a in post)))

    # robustness
    for ix in [Interaction(query="", chunks=[], answer=""),
               Interaction(query="?" * 50, chunks=[], answer="x"),
               Interaction(query="q", chunks=[Chunk("", 0.0, "d", {})], answer="")]:
        r = score_risk(ix)
        checks.append((f"no crash on degenerate input", 0 <= r.score <= 100))

    def boom(_): raise RuntimeError
    checks.append(("judge failure contained",
                   0 <= score_risk(ixs[0], judge=boom).score <= 100))

    # outcome correlation runs once labels exist
    sigs = [extract_signals(i) for i in ixs] * 3
    fails = [True, False, False, True, True, True] * 3
    corr = correlate_with_outcomes(sigs, fails)
    checks.append(("correlation returns ranked signals", len(corr) > 0))

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    print(f"\n{sum(1 for _, p in checks if p)}/{len(checks)} checks pass")
    return 0 if ok else 1


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    if "--demo" in argv:
        return _demo()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
