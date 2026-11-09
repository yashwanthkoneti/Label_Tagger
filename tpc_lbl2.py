"""
Policy document topic discovery pipeline.

Stages (run in order, each writes to --out):
  extract  PDFs -> intro text (Purpose/Scope/Introduction, else first 2 pages after TOC)
  embed    boilerplate removal -> chunked embeddings -> centering
  cluster  UMAP -> HDBSCAN -> keywords + representative docs per cluster
  label    LLM prompts per cluster (cheap) + per-doc prompts for noise docs (kNN context)
  verify   embed generated labels -> check each label lands nearest its own cluster;
           re-cluster labels to find near-duplicate topics

Install:
  pip install pymupdf sentence-transformers umap-learn hdbscan scikit-learn numpy

Usage:
  Edit the CONFIG block below (paths + settings), choose STAGES, then run:
  python policy_topics.py
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from types import SimpleNamespace

# ============================================================================ CONFIG
CONFIG = SimpleNamespace(
    pdfs=r"C:/path/to/policy_pdfs",   # folder with PDFs (searched recursively)
    out=r"C:/path/to/output",         # all intermediate + result files go here

    # embed
    df_ratio=0.15,       # share of docs a sentence must appear in to count as boilerplate
    drop_pattern=[],     # extra regexes for org-specific boilerplate, e.g. [r"Acme (Ltd|Group)"]
    remove_pcs=0,        # top principal components to remove after centering (0 = off)

    # cluster
    min_cluster_size=5,
    min_samples=None,

    # label
    dry_run=True,        # True = only write prompts.jsonl; False = call call_llm()
    min_prob=0.3,        # cluster members below this membership prob get a per-doc prompt
    k=5,                 # similar labelled docs shown in per-doc prompts

    # verify
    merge_threshold=0.25,  # cosine distance under which topic labels are merge candidates
)

# Stages to run, in order. Comment out the ones you don't need on a given run.
STAGES = [
    "extract",
    "embed",
    "cluster",
    "label",
    # "verify",   # needs topics.json, i.e. label with dry_run=False
]

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # free, CPU-friendly, 512-token window

INTRO_HEADINGS = r"(purpose|scope|introduction|objectives?|overview|background|policy statement|applicability)"
INTRO_RE = re.compile(rf"^\s*(\d+(\.\d+)*\.?\s+)?{INTRO_HEADINGS}\b.{{0,40}}$", re.I)
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(\.\d+)*\.?\s+[A-Z][^.]{2,60}$")
TOC_LINE_RE = re.compile(r"(\.{3,}|\s{3,}|\t)\s*\d{1,3}\s*$")
MAX_CHARS = 4000


# ----------------------------------------------------------------------------- extract
def page_text(page, margin=0.07):
    """Text of a page, dropping blocks in the top/bottom margin (running headers/footers)."""
    h = page.rect.height
    blocks = page.get_text("blocks", sort=True)
    keep = [b[4] for b in blocks if b[6] == 0 and b[1] > h * margin and b[3] < h * (1 - margin)]
    return "\n".join(keep)


def is_toc_page(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    if re.search(r"\b(table of )?contents\b", " ".join(lines[:5]), re.I):
        return True
    return sum(bool(TOC_LINE_RE.search(l)) for l in lines) / len(lines) > 0.4


def slice_intro(text):
    """From the first intro-type heading, up to the first unrelated numbered heading."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if INTRO_RE.match(l.strip())), None)
    if start is None:
        return None
    out = []
    for l in lines[start:]:
        s = l.strip()
        if out and NUMBERED_HEADING_RE.match(s) and not INTRO_RE.match(s):
            break
        out.append(l)
    return "\n".join(out)


def extract_one(path):
    import fitz  # pymupdf

    doc = fitz.open(path)
    n = doc.page_count
    method, start = "first_pages", 0

    # 1) PDF bookmarks, if present, give exact pages for Purpose/Scope
    toc = doc.get_toc()
    hit = next((e for e in toc if re.search(INTRO_HEADINGS, e[1], re.I)), None)
    if hit and hit[2] >= 1:
        start, method = hit[2] - 1, "bookmark"
    else:
        # 2) skip cover + TOC pages (look only in the first 6 pages)
        for i in range(min(6, n)):
            if is_toc_page(doc[i].get_text()):
                start = i + 1
        while start + 1 < n and is_toc_page(doc[start].get_text()):
            start += 1
        if start > 0:
            method = "after_toc"

    pages = "\n".join(page_text(doc[i]) for i in range(start, min(start + 2, n)))
    intro = slice_intro(pages)
    if intro and len(intro) > 200:
        text, method = intro, method + "+heading"
    else:
        text = pages
    text = re.sub(r"[ \t]+", " ", text).strip()[:MAX_CHARS]
    return {
        "file": str(path),
        "title": (doc.metadata or {}).get("title") or Path(path).stem,
        "method": method,
        "start_page": start + 1,
        "needs_ocr": len(text) < 200,  # likely scanned: route to OCR separately
        "text": text,
    }


def cmd_extract(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(a.pdfs).rglob("*.pdf"))
    with open(out / "docs.jsonl", "w") as f:
        for p in pdfs:
            try:
                rec = extract_one(p)
            except Exception as e:
                rec = {"file": str(p), "error": str(e), "text": "", "needs_ocr": True}
            f.write(json.dumps(rec) + "\n")
    print(f"extracted {len(pdfs)} PDFs -> {out/'docs.jsonl'}")


# ----------------------------------------------------------------------------- boilerplate
def sentences(text):
    text = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.;:!?])\s+(?=[A-Z0-9(])", text) if s.strip()]


def norm(s):
    s = re.sub(r"\d+", "0", s.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def remove_boilerplate(texts, df_ratio=0.15, min_df=5, ngram=6, ngram_cover=0.6, extra_patterns=()):
    """
    Drop sentences that are (near-)identical across many documents.
      - exact: normalized sentence appears in >= threshold docs
      - near:  >= ngram_cover of its word 6-grams are frequent (catches templates with
               a changed name/date, e.g. "This policy applies to all employees of X ...")
    Threshold is relative (default 15% of docs): real boilerplate spans many topics, while a
    sentence shared by one topic's docs is signal. Lower df_ratio only if your largest
    expected topic is small relative to the collection.
    Returns cleaned texts and the list of most common removed sentences (for review).
    """
    thr = max(min_df, int(df_ratio * len(texts)))
    sents = [sentences(t) for t in texts]
    sdf, gdf = Counter(), Counter()
    for ss in sents:
        normed = [norm(s) for s in ss]
        sdf.update(set(normed))
        grams = set()
        for n_ in normed:
            w = n_.split()
            grams.update(tuple(w[i:i + ngram]) for i in range(len(w) - ngram + 1))
        gdf.update(grams)

    pats = [re.compile(p, re.I) for p in extra_patterns]
    cleaned, removed = [], Counter()
    for ss in sents:
        keep = []
        for s in ss:
            n_ = norm(s); w = n_.split()
            grams = [tuple(w[i:i + ngram]) for i in range(len(w) - ngram + 1)]
            near = grams and sum(gdf[g] >= thr for g in grams) / len(grams) >= ngram_cover
            if sdf[n_] >= thr or near or any(p.search(s) for p in pats):
                removed[n_] += 1
            else:
                keep.append(s)
        # never wipe a document: if almost nothing survives, keep the original
        kept = " ".join(keep)
        cleaned.append(kept if len(kept) >= 0.1 * len(" ".join(ss)) else " ".join(ss))
    return cleaned, removed.most_common(50)


# ----------------------------------------------------------------------------- embed
def embed_texts(model, texts, words_per_chunk=300):
    """Chunk long texts to fit the model window; mean-pool chunk vectors; L2-normalize."""
    out = []
    for t in texts:
        w = t.split() or [""]
        chunks = [" ".join(w[i:i + words_per_chunk]) for i in range(0, len(w), words_per_chunk)]
        v = model.encode(chunks, normalize_embeddings=True).mean(axis=0)
        out.append(v / (np.linalg.norm(v) + 1e-12))
    return np.vstack(out)


def postprocess(X, remove_pcs=0):
    """Mean-centering removes the shared 'this is a policy document' direction.
    Optionally remove top principal components too (check they aren't topical first)."""
    X = X - X.mean(axis=0, keepdims=True)
    if remove_pcs:
        from sklearn.decomposition import PCA
        pcs = PCA(n_components=remove_pcs).fit(X).components_
        X = X - (X @ pcs.T) @ pcs
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def load_docs(out):
    return [json.loads(l) for l in open(Path(out) / "docs.jsonl")]


def cmd_embed(a):
    from sentence_transformers import SentenceTransformer

    out = Path(a.out)
    docs = [d for d in load_docs(out) if not d.get("needs_ocr")]
    cleaned, removed = remove_boilerplate([d["text"] for d in docs], df_ratio=a.df_ratio,
                                         extra_patterns=a.drop_pattern or [])
    with open(out / "boilerplate_removed.txt", "w") as f:
        for s, c in removed:
            f.write(f"{c}\t{s}\n")
    model = SentenceTransformer(EMBED_MODEL)
    X_raw = embed_texts(model, cleaned)
    np.save(out / "emb_raw.npy", X_raw)
    np.save(out / "emb.npy", postprocess(X_raw, a.remove_pcs))
    np.save(out / "centering_mean.npy", X_raw.mean(axis=0))
    with open(out / "docs_clean.jsonl", "w") as f:
        for d, c in zip(docs, cleaned):
            f.write(json.dumps({**d, "clean": c}) + "\n")
    print(f"embedded {len(docs)} docs; review {out/'boilerplate_removed.txt'}")


# ----------------------------------------------------------------------------- cluster
def ctfidf_keywords(cluster_texts, top=10):
    from sklearn.feature_extraction.text import CountVectorizer

    labels = list(cluster_texts)
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tf = vec.fit_transform([" ".join(cluster_texts[k]) for k in labels]).toarray().astype(float)
    tf = tf / (tf.sum(axis=1, keepdims=True) + 1e-12)
    idf = np.log(1 + tf.sum(axis=0).mean() / (tf.sum(axis=0) + 1e-12))
    score = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    return {k: vocab[np.argsort(-score[i])[:top]].tolist() for i, k in enumerate(labels)}


def cmd_cluster(a):
    out = Path(a.out)
    X = np.load(out / "emb.npy")
    docs = [json.loads(l) for l in open(out / "docs_clean.jsonl")]

    try:
        import umap
        Z = umap.UMAP(n_neighbors=min(15, len(X) - 1), n_components=min(10, len(X) - 2),
                      metric="cosine", min_dist=0.0, random_state=42).fit_transform(X)
    except ImportError:
        print("umap-learn not installed; clustering on raw embeddings")
        Z = X
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=a.min_cluster_size, min_samples=a.min_samples,
                             cluster_selection_method="eom", prediction_data=True).fit(Z)
        labels, probs = cl.labels_, cl.probabilities_
    except ImportError:
        from sklearn.cluster import HDBSCAN
        cl = HDBSCAN(min_cluster_size=a.min_cluster_size, min_samples=a.min_samples).fit(Z)
        labels, probs = cl.labels_, cl.probabilities_

    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[int(l)].append(i)
    kw = ctfidf_keywords({k: [docs[i]["clean"] for i in v] for k, v in groups.items() if k != -1})

    clusters = {}
    for k, idx in groups.items():
        if k == -1:
            continue
        c = X[idx].mean(axis=0); c /= np.linalg.norm(c)
        reps = [idx[j] for j in np.argsort(-(X[idx] @ c))[:5]]
        clusters[k] = {"size": len(idx), "keywords": kw[k], "representatives": reps, "centroid": c.tolist()}

    json.dump({"labels": labels.tolist(), "probs": probs.tolist(), "clusters": clusters},
              open(out / "clusters.json", "w"))
    with open(out / "clusters_review.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cluster", "size", "keywords", "sample_titles"])
        for k, c in sorted(clusters.items(), key=lambda x: -x[1]["size"]):
            w.writerow([k, c["size"], ", ".join(c["keywords"]),
                        " | ".join(docs[i]["title"] for i in c["representatives"])])
    n_noise = int((labels == -1).sum())
    print(f"{len(clusters)} clusters, {n_noise}/{len(labels)} noise -> {out/'clusters_review.csv'}")


# ----------------------------------------------------------------------------- label
def call_llm(prompt: str) -> str:
    """Plug in your provider here (any cheap model). Must return the model's text."""
    raise NotImplementedError


CLUSTER_PROMPT = """You are classifying organisational policy documents into topic areas.
Below are excerpts (Purpose/Scope/Introduction) from documents that were grouped together
because their content is similar, plus distinctive keywords for the group.

Keywords: {keywords}

Excerpts:
{excerpts}

Topic areas already named for other groups (reuse one if it genuinely fits; do not force it):
{existing}

Name the single policy topic area these documents share. Base it only on the excerpts.
Return JSON only:
{{"topic": "<2-5 words>", "definition": "<one sentence: what policies in this area govern>",
  "fits_existing": "<existing topic name or null>", "confidence": "high|medium|low",
  "outliers": [<excerpt numbers that do not fit>]}}"""

DOC_PROMPT = """Classify this policy document into a topic area.

Document excerpt (Purpose/Scope):
{excerpt}

The most similar already-classified documents and their topics:
{neighbours}

Available topic areas:
{topics}

Choose the best-fitting topic from the list, or propose a new one ONLY if none fit.
Return JSON only:
{{"topic": "<name>", "is_new": true|false, "definition": "<one sentence if new, else null>",
  "confidence": "high|medium|low"}}"""


def cmd_label(a):
    out = Path(a.out)
    docs = [json.loads(l) for l in open(out / "docs_clean.jsonl")]
    X = np.load(out / "emb.npy")
    C = json.load(open(out / "clusters.json"))
    labels, probs = np.array(C["labels"]), np.array(C["probs"])

    # Pass 1: one call per cluster (cheap), using representative docs
    topics = {}
    prompts = []
    for k, c in sorted(C["clusters"].items(), key=lambda x: -x[1]["size"]):
        excerpts = "\n\n".join(f"[{j+1}] {docs[i]['title']}: {docs[i]['clean'][:700]}"
                               for j, i in enumerate(c["representatives"]))
        existing = "\n".join(f"- {t['topic']}: {t['definition']}" for t in topics.values()) or "(none yet)"
        p = CLUSTER_PROMPT.format(keywords=", ".join(c["keywords"]), excerpts=excerpts, existing=existing)
        prompts.append({"cluster": k, "prompt": p})
        if not a.dry_run:
            topics[k] = json.loads(call_llm(p))

    # Pass 2: per-document calls only for noise / low-confidence members, with kNN context
    doc_topic = {i: topics.get(str(l), {}).get("topic") for i, l in enumerate(labels) if l != -1}
    uncertain = [i for i in range(len(docs)) if labels[i] == -1 or probs[i] < a.min_prob]
    topic_list = "\n".join(f"- {t['topic']}: {t['definition']}" for t in topics.values()) or "(run pass 1 first)"
    for i in uncertain:
        sims = X @ X[i]; sims[i] = -1
        nn = [j for j in np.argsort(-sims) if j in doc_topic][:a.k]
        neigh = "\n".join(f"- ({sims[j]:.2f}) {docs[j]['title']} -> {doc_topic[j]}" for j in nn)
        p = DOC_PROMPT.format(excerpt=docs[i]["clean"][:1500], neighbours=neigh or "(none)", topics=topic_list)
        prompts.append({"doc": i, "prompt": p})
        if not a.dry_run:
            doc_topic[i] = json.loads(call_llm(p))["topic"]

    with open(out / "prompts.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    if not a.dry_run:
        json.dump({"cluster_topics": topics, "doc_topics": {str(k): v for k, v in doc_topic.items()}},
                  open(out / "topics.json", "w"), indent=2)
    print(f"{len(C['clusters'])} cluster prompts + {len(uncertain)} doc prompts -> {out/'prompts.jsonl'}")


# ----------------------------------------------------------------------------- verify
def cmd_verify(a):
    """
    Grounding check for LLM topics:
      embed "topic: definition" with the same model + same centering, then for each cluster
      check its label is closer to its own centroid than to every other centroid.
      A label that ranks its own cluster low is likely generic or hallucinated.
    Also groups labels that are near-synonyms (candidates to merge).
    """
    from sentence_transformers import SentenceTransformer

    out = Path(a.out)
    C = json.load(open(out / "clusters.json"))
    T = json.load(open(out / "topics.json"))["cluster_topics"]
    keys = [k for k in C["clusters"] if k in T]
    cents = np.array([C["clusters"][k]["centroid"] for k in keys])

    model = SentenceTransformer(EMBED_MODEL)
    L = model.encode([f"{T[k]['topic']}: {T[k]['definition']}" for k in keys], normalize_embeddings=True)
    L = L - np.load(out / "centering_mean.npy")  # same space as document vectors
    L /= np.linalg.norm(L, axis=1, keepdims=True)

    S = L @ cents.T  # label x centroid
    rows = []
    for i, k in enumerate(keys):
        order = np.argsort(-S[i])
        rank = int(np.where(order == i)[0][0]) + 1
        best_other = max(S[i][j] for j in range(len(keys)) if j != i) if len(keys) > 1 else 0
        rows.append([k, T[k]["topic"], rank, round(float(S[i, i]), 3), round(float(S[i, i] - best_other), 3),
                     "OK" if rank == 1 else ("CHECK" if rank <= 2 else "SUSPECT")])
    with open(out / "label_check.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cluster", "topic", "own_rank", "sim_own", "margin", "status"]); w.writerows(rows)

    # near-duplicate labels -> merge candidates
    from sklearn.cluster import AgglomerativeClustering
    if len(keys) > 2:
        g = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                    distance_threshold=a.merge_threshold).fit(L).labels_
        merges = defaultdict(list)
        for k, gi in zip(keys, g):
            merges[gi].append(T[k]["topic"])
        with open(out / "merge_candidates.txt", "w") as f:
            for v in merges.values():
                if len(v) > 1:
                    f.write(" | ".join(v) + "\n")
    print(f"wrote {out/'label_check.csv'} and {out/'merge_candidates.txt'}")


# ----------------------------------------------------------------------------- run
def main():
    stages = {"extract": cmd_extract, "embed": cmd_embed, "cluster": cmd_cluster,
              "label": cmd_label, "verify": cmd_verify}
    for name in STAGES:
        print(f"=== {name}")
        stages[name](CONFIG)


if __name__ == "__main__":
    main()
