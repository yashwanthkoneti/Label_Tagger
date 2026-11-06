import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

X = X_reduced.astype(int)
theme = df['theme']

# ---------- 1. Decision tree: one theme vs rest ----------
tree_rules = {}
for t in theme.unique():
    y = (theme == t).astype(int)
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=max(10, int(0.01 * len(X))),
        class_weight='balanced',
        random_state=0,
    )
    f1 = cross_val_score(tree, X, y, cv=5, scoring='f1').mean()
    tree.fit(X, y)
    tree_rules[t] = export_text(tree, feature_names=list(X.columns))

    top = pd.Series(tree.feature_importances_, index=X.columns).nlargest(5)
    print(f"\n=== {t} | CV F1 = {f1:.2f} ===")
    print(top[top > 0].round(3).to_string())
    print(tree_rules[t])

# ---------- 2. Clustering within each theme ----------
def cluster_theme(sub, k_range=range(2, 7)):
    if len(sub) < 30:
        return None
    D = squareform(pdist(sub.values.astype(bool), 'hamming'))
    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        labels = AgglomerativeClustering(
            n_clusters=k, metric='precomputed', linkage='average').fit_predict(D)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(D, labels, metric='precomputed')
        if s > best_score:
            best_k, best_score, best_labels = k, s, labels
    return best_k, best_score, best_labels

for t in theme.unique():
    sub = X[theme == t]
    res = cluster_theme(sub)
    if res is None:
        print(f"\n{t}: too few rows to cluster")
        continue
    k, score, labels = res
    print(f"\n=== {t}: k={k}, silhouette={score:.2f}, sizes={np.bincount(labels)} ===")

    # Profile: flags that differ most from the theme average
    prof = sub.groupby(labels).mean()
    diff = prof - sub.mean()
    for c in range(k):
        d = diff.loc[c]
        top = d.reindex(d.abs().sort_values(ascending=False).index).head(5)
        print(f"  Cluster {c}: " + ", ".join(f"{f} ({prof.loc[c, f]:.0%} vs {sub.mean()[f]:.0%})" for f in top.index))
