from typing import List, Dict, Any, Optional, Tuple
import numpy as np

def autocluster_text(
    texts: List[str],
    *,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    try_umap: bool = True,
    k_grid: Tuple[int, int, int] = (5, 31, 5),     # start, stop, step for KMeans
    hdb_min_cluster_size: List[int] = [5, 10, 15],
    hdb_min_samples: List[Optional[int]] = [None, 1, 5],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Retourne:
      {
        "labels": np.ndarray [n_samples] (labels -1 = bruit HDBSCAN),
        "pipeline": {"embed": model_name, "reducer": <None|UMAP(...)>, "cluster": <KMeans/HDBSCAN params>},
        "score": float (silhouette ou CH),
        "algo": "kmeans"|"hdbscan"
      }
    """
    # Embeddings
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.cluster import KMeans
    import warnings

    model = SentenceTransformer(embed_model)
    X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # Optionnel: réduction UMAP
    candidates = []
    mats = [("none", X)]
    if try_umap:
        try:
            import umap
            reducer = umap.UMAP(n_components=10, metric="cosine", random_state=random_state, n_neighbors=15, min_dist=0.0)
            Xu = reducer.fit_transform(X)
            mats.append(("umap", Xu))
            umap_obj = reducer
        except Exception:
            umap_obj = None
    else:
        umap_obj = None

    best = {"score": -1e9}

    # KMeans grid
    for tag, M in mats:
        for k in range(*k_grid):
            if k <= 1 or k >= len(M):
                continue
            try:
                km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
                y = km.fit_predict(M)
                # silhouette sur cosinus si sans UMAP, euclid sinon
                metric = "cosine" if tag == "none" else "euclidean"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if len(set(y)) > 1:
                        s = silhouette_score(M, y, metric=metric)
                    else:
                        s = -1.0
                # en back-up, CH
                if s <= 0 and len(set(y)) > 1:
                    s = calinski_harabasz_score(M, y)
                if s > best["score"]:
                    best = {
                        "score": float(s),
                        "labels": y,
                        "algo": "kmeans",
                        "pipeline": {
                            "embed": embed_model,
                            "reducer": tag if tag == "none" else "umap(n_components=10,cosine)",
                            "cluster": f"KMeans(k={k})"
                        }
                    }
            except Exception:
                pass

    # HDBSCAN (auto-k)
    try:
        import hdbscan
        for tag, M in mats:
            for mcs in hdb_min_cluster_size:
                for ms in hdb_min_samples:
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=mcs,
                            min_samples=ms,
                            metric="euclidean" if tag != "none" else "euclidean",  # HDBSCAN euclid par défaut
                            cluster_selection_epsilon=0.0
                        )
                        y = clusterer.fit_predict(M)
                        # silhouette sur les points non-bruit
                        mask = y != -1
                        if mask.sum() >= 2 and len(set(y[mask])) > 1:
                            s = silhouette_score(M[mask], y[mask], metric="euclidean")
                        else:
                            s = -1.0
                        if s > best["score"]:
                            best = {
                                "score": float(s),
                                "labels": y,
                                "algo": "hdbscan",
                                "pipeline": {
                                    "embed": embed_model,
                                    "reducer": tag if tag == "none" else "umap(n_components=10,cosine)",
                                    "cluster": f"HDBSCAN(min_cluster_size={mcs}, min_samples={ms})"
                                }
                            }
                    except Exception:
                        pass
    except Exception:
        pass

    # Fallback si rien de concluant
    if best["score"] == -1e9:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=5, n_init="auto", random_state=random_state).fit(X)
        best = {
            "score": -1.0,
            "labels": km.labels_,
            "algo": "kmeans",
            "pipeline": {"embed": embed_model, "reducer": "none", "cluster": "KMeans(k=5)"}
        }

    return best
