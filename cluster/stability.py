# cluster/stability.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import random

def stability_assessment_against_full(embs, full_labels, n_clusters, n_rounds=50, sample_frac=0.8, random_seed=42, kmeans_n_init=10, verbose=True):
    np.random.seed(random_seed)
    random.seed(random_seed)
    n = embs.shape[0]
    aris = []

    for r in range(n_rounds):
        idx = np.random.choice(n, size=int(n * sample_frac), replace=False)
        kmeans_sub = KMeans(n_clusters=n_clusters, random_state=random.randint(0, 999999), n_init=kmeans_n_init)
        sub_labels = kmeans_sub.fit_predict(embs[idx])
        ref_labels = np.array(full_labels)[idx]
        ari = adjusted_rand_score(ref_labels, sub_labels)
        aris.append(ari)
        if verbose and (r+1) % max(1, n_rounds//5) == 0:
            print(f"round {r+1}/{n_rounds}  ARI={ari:.4f}")

    aris = np.array(aris)
    return {"aris": aris, "mean_ari": float(np.mean(aris)), "std_ari": float(np.std(aris)), "n_rounds": n_rounds, "sample_frac": sample_frac}

def plot_stability(results, bins=20):
    aris = results["aris"]
    mean_ari = results["mean_ari"]
    std_ari = results["std_ari"]
    plt.figure(figsize=(6,3.5))
    plt.hist(aris, bins=bins, edgecolor='k', alpha=0.7)
    plt.axvline(mean_ari, color='r', linestyle='--', label=f"mean={mean_ari:.3f}")
    plt.title("Stability (ARI distribution across subsamples)")
    plt.xlabel("ARI")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f"Mean ARI: {mean_ari:.4f}, Std ARI: {std_ari:.4f}")
