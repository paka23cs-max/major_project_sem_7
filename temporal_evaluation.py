import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_anomaly_scores(recon_errors, alpha=0.7, beta=0.3):
    recon_errors = np.array(recon_errors)

    temporal_diff = np.abs(np.diff(recon_errors, prepend=recon_errors[0]))

    recon_norm = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)
    temp_norm = (temporal_diff - temporal_diff.min()) / (temporal_diff.max() - temporal_diff.min() + 1e-8)

    scores = alpha * recon_norm + beta * temp_norm
    return scores


def smooth_scores(scores, window=5):
    return np.convolve(scores, np.ones(window)/window, mode='same')


def compute_threshold(scores, method="mean_std"):
    scores = np.array(scores)

    if method == "percentile":
        return np.percentile(scores, 90)

    elif method == "mean_std":
        return np.mean(scores) + 2 * np.std(scores)

    else:
        raise ValueError("Unknown threshold method")


def evaluate_anomaly_detection(recon_errors, labels):
    recon_errors = np.array(recon_errors)
    labels = np.array(labels)

    scores = compute_anomaly_scores(recon_errors)
    scores = smooth_scores(scores)

    threshold = compute_threshold(scores, method="mean_std")

    preds = (scores > threshold).astype(int)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, scores)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "threshold": threshold
    }


def print_metrics(metrics, title="Results"):
    print(f"\n--- {title} ---")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.6f}")