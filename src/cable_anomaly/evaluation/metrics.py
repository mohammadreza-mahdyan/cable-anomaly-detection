from sklearn.metrics import roc_auc_score, average_precision_score

def binary_metrics(y_true, scores):
    return {
        "auroc": roc_auc_score(y_true, scores),
        "auprc": average_precision_score(y_true, scores),
    }
