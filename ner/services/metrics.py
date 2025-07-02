import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_metrics(eval_pred: tuple, pad_mask_id: int = -100, id2label: dict = None):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    
    true_labels = [[id2label[l] for l in label if l != pad_mask_id] for label in labels]
    pred_labels = [[id2label[p] for p, l in zip(pred, label) if l != pad_mask_id] for pred, label in zip(predictions, labels)]

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    
    metrics = {
        'precision': float(precision) * 100,
        'recall': float(recall) * 100,
        'f1': float(f1) * 100,
        'accuracy': float(accuracy) * 100,
    }
    
    return metrics