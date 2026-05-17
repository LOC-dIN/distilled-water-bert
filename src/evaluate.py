import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _get_predictions(eval_pred):
    if hasattr(eval_pred, 'predictions'):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    return logits, labels, preds


def compute_metrics(eval_pred):
    _, labels, preds = _get_predictions(eval_pred)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1': f1_score(labels, preds, average='weighted', zero_division=0),
    }


def full_evaluation(labels, preds, probs=None, label_names=None):
    labels = np.array(labels)
    preds = np.array(preds)

    print('Accuracy:', accuracy_score(labels, preds))
    print('Precision:', precision_score(labels, preds, average='weighted', zero_division=0))
    print('Recall:', recall_score(labels, preds, average='weighted', zero_division=0))
    print('F1:', f1_score(labels, preds, average='weighted', zero_division=0))

    if probs is not None:
        try:
            print('ROC-AUC:', roc_auc_score(labels, probs))
        except ValueError:
            print('ROC-AUC: unavailable')

    print('\nConfusion Matrix:')
    print(confusion_matrix(labels, preds))

    if label_names:
        print('Labels:', label_names)
