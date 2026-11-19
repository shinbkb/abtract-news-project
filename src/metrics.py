import torch

def metrics_from_predictions(y_true, y_pred, num_classes):
    y_true, y_pred = y_true.cpu().long(), y_pred.cpu().long()
    cm = torch.bincount(y_true * num_classes + y_pred,
                        minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        'accuracy': (y_true == y_pred).float().mean().item(),
        'macro_precision': precision.mean().item(),
        'macro_recall': recall.mean().item(),
        'macro_f1': f1.mean().item(),
        'confusion_matrix': cm
    }
