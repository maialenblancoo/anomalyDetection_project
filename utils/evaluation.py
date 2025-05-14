import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, precision_recall_curve
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)

def evaluate_threshold(errors, y_true, val_errors, factor=1.5):
    """Evalúa el rendimiento del modelo utilizando un umbral dinámico."""
    threshold = np.mean(val_errors) + factor * np.std(val_errors)
    y_pred = (errors > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    try:
        auc_roc = roc_auc_score(y_true, errors)
    except:
        auc_roc = np.nan

    precision_pr, recall_pr, _ = precision_recall_curve(y_true, errors)
    auc_pr = auc(recall_pr, precision_pr)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'threshold': threshold
    }

class AnomalyInterpreter:
    def __init__(self, model, test_data, reconstructed_data, errors, thresholds, timestamps=None):
        self.model = model
        self.test_data = test_data
        self.reconstructed_data = reconstructed_data
        self.errors = errors
        self.thresholds = thresholds
        self.timestamps = timestamps
        self.anomaly_indices = np.where(errors > thresholds)[0]

    def plot_reconstruction_comparison(self, index, window_size=10):
        if index >= len(self.test_data):
            logger.warning(f"Index {index} out of range. Using last valid index.")
            index = len(self.test_data) - 1

        original = self.test_data[index].squeeze()
        reconstructed = self.reconstructed_data[index].squeeze()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(original, label='Original', color='blue')
        ax1.plot(reconstructed, label='Reconstructed', color='red', linestyle='--')
        ax1.set_title(f'Original vs Reconstructed Comparison (Error: {self.errors[index]:.4f})')
        ax1.legend()
        ax1.grid(True)

        point_errors = np.abs(original - reconstructed)
        ax2.plot(point_errors, label='Point-wise Error', color='green')
        ax2.axhline(y=np.mean(point_errors), color='r', linestyle='-', label='Mean Error')
        ax2.set_title('Point-wise Reconstruction Error')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

    def visualize_anomaly_distribution(self):
        plt.figure(figsize=(12, 6))

        plt.hist(self.errors, bins=50, alpha=0.6, color='blue', density=True, label='Error Distribution')

        x = np.linspace(min(self.errors), max(self.errors), 1000)
        mu, std = np.mean(self.errors), np.std(self.errors)
        pdf = norm.pdf(x, mu, std)
        plt.plot(x, pdf, 'r-', linewidth=2, label='Normal Distribution')

        static_threshold = mu + 1.5 * std
        plt.axvline(x=static_threshold, color='green', linestyle='--', label=f'Static Threshold (1.5σ)')

        if isinstance(self.thresholds, np.ndarray):
            plt.axvline(x=np.mean(self.thresholds), color='purple', linestyle=':', label=f'Mean Dynamic Threshold')
        else:
            plt.axvline(x=self.thresholds, color='purple', linestyle=':', label=f'Adaptive Threshold')

        plt.title('Reconstruction Error Distribution')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)

        return plt.gcf()

    def visualize_contribution_heatmap(self, index, window_size=10):
        if index >= len(self.test_data):
            logger.warning(f"Index {index} out of range. Using last valid index.")
            index = len(self.test_data) - 1

        original = self.test_data[index].squeeze()
        reconstructed = self.reconstructed_data[index].squeeze()

        point_errors = np.abs(original - reconstructed)
        normalized_errors = (point_errors - np.min(point_errors)) / (np.max(point_errors) - np.min(point_errors) + 1e-10)

        plt.figure(figsize=(12, 4))
        plt.imshow(normalized_errors.reshape(1, -1), aspect='auto', cmap='hot')
        plt.colorbar(label='Error Contribution')
        plt.title(f'Point-wise Error Contribution - Sequence {index}')
        plt.xlabel('Time')
        plt.yticks([])

        return plt.gcf()

    def analyze_top_anomalies(self, top_k=5):
        top_indices = self.anomaly_indices[np.argsort(self.errors[self.anomaly_indices])[-top_k:]]

        results = []
        for idx in top_indices:
            original = self.test_data[idx].squeeze()
            reconstructed = self.reconstructed_data[idx].squeeze()
            error = self.errors[idx]
            threshold = self.thresholds[idx] if isinstance(self.thresholds, np.ndarray) else self.thresholds

            error_ratio = error / threshold
            max_point_error = np.max(np.abs(original - reconstructed))
            error_start_idx = np.argmax(np.abs(original - reconstructed) > np.std(original))

            anomaly_info = {
                'index': idx,
                'error': error,
                'threshold': threshold,
                'error_ratio': error_ratio,
                'max_point_error': max_point_error,
                'error_start_idx': error_start_idx
            }

            if self.timestamps is not None and idx < len(self.timestamps):
                anomaly_info['timestamp'] = self.timestamps[idx]

            results.append(anomaly_info)

        return results

def visualize_results(results_dict):
    """Visualización avanzada de los resultados del modelo."""

    lstm_errors = results_dict['errors']['lstm']
    gru_errors = results_dict['errors']['gru']
    transformer_errors = results_dict['errors']['transformer']
    ensemble_errors = results_dict['errors']['ensemble']

    lstm_thresholds = results_dict['thresholds']['lstm']
    gru_thresholds = results_dict['thresholds']['gru']
    transformer_thresholds = results_dict['thresholds']['transformer']
    ensemble_thresholds = results_dict['thresholds']['ensemble']

    y_test = results_dict['data']['y_test']
    timestamps = results_dict['data']['timestamps']
    interpreter = results_dict['interpreter']

    # 1. Reconstruction error vs threshold comparison
    plt.figure(figsize=(15, 10))

    plt.subplot(4, 1, 1)
    plt.plot(lstm_errors, label='LSTM Error', alpha=0.7)
    plt.plot(lstm_thresholds, label='LSTM Threshold', linestyle='--', color='r')
    plt.fill_between(range(len(y_test)), 0, 1, where=y_test > 0, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
    plt.ylim(0, max(lstm_errors) * 1.1)
    plt.legend()
    plt.title('LSTM Reconstruction Error vs Dynamic Threshold')

    plt.subplot(4, 1, 2)
    plt.plot(gru_errors, label='GRU Error', alpha=0.7)
    plt.plot(gru_thresholds, label='GRU Threshold', linestyle='--', color='r')
    plt.fill_between(range(len(y_test)), 0, 1, where=y_test > 0, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
    plt.ylim(0, max(gru_errors) * 1.1)
    plt.legend()
    plt.title('GRU Reconstruction Error vs Dynamic Threshold')

    plt.subplot(4, 1, 3)
    plt.plot(transformer_errors, label='Transformer Error', alpha=0.7)
    plt.plot(transformer_thresholds, label='Transformer Threshold', linestyle='--', color='r')
    plt.fill_between(range(len(y_test)), 0, 1, where=y_test > 0, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
    plt.ylim(0, max(transformer_errors) * 1.1)
    plt.legend()
    plt.title('Transformer Reconstruction Error vs Dynamic Threshold')

    plt.subplot(4, 1, 4)
    plt.plot(ensemble_errors, label='Ensemble Error', alpha=0.7)
    plt.plot(ensemble_thresholds, label='Ensemble Threshold', linestyle='--', color='r')
    plt.fill_between(range(len(y_test)), 0, 1, where=y_test > 0, alpha=0.3, color='red', transform=plt.gca().get_xaxis_transform())
    plt.ylim(0, max(ensemble_errors) * 1.1)
    plt.legend()
    plt.title('Ensemble Reconstruction Error vs Dynamic Threshold')

    plt.tight_layout()
    return plt.gcf()

def plot_roc_curves(results_dict):
    """Visualiza las curvas ROC para todos los modelos."""
    from sklearn.metrics import roc_curve, auc
    
    y_test = results_dict['data']['y_test']
    lstm_errors = results_dict['errors']['lstm']
    gru_errors = results_dict['errors']['gru']
    transformer_errors = results_dict['errors']['transformer']
    ensemble_errors = results_dict['errors']['ensemble']
    
    plt.figure(figsize=(10, 8))

    fpr, tpr, _ = roc_curve(y_test, lstm_errors)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'LSTM (AUC = {roc_auc:.3f})')

    fpr, tpr, _ = roc_curve(y_test, gru_errors)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'GRU (AUC = {roc_auc:.3f})')

    fpr, tpr, _ = roc_curve(y_test, transformer_errors)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Transformer (AUC = {roc_auc:.3f})')

    fpr, tpr, _ = roc_curve(y_test, ensemble_errors)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Ensemble (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of the Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    return plt.gcf()

def plot_precision_recall_curves(results_dict):
    """Visualiza las curvas de Precision-Recall para todos los modelos."""
    from sklearn.metrics import precision_recall_curve, auc
    
    y_test = results_dict['data']['y_test']
    lstm_errors = results_dict['errors']['lstm']
    gru_errors = results_dict['errors']['gru']
    transformer_errors = results_dict['errors']['transformer']
    ensemble_errors = results_dict['errors']['ensemble']
    
    plt.figure(figsize=(10, 8))

    precision, recall, _ = precision_recall_curve(y_test, lstm_errors)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'LSTM (AUC = {pr_auc:.3f})')

    precision, recall, _ = precision_recall_curve(y_test, gru_errors)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'GRU (AUC = {pr_auc:.3f})')

    precision, recall, _ = precision_recall_curve(y_test, transformer_errors)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'Transformer (AUC = {pr_auc:.3f})')

    precision, recall, _ = precision_recall_curve(y_test, ensemble_errors)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'Ensemble (AUC = {pr_auc:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves of the Models')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    return plt.gcf()

def create_results_summary(results_dict):
    """Crea un resumen de los resultados de evaluación de todos los modelos."""
    models = ['lstm', 'gru', 'transformer', 'ensemble']
    metrics = ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'tp', 'fp', 'tn', 'fn']
    
    summary = {}
    for model in models:
        summary[model] = {metric: results_dict['results'][model][metric] for metric in metrics}
        summary[model]['threshold'] = results_dict['thresholds'][model]
    
    return summary

def print_results_table(summary):
    """Imprime una tabla formateada con los resultados de evaluación."""
    models = list(summary.keys())
    
    print("\n=== Results Summary ===")
    print(f"{'Model':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC-ROC':<10} {'AUC-PR':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
    print("-" * 100)
    
    for model in models:
        result = summary[model]
        print(f"{model:<12} {result['precision']:<10.4f} {result['recall']:<10.4f} {result['f1']:<10.4f} "
              f"{result['auc_roc']:<10.4f} {result['auc_pr']:<10.4f} {result['tp']:<6d} {result['fp']:<6d} "
              f"{result['tn']:<6d} {result['fn']:<6d}")
    
    print("-" * 100)