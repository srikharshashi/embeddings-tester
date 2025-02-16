import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from logger_service.logger import LoggerService
from configuration_manager.config_manager import ConfigManager

class PlotGenerator:
    def __init__(self, model_name):
        self.logger = LoggerService()
        self.config = ConfigManager()
        # Ensure paths are web-safe by replacing slashes
        self.model_name = model_name.replace('/', '_')
        self.output_dir = os.path.join(self.config.output_config['image_storage'], self.model_name)
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure model directory exists

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _save_plot(self, plot_name):
        """Save plot with proper naming convention"""
        filename = f"{self.model_name}_{plot_name}.png"
        path = os.path.join(self.output_dir, filename)
        web_path = f"images/{self.model_name}/{filename}"  # Relative path for web
        self.logger.debug(f'Path trying to save to for ${plot_name} is ${path}')
        plt.savefig(path)
        plt.close()
        return web_path  # Return web-friendly path

    @LoggerService.log_function(level='info')
    def generate_confusion_matrix(self, final_results):
        """Generate and save confusion matrix visualization"""
        cm = np.array(final_results['confusion_matrix_data']['confusion_matrix'])
        categories = final_results['confusion_matrix_data']['categories']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.colorbar()
        
        tick_marks = np.arange(len(categories))
        plt.xticks(tick_marks, categories, rotation=45, ha='right')
        plt.yticks(tick_marks, categories)
        
        for i in range(len(categories)):
            for j in range(len(categories)):
                plt.text(j, i, str(cm[i, j]),
                         horizontalalignment='center',
                         verticalalignment='center')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return self._save_plot('confusion_matrix')

    @LoggerService.log_function(level='info')
    def generate_roc_curve(self, raw_results, final_results):
        """Generate ROC curve for each category"""
        categories = list(set([result['category'] for result in raw_results.values()]))
        true_labels = []
        predicted_probs = []
        
        for idx in raw_results:
            true_cat = final_results['confusion_matrix_data']['categories'][0]
            if idx in raw_results:
                true_cat = raw_results[idx]['category']
            true_labels.append(true_cat)
            probs = [raw_results[idx]['confidence'] if raw_results[idx]['category'] == cat else 0 
                    for cat in categories]
            predicted_probs.append(probs)

        y_true = label_binarize(true_labels, classes=categories)
        y_pred = np.array(predicted_probs)

        plt.figure(figsize=(10, 8))
        for i, category in enumerate(categories):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{category} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.model_name}')
        plt.legend(loc='lower right', bbox_to_anchor=(1.6, 0))
        return self._save_plot('roc_curve')

    @LoggerService.log_function(level='info')
    def generate_precision_recall_curve(self, raw_results, final_results):
        """Generate Precision-Recall curve for each category"""
        categories = list(set([result['category'] for result in raw_results.values()]))
        true_labels = []
        predicted_probs = []
        
        for idx in raw_results:
            true_cat = final_results['confusion_matrix_data']['categories'][0]
            if idx in raw_results:
                true_cat = raw_results[idx]['category']
            true_labels.append(true_cat)
            probs = [raw_results[idx]['confidence'] if raw_results[idx]['category'] == cat else 0 
                    for cat in categories]
            predicted_probs.append(probs)

        y_true = label_binarize(true_labels, classes=categories)
        y_pred = np.array(predicted_probs)

        plt.figure(figsize=(10, 8))
        for i, category in enumerate(categories):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            plt.plot(recall, precision, label=f'{category}')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {self.model_name}')
        plt.legend(loc='lower right', bbox_to_anchor=(1.6, 0))
        return self._save_plot('precision_recall_curve')

    @LoggerService.log_function(level='info')
    def generate_error_analysis(self, raw_results):
        """Generate error analysis visualization"""
        confidences = [result['confidence'] for result in raw_results.values()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution - {self.model_name}')
        return self._save_plot('error_analysis')

    @LoggerService.log_function(level='info')
    def generate_all_plots(self, raw_results, final_results):
        """Generate all plots for the model"""
        plot_paths = {
            'confusion_matrix': self.generate_confusion_matrix(final_results),
            'roc_curve': self.generate_roc_curve(raw_results, final_results),
            'precision_recall': self.generate_precision_recall_curve(raw_results, final_results),
            'error_analysis': self.generate_error_analysis(raw_results)
        }
        return plot_paths
