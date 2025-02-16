import os
import numpy as np
import matplotlib.pyplot as plt
from logger_service.logger import LoggerService

class ModelComparisonPlotter:
    def __init__(self, output_dir):
        self.logger = LoggerService()
        self.output_dir = output_dir
        self.comparison_dir = os.path.join(output_dir, 'comparisons')
        os.makedirs(self.comparison_dir, exist_ok=True)  # Ensure comparisons directory exists

    def _save_plot(self, plot_name):
        """Save plot to the comparisons directory"""
        filename = f"comparison_{plot_name}.png"
        path = os.path.join(self.comparison_dir, filename)
        web_path = f"images/comparisons/{filename}"  # Relative path for web
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return web_path  # Return web-friendly path

    @LoggerService.log_function(level='info')
    def plot_accuracy_comparison(self, model_results):
        """Generate accuracy comparison bar plot"""
        models = list(model_results.keys())
        accuracies = [results['final_results']['basic_metrics']['accuracy'] 
                     for results in model_results.values()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        return self._save_plot('accuracy')

    @LoggerService.log_function(level='info')
    def plot_metrics_comparison(self, model_results):
        """Generate detailed metrics comparison plot"""
        models = list(model_results.keys())
        metrics = ['precision', 'recall', 'f1_score']
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for results in model_results.values():
            detailed_metrics = results['final_results']['detailed_metrics']
            for metric in metrics:
                data[metric].append(detailed_metrics[metric])

        # Plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.25

        # Plot bars for each metric
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, data[metric], width, label=metric.capitalize())

        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Metrics Comparison')
        plt.xticks(x + width, models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        return self._save_plot('metrics')

    @LoggerService.log_function(level='info')
    def plot_confidence_distributions(self, model_results):
        """Generate confidence distribution comparison plot"""
        plt.figure(figsize=(12, 6))
        
        for model_name, results in model_results.items():
            confidences = [result['confidence'] 
                         for result in results['raw_results'].values()]
            plt.hist(confidences, bins=20, alpha=0.5, label=model_name)

        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution Comparison')
        plt.legend()
        plt.tight_layout()
        
        return self._save_plot('confidence')

    @LoggerService.log_function(level='info')
    def generate_all_comparison_plots(self, model_results):
        """Generate all comparison plots"""
        try:
            return {
                'accuracy': self.plot_accuracy_comparison(model_results),
                'metrics': self.plot_metrics_comparison(model_results),
                'confidence': self.plot_confidence_distributions(model_results)
            }
        except Exception as e:
            self.logger.error(f"Error generating comparison plots: {str(e)}")
            raise
