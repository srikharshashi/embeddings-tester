import pandas as pd
from jinja2 import Environment, FileSystemLoader
from logger_service.logger import LoggerService
from configuration_manager.config_manager import ConfigManager
import os
import shutil

class OutputGenerator:
    def __init__(self):
        self.logger = LoggerService()
        self.config = ConfigManager()
        self.env = Environment(loader=FileSystemLoader('templates'))
        self.output_dir = os.path.dirname(self.config.output_config['output_file'])
        # Create base output directories
        self._initialize_directories()

    def _initialize_directories(self):
        """Create necessary output directories"""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create images directory
        self.image_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Update image storage config to use the correct path
        self.config.output_config['image_storage'] = self.image_dir

    def _copy_images(self):
        """Copy images to output directory structure"""
        # Since we're already saving directly to the output/images directory,
        # we don't need to copy anything anymore
        pass

    @LoggerService.log_function(level='info')
    def create_model_performance_table(self, model_results):
        """Create a DataFrame with model performance metrics"""
        data = []
        for model_name, results in model_results.items():
            metrics = results['final_results']
            row = {
                'Model': model_name,
                'Accuracy': metrics['basic_metrics']['accuracy'],
                'Precision': metrics['detailed_metrics']['precision'],
                'Recall': metrics['detailed_metrics']['recall'],
                'F1 Score': metrics['detailed_metrics']['f1_score'],
                'Mean Confidence': metrics['confidence_stats']['mean_confidence']
            }
            data.append(row)
        return pd.DataFrame(data)

    @LoggerService.log_function(level='info')
    def generate_report(self, model_results, comparison_plots):
        """Generate comprehensive HTML report using Bootstrap"""
        try:
            # Copy images to output directory
            self._copy_images()
            
            # Create performance metrics table
            metrics_table = self.create_model_performance_table(model_results)
            
            # Load template
            template = self.env.get_template('report_template.html')
            
            # Prepare context for template
            context = {
                'title': 'Model Evaluation Report',
                'metrics_table': metrics_table.to_html(classes='table table-striped', index=False),
                'models': {}
            }

            # Add plot paths and results for each model
            for model_name, results in model_results.items():
                context['models'][model_name] = {
                    'confusion_matrix': results['plots']['confusion_matrix'],
                    'roc_curve': results['plots']['roc_curve'],
                    'precision_recall': results['plots']['precision_recall'],
                    'error_analysis': results['plots']['error_analysis'],
                    'metrics': results['final_results']
                }

            # Add comparison plots
            context['comparison_plots'] = comparison_plots

            # Render template
            html_output = template.render(context)
            
            # Save the report
            output_path = self.config.output_config['output_file']
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            self.logger.info(f"Report successfully generated at: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise
