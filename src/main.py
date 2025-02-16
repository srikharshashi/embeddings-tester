import logging
import os
import json
from logger_service.logger import LoggerService
from embedding_manager.embedding_manager import EmbeddingManager
from model_validator.model_validator import ModelValidator
from configuration_manager.config_manager import ConfigManager
from results_validator.results_validator import ResultsValidator
from plot_generator.plot_generator import PlotGenerator
from plot_generator.model_comparison_plotter import ModelComparisonPlotter
from output_generator.output_generator import OutputGenerator

@LoggerService.log_function(level='info')
def main():
    config = ConfigManager()
    
    # Load test data
    with open(config.test_data_config['categories_file'], 'r') as f:
        categories_data = json.load(f)
    
    validator = ModelValidator()
    valid_models = validator.validate_models(config.transformer_models)
    
    # Dictionary to store results for all models
    all_model_results = {}
    
    for valid_model in valid_models:
        mgr = EmbeddingManager(valid_model)
        mgr.create_categorical_embeddings(categories_data)
        
        model_obj = {
            "mgr": mgr,
            "results": {}
        }
        
        validator = ResultsValidator(model_manager=mgr)
        model_obj['results']['raw_results'] = validator.calculate_similarities()
        model_obj['results']['final_results'] = validator.evaluate_all_metrics(
            model_obj['results']['raw_results']
        )
        
        model_name_trunc = valid_model.split('/')[1]
        plot_generator = PlotGenerator(model_name_trunc)
        model_obj['results']['plots'] = plot_generator.generate_all_plots(
            model_obj['results']['raw_results'],
            model_obj['results']['final_results']
        )
        
        # Store results for comparison
        all_model_results[model_name_trunc] = model_obj['results']
    
    # Generate comparison plots
    comparison_plotter = ModelComparisonPlotter(config.output_config['image_storage'])
    comparison_plots = comparison_plotter.generate_all_comparison_plots(all_model_results)
    
    # Generate HTML report
    output_generator = OutputGenerator()
    report_path = output_generator.generate_report(all_model_results, comparison_plots)
    
    logger = LoggerService()
    logger.info(f"Evaluation complete. Report generated at: {report_path}")

if __name__ == "__main__":
    main()


