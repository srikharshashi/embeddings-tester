# Test Embeddings Evaluation Project

## Overview
This project evaluates text embeddings and models using various metrics and visualizations. It includes modules to process test transactions, validate results, generate plots, and produce a comprehensive HTML report.

## Project Structure
- **config.json**: Contains configuration settings for models, embeddings, test data, output, and logging.
- **src/**
  - **results_validator/**: Validates results and computes metrics (e.g., accuracy, precision-recall, confusion matrix).
  - **plot_generator/**: Generates visualizations including confusion matrices, ROC curves, precision-recall curves, and error analysis.
  - **model_validator/**: Validates Hugging Face models.
  - **output_generator/**: Generates HTML reports using Jinja2 templates.
  - **configuration_manager/**: Manages configuration loading and directory creation.
  - **embedding_manager/**: (Referenced for creating embeddings.)
  - **main.py**: Entry point that orchestrates model evaluation, plotting, and report generation.
- **input/**: Contains JSON input files (e.g., testtxns.json, categories.json).
- **output/**: Stores the generated HTML report and images.
- **logs/**: Stores application log files.

## Setup and Usage
1. **Install dependencies:**  
   Ensure Python is installed, then install required packages (e.g., scikit-learn, numpy, matplotlib, pandas, jinja2, huggingface_hub):
   ```
   pip install -r requirements.txt
   ```
2. **Configure the project:**  
   Update settings in `config.json` (located in the project root) to match your environment for models, file paths, and logging.
3. **Prepare Input Data:**  
   Place your test transactions and categories JSON files in the `input/` folder.
4. **Run the evaluation:**  
   Execute the main module:
   ```
   python src/main.py
   ```
5. **View the Report:**  
   The output HTML report and generated images will be found in the `output/` folder.

## Configuration Details
- **Models:** Configured under the "models" key in `config.json`.
- **Embeddings:** Managed by the embedding_manager and stored in the directory specified under "embedding_settings."
- **Logging:** Configured via `config.json` and logs are stored in the logs folder.
- **Output:** HTML report and images are generated in the output folder based on settings in `config.json`.

