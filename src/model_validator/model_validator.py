from typing import List, Optional
from huggingface_hub import HfApi  
from logger_service.logger import LoggerService

class ModelValidator:
    """A class to validate Hugging Face models."""
    
    def __init__(self, allowed_model_types: Optional[List[str]] = None):
        self.allowed_model_types = allowed_model_types or ['sentence-transformers']
        self.logger = LoggerService()
        
    def _validate_model_name(self, model_name: str) -> bool:
        if not isinstance(model_name, str):
            print(f"Invalid model name type: {type(model_name)}. Expected string.")
            return False
            
        if not model_name or not model_name.strip():
            print("Model name cannot be empty.")
            return False
            
        if '/' not in model_name:
            print(f"Invalid model name format: {model_name}. Expected format: 'organization/model-name'")
            return False
            
        return True

    @LoggerService.log_function(level='info')
    def validate_models(self, model_names: List[str]) -> List[str]:
        if not isinstance(model_names, list):
            self.logger.error("model_names must be a list")
            raise ValueError("model_names must be a list")

            
        valid_models = []
        api = HfApi()
        
        for model_name in model_names:
            if not self._validate_model_name(model_name):
                self.logger.error(f"Invalid model name: {model_name}")
                continue
                
            self.logger.info(f"Validating model '{model_name}'")
            try:
                # Call model_info to check if the model exists without loading it
                api.model_info(repo_id=model_name)
                valid_models.append(model_name)
                self.logger.info(f"Model '{model_name}' exists on Hugging Face Hub.")
            except Exception as e:
                self.logger.error(f"Model '{model_name}' not found or error occurred: {str(e)}")
                print(f"✗ Model '{model_name}' not found or error occurred: {str(e)}")
                
        return valid_models

    def is_valid_model(self, model_name: str) -> bool:
        return len(self.validate_models([model_name])) > 0
