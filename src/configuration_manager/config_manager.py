import json
import os
from typing import List, Dict, Any

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        self.config_data = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON configuration file")
    
    def _create_directories(self):
        directories = [
            self.config_data['embedding_settings']['embeddings_output_dir'],
            os.path.dirname(self.config_data['logging']['log_file']),
            os.path.dirname(self.config_data['output']['output_file']),
            self.config_data['output']['image_storage'],
            os.path.dirname(self.config_data['test_data']['transactions_file']),
            os.path.dirname(self.config_data['test_data']['categories_file'])
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @property
    def transformer_models(self) -> List[str]:
        return self.config_data['models']['transformer_models']
    
    @property
    def default_model(self) -> str:
        return self.config_data['models']['default_model']
    
    @property
    def embeddings_output_dir(self) -> str:
        return self.config_data['embedding_settings']['embeddings_output_dir']
    
    @property
    def default_embedding_file(self) -> str:
        return self.config_data['embedding_settings']['default_embedding_file']
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        return self.config_data['logging']
    
    @property
    def test_data_config(self) -> Dict[str, Any]:
        return self.config_data['test_data']
    
    @property
    def output_config(self) -> Dict[str, Any]:
        return self.config_data['output']

