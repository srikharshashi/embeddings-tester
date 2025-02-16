import json
from sentence_transformers import SentenceTransformer
from logger_service.logger import LoggerService

class EmbeddingManager:
    def __init__(self,transformer_name):
        self.embeddings = {}
        self.logger = LoggerService()
        self.transformer_name= transformer_name
        self.model = SentenceTransformer(self.transformer_name)
    
    @LoggerService.log_function(level='info')
    def load_from_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.embeddings.update(data)

    @LoggerService.log_function(level='info')
    def create_categorical_embeddings(self, categories):
        category_vectors = {
            category: [self.model.encode(keyword) for keyword in keywords]
            for category, keywords in categories.items()
        }
        self.embeddings.update(category_vectors)
        return category_vectors

    @LoggerService.log_function()
    def dump_to_json(self, file_path=f'embeddings/default.json'):
        trs_name=self.transformer_name.split('/')[1]
        file_path=f'embeddings/{trs_name}.json'
        with open(file_path, "w") as f:
            json.dump({k: [v.tolist() for v in v_list] for k, v_list in self.embeddings.items()}, f)

