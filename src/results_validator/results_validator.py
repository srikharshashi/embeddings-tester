from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from logger_service.logger import LoggerService
from configuration_manager.config_manager import ConfigManager
import json

class ResultsValidator:
    def __init__(self, model_manager):
        self.model = model_manager.model
        self.category_embeddings = model_manager.embeddings
        self.logger = LoggerService()
        self.config = ConfigManager()
        self.query_vectors = {}
        self.test_transactions = self._load_test_transactions()
        self.true_labels = self._load_true_labels()

    @LoggerService.log_function(level='info')
    def _load_test_transactions(self):
        """Load test transactions (keys) from config file path"""
        transactions_file = self.config.test_data_config['transactions_file']
        with open(transactions_file, 'r') as f:
            data = json.load(f)
        # Extract transaction texts (keys) and assign numeric indices
        return {str(idx): text for idx, text in enumerate(data.keys())}

    @LoggerService.log_function(level='info')
    def _load_true_labels(self):
        """Load ground truth labels (values) from the same transactions file"""
        transactions_file = self.config.test_data_config['transactions_file']
        with open(transactions_file, 'r') as f:
            data = json.load(f)
        # Extract labels (values) and maintain same numeric indices
        return {str(idx): label for idx, label in enumerate(data.values())}

    @LoggerService.log_function(level='info')
    def generate_query_vectors(self):
        """Generate embeddings for test transactions"""
        self.query_vectors = {
            trans_id: self.model.encode(trans_text)
            for trans_id, trans_text in self.test_transactions.items()
        }
        return self.query_vectors
    
    @LoggerService.log_function(level='info')
    def calculate_similarities(self):
        """Calculate cosine similarities between query vectors and category embeddings"""
        if not self.query_vectors:
            self.generate_query_vectors()
            
        results = {}
        for trans_id, query_vector in self.query_vectors.items():
            category_scores = {}
            for category, embeddings in self.category_embeddings.items():
                scores = [
                    cosine_similarity([query_vector], [kw_emb])[0][0] 
                    for kw_emb in embeddings
                ]
                category_scores[category] = max(scores)
            
            best_match = max(category_scores, key=category_scores.get)
            results[trans_id] = {
                'category': best_match,
                'confidence': float(format(float(category_scores[best_match]), '.5f')),
                'actual_text': self.test_transactions[trans_id]
            }
        
        return results

    @LoggerService.log_function(level='info')
    def calculate_accuracy(self, results):
        """Calculate classification accuracy"""
        if not self.true_labels:
            self.logger.warning("No true labels available for accuracy calculation")
            return None
            
        y_true = []
        y_pred = []
        for idx in range(len(results)):
            str_idx = str(idx)
            if str_idx in self.true_labels and str_idx in results:
                y_true.append(self.true_labels[str_idx])
                y_pred.append(results[str_idx]['category'])
        
        return {
            'accuracy': float(format(accuracy_score(y_true, y_pred), '.4f'))
        }

    @LoggerService.log_function(level='info')
    def calculate_precision_recall_f1(self, results):
        """Calculate precision, recall, and F1 score for each category"""
        y_true = []
        y_pred = []
        for idx in range(len(results)):
            str_idx = str(idx)
            if str_idx in self.true_labels and str_idx in results:
                y_true.append(self.true_labels[str_idx])
                y_pred.append(results[str_idx]['category'])
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        return {
            'precision': float(format(precision, '.4f')),
            'recall': float(format(recall, '.4f')),
            'f1_score': float(format(f1, '.4f'))
        }

    @LoggerService.log_function(level='info')
    def generate_confusion_matrix(self, results):
        """Generate and save confusion matrix visualization"""
        y_true = []
        y_pred = []
        for idx in range(len(results)):
            str_idx = str(idx)
            if str_idx in self.true_labels and str_idx in results:
                y_true.append(self.true_labels[str_idx])
                y_pred.append(results[str_idx]['category'])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        unique_categories = sorted(set(y_true + y_pred))
        

        
        return {
            'confusion_matrix': cm.tolist(),
            'categories': unique_categories
        }

    @LoggerService.log_function(level='info')
    def calculate_confidence_stats(self, results):
        """Calculate confidence score statistics"""
        confidences = [result['confidence'] for result in results.values()]
        
        return {
            'mean_confidence': float(format(np.mean(confidences), '.4f')),
            'median_confidence': float(format(np.median(confidences), '.4f')),
            'min_confidence': float(format(min(confidences), '.4f')),
            'max_confidence': float(format(max(confidences), '.4f'))
        }

    @LoggerService.log_function(level='info')
    def evaluate_all_metrics(self, results):
        """Calculate all classification metrics"""
        metrics = {
            'basic_metrics': self.calculate_accuracy(results),
            'detailed_metrics': self.calculate_precision_recall_f1(results),
            'confidence_stats': self.calculate_confidence_stats(results),
            'confusion_matrix_data': self.generate_confusion_matrix(results)
        }
        
        return metrics
