import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Tuple, Any


class RecommendationEvaluator:
    def __init__(self):
        self.ground_truth = {}

    def load_ground_truth(self, file_path: str):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            topic = row['topic']
            relevant_books = eval(row['relevant_books'])
            scores = eval(row['relevance_scores'])
            self.ground_truth[topic] = list(zip(relevant_books, scores))

    def extract_book_titles(self, text: str) -> List[str]:
        patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'《([^》]+)》',
            r'Title[：:]\s*([^\n\r,，]+)',
            r'(\d+\.\s*[^\n\r]+)'
        ]

        titles = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            titles.extend(matches)

        cleaned = []
        for title in titles:
            title = title.strip().replace('\n', ' ')
            if len(title) > 3 and title not in cleaned:
                cleaned.append(title)

        return cleaned[:10]

    def calculate_precision(self, predicted: List[str], relevant: List[Tuple[str, float]], k: int = 10) -> float:
        if not predicted or not relevant:
            return 0.0

        relevant_set = set([book[0] for book in relevant])
        predicted_set = set(predicted[:k])
        intersection = predicted_set.intersection(relevant_set)

        return len(intersection) / len(predicted_set) if predicted_set else 0.0

    def calculate_recall(self, predicted: List[str], relevant: List[Tuple[str, float]], k: int = 10) -> float:
        if not predicted or not relevant:
            return 0.0

        relevant_set = set([book[0] for book in relevant])
        predicted_set = set(predicted[:k])
        intersection = predicted_set.intersection(relevant_set)

        return len(intersection) / len(relevant_set) if relevant_set else 0.0

    def calculate_f1(self, precision: float, recall: float) -> float:
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def calculate_ndcg(self, predicted: List[str], relevant: List[Tuple[str, float]], k: int = 10) -> float:
        if not predicted or not relevant:
            return 0.0

        relevance_dict = {book[0]: book[1] for book in relevant}

        dcg = 0.0
        for i, book in enumerate(predicted[:k]):
            relevance = relevance_dict.get(book, 0.0)
            dcg += relevance / np.log2(i + 2)

        sorted_relevance = sorted([score for _, score in relevant], reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance[:k]):
            idcg += relevance / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_topic(self, topic: str, prediction_text: str) -> Dict[str, float]:
        predicted_books = self.extract_book_titles(prediction_text)
        ground_truth_data = self.ground_truth.get(topic, [])

        precision = self.calculate_precision(predicted_books, ground_truth_data)
        recall = self.calculate_recall(predicted_books, ground_truth_data)
        f1 = self.calculate_f1(precision, recall)
        ndcg = self.calculate_ndcg(predicted_books, ground_truth_data)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ndcg': ndcg
        }

    def run_evaluation(self, predictions: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        results = []
        for topic, pred_data in predictions.items():
            if topic in self.ground_truth:
                recommendation_text = pred_data.get('recommendations', '')
                topic_result = self.evaluate_topic(topic, recommendation_text)
                topic_result['topic'] = topic
                results.append(topic_result)

        avg_metrics = {}
        if results:
            for metric in ['precision', 'recall', 'f1', 'ndcg']:
                avg_metrics[f'avg_{metric}'] = np.mean([r[metric] for r in results])

        return {
            'individual_results': results,
            'average_metrics': avg_metrics
        }

    def save_results(self, results: Dict[str, Any], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)