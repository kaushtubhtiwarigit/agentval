"""
Evaluation metrics orchestration
"""
from typing import Dict, List
from evaluation.judge import llm_judge
import config

class EvaluationMetrics:
    """Run comprehensive evaluation on test cases"""
    
    def __init__(self):
        self.judge = llm_judge
        self.thresholds = {
            'faithfulness': config.FAITHFULNESS_THRESHOLD,
            'context_recall': config.RECALL_THRESHOLD,
            'answer_accuracy': config.ACCURACY_THRESHOLD,
            'safety': config.SAFETY_THRESHOLD
        }
    
    def evaluate_case(self, test_case: Dict) -> Dict:
        """
        Evaluate a single test case across all metrics
        
        test_case format:
        {
            'id': str,
            'question': str,
            'context': str (optional),
            'ground_truth': str,
            'predicted_answer': str,
            'safety_test': bool (optional)
        }
        """
        results = {
            'test_case_id': test_case['id'],
            'metrics': {},
            'passed': True
        }
        
        # Faithfulness (if context provided)
        if 'context' in test_case:
            faith_result = self.judge.evaluate_faithfulness(
                test_case['question'],
                test_case['context'],
                test_case['predicted_answer']
            )
            results['metrics']['faithfulness'] = faith_result['score']
            
            if faith_result['score'] < self.thresholds['faithfulness']:
                results['passed'] = False
        
        # Context Recall (if context provided)
        if 'context' in test_case and 'ground_truth' in test_case:
            recall_result = self.judge.evaluate_context_recall(
                test_case['question'],
                test_case['context'],
                test_case['ground_truth']
            )
            results['metrics']['context_recall'] = recall_result['score']
            
            if recall_result['score'] < self.thresholds['context_recall']:
                results['passed'] = False
        
        # Answer Accuracy
        if 'ground_truth' in test_case:
            accuracy_result = self.judge.evaluate_answer_accuracy(
                test_case['question'],
                test_case['predicted_answer'],
                test_case['ground_truth']
            )
            results['metrics']['answer_accuracy'] = accuracy_result['score']
            
            if accuracy_result['score'] < self.thresholds['accuracy']:
                results['passed'] = False
        
        # Safety (prompt injection detection)
        if test_case.get('safety_test', False):
            safety_result = self.judge.evaluate_prompt_injection_safety(
                test_case['question'],
                test_case['predicted_answer']
            )
            results['metrics']['safety'] = safety_result['score']
            
            if safety_result['score'] < self.thresholds['safety']:
                results['passed'] = False
        
        return results
    
    def evaluate_batch(self, test_cases: List[Dict]) -> Dict:
        """Evaluate multiple test cases"""
        all_results = []
        passed_count = 0
        
        metric_totals = {
            'faithfulness': [],
            'context_recall': [],
            'answer_accuracy': [],
            'safety': []
        }
        
        print(f"\n🧪 Evaluating {len(test_cases)} test cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test_case['id']}...", end=" ")
            
            result = self.evaluate_case(test_case)
            all_results.append(result)
            
            if result['passed']:
                passed_count += 1
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Aggregate metrics
            for metric, score in result['metrics'].items():
                metric_totals[metric].append(score)
        
        # Calculate averages
        avg_metrics = {}
        for metric, scores in metric_totals.items():
            if scores:
                avg_metrics[f'avg_{metric}'] = sum(scores) / len(scores)
        
        summary = {
            'total_cases': len(test_cases),
            'passed_cases': passed_count,
            'failed_cases': len(test_cases) - passed_count,
            'pass_rate': passed_count / len(test_cases) if test_cases else 0,
            **avg_metrics,
            'results': all_results
        }
        
        return summary

# Global evaluator
evaluator = EvaluationMetrics()
