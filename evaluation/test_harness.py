"""
pytest integration for CI/CD evaluation pipeline
"""
import json
from pathlib import Path
from typing import List, Dict
from evaluation.metrics_eval import evaluator
from sdk.wrapper import create_instrumented_llm
import config

class TestHarness:
    """Load and run evaluation test suites"""
    
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or 'tests/golden_dataset.json'
        self.llm = create_instrumented_llm()
    
    def load_dataset(self) -> List[Dict]:
        """Load golden dataset"""
        path = Path(self.dataset_path)
        if not path.exists():
            print(f"⚠️  Dataset not found: {self.dataset_path}")
            return []
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data.get('test_cases', [])
    
    def generate_predictions(self, test_cases: List[Dict]) -> List[Dict]:
        """Generate predictions for test cases"""
        print(f"\n🤖 Generating predictions for {len(test_cases)} cases...\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Generating...", end=" ")
            
            # Build prompt (with context if provided)
            prompt = test_case['question']
            if 'context' in test_case:
                prompt = f"Context: {test_case['context']}\n\nQuestion: {test_case['question']}"
            
            # Generate answer
            result = self.llm.generate(prompt)
            test_case['predicted_answer'] = result['response']
            
            print(f"✓ ({result['tokens']} tokens, ${result['cost']:.6f})")
        
        return test_cases
    
    def run_evaluation_suite(self) -> Dict:
        """Run full evaluation: generate + evaluate"""
        # Load test cases
        test_cases = self.load_dataset()
        
        if not test_cases:
            return {
                'error': 'No test cases loaded',
                'passed': False
            }
        
        # Generate predictions
        test_cases_with_predictions = self.generate_predictions(test_cases)
        
        # Evaluate
        results = evaluator.evaluate_batch(test_cases_with_predictions)
        
        return results
    
    def assert_pass_thresholds(self, results: Dict):
        """Assert evaluation meets thresholds (for pytest)"""
        assertions = []
        
        # Check pass rate
        if results['pass_rate'] < 0.8:  # 80% pass threshold
            assertions.append(f"Pass rate {results['pass_rate']:.1%} below 80% threshold")
        
        # Check individual metrics
        if 'avg_faithfulness' in results:
            if results['avg_faithfulness'] < config.FAITHFULNESS_THRESHOLD:
                assertions.append(f"Faithfulness {results['avg_faithfulness']:.2f} below threshold")
        
        if 'avg_answer_accuracy' in results:
            if results['avg_answer_accuracy'] < config.ACCURACY_THRESHOLD:
                assertions.append(f"Accuracy {results['avg_answer_accuracy']:.2f} below threshold")
        
        if 'avg_safety' in results:
            if results['avg_safety'] < config.SAFETY_THRESHOLD:
                assertions.append(f"Safety {results['avg_safety']:.2f} below threshold")
        
        if assertions:
            raise AssertionError("\n❌ Evaluation Failed:\n" + "\n".join(f"  - {a}" for a in assertions))
        
        print("\n✅ All evaluation thresholds passed!")

# Global test harness
test_harness = TestHarness()
