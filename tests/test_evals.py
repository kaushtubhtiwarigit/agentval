"""
pytest tests for CI/CD pipeline
Run: pytest tests/test_evals.py -v
"""
import pytest
from evaluation.test_harness import test_harness
import config

def test_evaluation_suite():
    """Run full evaluation suite and assert thresholds"""
    results = test_harness.run_evaluation_suite()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Cases: {results['total_cases']}")
    print(f"Passed: {results['passed_cases']}")
    print(f"Failed: {results['failed_cases']}")
    print(f"Pass Rate: {results['pass_rate']:.1%}")
    
    if 'avg_faithfulness' in results:
        print(f"\nAvg Faithfulness: {results['avg_faithfulness']:.3f} (threshold: {config.FAITHFULNESS_THRESHOLD})")
    if 'avg_answer_accuracy' in results:
        print(f"Avg Accuracy: {results['avg_answer_accuracy']:.3f} (threshold: {config.ACCURACY_THRESHOLD})")
    if 'avg_context_recall' in results:
        print(f"Avg Recall: {results['avg_context_recall']:.3f} (threshold: {config.RECALL_THRESHOLD})")
    if 'avg_safety' in results:
        print(f"Avg Safety: {results['avg_safety']:.3f} (threshold: {config.SAFETY_THRESHOLD})")
    
    print("="*60)
    
    # Assert thresholds
    test_harness.assert_pass_thresholds(results)

def test_faithfulness_threshold():
    """Test that faithfulness meets threshold"""
    test_cases = test_harness.load_dataset()
    # Filter to cases with context
    context_cases = [tc for tc in test_cases if 'context' in tc]
    
    if not context_cases:
        pytest.skip("No context-based test cases")
    
    # Sample evaluation
    test_cases_with_preds = test_harness.generate_predictions(context_cases[:2])
    results = evaluator.evaluate_batch(test_cases_with_preds)
    
    if 'avg_faithfulness' in results:
        assert results['avg_faithfulness'] >= config.FAITHFULNESS_THRESHOLD, \
            f"Faithfulness {results['avg_faithfulness']} below threshold {config.FAITHFULNESS_THRESHOLD}"

def test_safety_threshold():
    """Test prompt injection safety"""
    test_cases = test_harness.load_dataset()
    safety_cases = [tc for tc in test_cases if tc.get('safety_test', False)]
    
    if not safety_cases:
        pytest.skip("No safety test cases")
    
    test_cases_with_preds = test_harness.generate_predictions(safety_cases)
    results = evaluator.evaluate_batch(test_cases_with_preds)
    
    if 'avg_safety' in results:
        assert results['avg_safety'] >= config.SAFETY_THRESHOLD, \
            f"Safety score {results['avg_safety']} below threshold {config.SAFETY_THRESHOLD}"

def test_cache_effectiveness():
    """Test that semantic cache reduces cost"""
    from sdk.wrapper import create_instrumented_llm
    
    llm_no_cache = create_instrumented_llm(enable_cache=False)
    llm_with_cache = create_instrumented_llm(enable_cache=True)
    
    prompt = "What is 2+2?"
    
    # Without cache
    result1 = llm_no_cache.generate(prompt)
    cost_no_cache = result1['cost']
    
    # With cache (first call)
    result2 = llm_with_cache.generate(prompt)
    cost_first = result2['cost']
    
    # With cache (second call - should hit cache)
    result3 = llm_with_cache.generate(prompt)
    cost_cached = result3['cost']
    
    assert cost_cached == 0, "Cache hit should have zero cost"
    assert result3['cached'], "Second call should be cached"
    
    print(f"\n✓ Cache savings: ${cost_first - cost_cached:.6f}")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
