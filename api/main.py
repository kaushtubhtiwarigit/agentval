"""
FastAPI server for AgentEval platform
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import config
from api.models import (
    GenerateRequest, GenerateResponse,
    MetricsSummaryResponse, CostReportResponse,
    EvaluationRequest, EvaluationResponse
)
from sdk.wrapper import create_instrumented_llm
from sdk.metrics import metrics_collector
from optimization.cost_analyzer import cost_analyzer
from optimization.semantic_cache import SemanticCache
from evaluation.test_harness import test_harness

app = FastAPI(
    title="AgentEval Platform",
    description="LLM Evaluation, Guardrails & Granular Tracing",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument with OpenTelemetry
if config.ENABLE_TRACING:
    FastAPIInstrumentor.instrument_app(app)

@app.get("/")
def root():
    return {
        "service": "AgentEval Platform",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/v1/generate",
            "metrics": "/api/v1/metrics",
            "cost_report": "/api/v1/cost-report",
            "evaluate": "/api/v1/evaluate",
            "cache_stats": "/api/v1/cache/stats"
        }
    }

@app.post("/api/v1/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate with instrumented LLM"""
    try:
        llm = create_instrumented_llm(request.model)
        llm.enable_cache = request.enable_cache
        llm.enable_routing = request.enable_routing
        
        result = llm.generate(
            request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerateResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/metrics", response_model=MetricsSummaryResponse)
def get_metrics(hours: int = 24):
    """Get metrics summary"""
    try:
        summary = metrics_collector.get_metrics_summary(hours)
        return MetricsSummaryResponse(**summary)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cost-report", response_model=CostReportResponse)
def get_cost_report(hours: int = 24):
    """Get cost analysis report"""
    try:
        report = cost_analyzer.generate_cost_report(hours)
        
        # Flatten metrics for response
        report['metrics'] = {
            'avg_faithfulness': report.get('avg_faithfulness', 0),
            'avg_accuracy': report.get('avg_accuracy', 0)
        }
        
        return CostReportResponse(**report)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/evaluate", response_model=EvaluationResponse)
def run_evaluation(request: EvaluationRequest):
    """Run evaluation suite"""
    try:
        if request.use_golden_dataset or not request.test_cases:
            results = test_harness.run_evaluation_suite()
        else:
            # Custom test cases
            test_cases_with_preds = test_harness.generate_predictions(request.test_cases)
            results = evaluator.evaluate_batch(test_cases_with_preds)
        
        # Build metrics dict
        metrics = {}
        for key in ['avg_faithfulness', 'avg_context_recall', 'avg_answer_accuracy', 'avg_safety']:
            if key in results:
                metrics[key] = results[key]
        
        return EvaluationResponse(
            total_cases=results['total_cases'],
            passed_cases=results['passed_cases'],
            failed_cases=results['failed_cases'],
            pass_rate=results['pass_rate'],
            metrics=metrics,
            details=results.get('results')
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/cache/stats")
def get_cache_stats():
    """Get semantic cache statistics"""
    try:
        cache = SemanticCache()
        stats = cache.get_stats()
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/cache/clear")
def clear_cache():
    """Clear semantic cache"""
    try:
        cache = SemanticCache()
        cache.clear()
        return {"message": "Cache cleared successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
