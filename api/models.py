"""
Pydantic models for API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for generation")
    model: Optional[str] = Field(None, description="Model name (optional)")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1024, ge=1, le=8192)
    enable_cache: Optional[bool] = Field(True)
    enable_routing: Optional[bool] = Field(True)

class GenerateResponse(BaseModel):
    response: str
    trace_id: str
    model: str
    cached: bool
    cost: float
    tokens: int
    input_tokens: int
    output_tokens: int

class MetricsSummaryResponse(BaseModel):
    time_window_hours: int
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cost: float
    total_tokens: int
    avg_cost_per_request: float
    model_usage: Dict[str, int]

class CostReportResponse(BaseModel):
    time_window_hours: int
    summary: Dict[str, Any]
    by_model: List[Dict]
    most_expensive_model: Optional[str]
    recommendations: List[str]

class EvaluationRequest(BaseModel):
    test_cases: Optional[List[Dict]] = Field(None, description="Custom test cases")
    use_golden_dataset: Optional[bool] = Field(True)

class EvaluationResponse(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    metrics: Dict[str, float]
    details: Optional[List[Dict]] = None
