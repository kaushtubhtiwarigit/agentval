"""
OpenTelemetry instrumentation for LLM calls
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode
from datetime import datetime
import time
import config

# Initialize OpenTelemetry
if config.ENABLE_TRACING:
    trace.set_tracer_provider(TracerProvider())
    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{config.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces"
    )
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

tracer = trace.get_tracer(config.OTEL_SERVICE_NAME)

class LLMTracer:
    """Context manager for tracing LLM operations"""
    
    def __init__(self, operation_name: str, model_name: str, **attributes):
        self.operation_name = operation_name
        self.model_name = model_name
        self.attributes = attributes
        self.span = None
        self.start_time = None
        self.first_token_time = None
    
    def __enter__(self):
        self.span = tracer.start_span(self.operation_name)
        self.start_time = time.time()
        
        # Set standard attributes
        self.span.set_attribute('llm.model', self.model_name)
        self.span.set_attribute('llm.operation', self.operation_name)
        
        for key, value in self.attributes.items():
            self.span.set_attribute(f'llm.{key}', str(value))
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(Status(StatusCode.OK))
        
        # Calculate duration
        duration_ms = (time.time() - self.start_time) * 1000
        self.span.set_attribute('llm.duration_ms', duration_ms)
        
        self.span.end()
        return False
    
    def mark_first_token(self):
        """Mark Time To First Token (TTFT)"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            ttft_ms = (self.first_token_time - self.start_time) * 1000
            self.span.set_attribute('llm.ttft_ms', ttft_ms)
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token usage"""
        self.span.set_attribute('llm.input_tokens', input_tokens)
        self.span.set_attribute('llm.output_tokens', output_tokens)
        self.span.set_attribute('llm.total_tokens', input_tokens + output_tokens)
        
        # Calculate tokens per second
        if self.start_time:
            duration_s = time.time() - self.start_time
            tokens_per_sec = output_tokens / duration_s if duration_s > 0 else 0
            self.span.set_attribute('llm.tokens_per_sec', tokens_per_sec)
    
    def record_cost(self, input_cost: float, output_cost: float):
        """Record inference cost"""
        self.span.set_attribute('llm.input_cost', input_cost)
        self.span.set_attribute('llm.output_cost', output_cost)
        self.span.set_attribute('llm.total_cost', input_cost + output_cost)
    
    def record_cache_hit(self, hit: bool, similarity: float = None):
        """Record cache hit/miss"""
        self.span.set_attribute('llm.cache_hit', hit)
        if similarity is not None:
            self.span.set_attribute('llm.cache_similarity', similarity)
    
    def record_routing_decision(self, selected_model: str, complexity_score: float):
        """Record model routing decision"""
        self.span.set_attribute('llm.routed_to', selected_model)
        self.span.set_attribute('llm.complexity_score', complexity_score)

def create_span(operation_name: str, **attributes):
    """Create a simple span without context manager"""
    span = tracer.start_span(operation_name)
    for key, value in attributes.items():
        span.set_attribute(key, str(value))
    return span
