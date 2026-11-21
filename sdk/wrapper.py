"""
Instrumented LLM wrapper with tracing, caching, and routing
"""
import google.generativeai as genai
from typing import Optional, Dict, Any
import uuid
import config
from sdk.tracer import LLMTracer
from sdk.metrics import metrics_collector
from optimization.semantic_cache import SemanticCache
from optimization.model_router import ModelRouter

genai.configure(api_key=config.GOOGLE_API_KEY)

class InstrumentedLLM:
    """Wrapped LLM with automatic tracing and optimization"""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 enable_cache: bool = True,
                 enable_routing: bool = True):
        self.model_name = model_name or config.DEFAULT_MODEL
        self.enable_cache = enable_cache and config.ENABLE_SEMANTIC_CACHE
        self.enable_routing = enable_routing and config.ENABLE_MODEL_ROUTING
        
        self.cache = SemanticCache() if self.enable_cache else None
        self.router = ModelRouter() if self.enable_routing else None
        
        self.model = genai.GenerativeModel(self.model_name)
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.7,
                max_tokens: int = 1024,
                metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate with full instrumentation
        Returns: {response, trace_id, metrics, cached}
        """
        trace_id = str(uuid.uuid4())
        
        # Check cache first
        if self.enable_cache:
            cached_result = self.cache.get(prompt)
            if cached_result:
                print(f"[Cache HIT] Similarity: {cached_result['similarity']:.3f}")
                
                # Record cache hit
                with LLMTracer(
                    'llm.generate.cached',
                    self.model_name,
                    prompt_length=len(prompt)
                ) as trace:
                    trace.record_cache_hit(True, cached_result['similarity'])
                    trace.record_tokens(0, 0)  # No tokens used
                    trace.record_cost(0, 0)
                
                return {
                    'response': cached_result['response'],
                    'trace_id': trace_id,
                    'cached': True,
                    'cache_similarity': cached_result['similarity'],
                    'cost': 0,
                    'tokens': 0
                }
        
        # Model routing
        selected_model = self.model_name
        complexity_score = 0.5
        
        if self.enable_routing:
            routing_decision = self.router.route(prompt)
            selected_model = routing_decision['model']
            complexity_score = routing_decision['complexity_score']
            
            if selected_model != self.model_name:
                print(f"[Router] {self.model_name} → {selected_model} (complexity: {complexity_score:.2f})")
                self.model = genai.GenerativeModel(selected_model)
        
        # Generate with tracing
        with LLMTracer(
            'llm.generate',
            selected_model,
            prompt_length=len(prompt),
            temperature=temperature,
            max_tokens=max_tokens
        ) as trace:
            try:
                # Call LLM
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                
                # Mark first token (approximation for non-streaming)
                trace.mark_first_token()
                
                response_text = response.text
                
                # Count tokens (approximation)
                input_tokens = len(prompt.split()) * 1.3  # Rough estimate
                output_tokens = len(response_text.split()) * 1.3
                
                trace.record_tokens(int(input_tokens), int(output_tokens))
                
                # Calculate cost
                cost_info = metrics_collector.record_cost(
                    trace_id,
                    selected_model,
                    int(input_tokens),
                    int(output_tokens)
                )
                
                trace.record_cost(cost_info['input_cost'], cost_info['output_cost'])
                
                # Record routing decision
                if self.enable_routing:
                    trace.record_routing_decision(selected_model, complexity_score)
                
                # Cache result
                if self.enable_cache:
                    self.cache.set(prompt, response_text)
                
                return {
                    'response': response_text,
                    'trace_id': trace_id,
                    'cached': False,
                    'model': selected_model,
                    'cost': cost_info['total_cost'],
                    'tokens': int(input_tokens + output_tokens),
                    'input_tokens': int(input_tokens),
                    'output_tokens': int(output_tokens)
                }
            
            except Exception as e:
                print(f"[LLM Error] {str(e)}")
                raise

def create_instrumented_llm(model_name: Optional[str] = None) -> InstrumentedLLM:
    """Factory function"""
    return InstrumentedLLM(model_name)
