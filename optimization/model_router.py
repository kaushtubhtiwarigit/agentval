"""
Tiered model routing (Flash vs Pro) based on query complexity
"""
import google.generativeai as genai
from typing import Dict
import config

genai.configure(api_key=config.GOOGLE_API_KEY)

class ModelRouter:
    """Route queries to appropriate model tier"""
    
    def __init__(self):
        self.complexity_threshold = config.ROUTING_COMPLEXITY_THRESHOLD
        self.simple_model = config.DEFAULT_MODEL  # gemini-1.5-flash
        self.complex_model = config.COMPLEX_MODEL  # gemini-1.5-pro
        
        # Use a lightweight model for classification
        self.classifier = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_complexity(self, prompt: str) -> float:
        """
        Analyze query complexity
        Returns: complexity score 0.0-1.0
        """
        analysis_prompt = f"""Analyze the complexity of this query on a scale of 0.0 to 1.0:

0.0 = Simple (factual lookup, yes/no, basic classification)
0.5 = Medium (requires reasoning, multiple steps)
1.0 = Complex (deep analysis, synthesis, creative reasoning)

Query: {prompt}

Respond with ONLY a number between 0.0 and 1.0, nothing else."""
        
        try:
            response = self.classifier.generate_content(analysis_prompt)
            score_text = response.text.strip()
            
            # Parse score
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        
        except Exception as e:
            print(f"[Router] Complexity analysis failed: {e}")
            # Default to medium complexity
            return 0.5
    
    def route(self, prompt: str) -> Dict[str, str]:
        """
        Route query to appropriate model
        Returns: {model, complexity_score, reasoning}
        """
        # Heuristic-based fast routing
        prompt_lower = prompt.lower()
        
        # Fast path for obviously simple queries
        simple_indicators = ['what is', 'who is', 'when is', 'yes or no', 'true or false']
        if any(indicator in prompt_lower for indicator in simple_indicators) and len(prompt.split()) < 15:
            return {
                'model': self.simple_model,
                'complexity_score': 0.2,
                'reasoning': 'Simple factual query'
            }
        
        # Fast path for obviously complex queries
        complex_indicators = ['analyze', 'synthesize', 'compare and contrast', 'evaluate', 'reasoning']
        if any(indicator in prompt_lower for indicator in complex_indicators) and len(prompt.split()) > 30:
            return {
                'model': self.complex_model,
                'complexity_score': 0.9,
                'reasoning': 'Complex analytical query'
            }
        
        # Use LLM-based complexity analysis for edge cases
        complexity_score = self.analyze_complexity(prompt)
        
        if complexity_score >= self.complexity_threshold:
            selected_model = self.complex_model
            reasoning = f'High complexity ({complexity_score:.2f})'
        else:
            selected_model = self.simple_model
            reasoning = f'Low-medium complexity ({complexity_score:.2f})'
        
        return {
            'model': selected_model,
            'complexity_score': complexity_score,
            'reasoning': reasoning
        }
    
    def get_cost_savings(self, total_requests: int, flash_count: int, pro_count: int) -> Dict:
        """
        Calculate cost savings from routing vs using Pro for everything
        """
        # Average tokens per request (estimate)
        avg_input_tokens = 500
        avg_output_tokens = 200
        
        # Actual cost with routing
        flash_cost = flash_count * (
            (avg_input_tokens / 1000) * config.GEMINI_FLASH_INPUT_COST_PER_1K +
            (avg_output_tokens / 1000) * config.GEMINI_FLASH_OUTPUT_COST_PER_1K
        )
        
        pro_cost = pro_count * (
            (avg_input_tokens / 1000) * config.GEMINI_PRO_INPUT_COST_PER_1K +
            (avg_output_tokens / 1000) * config.GEMINI_PRO_OUTPUT_COST_PER_1K
        )
        
        actual_cost = flash_cost + pro_cost
        
        # Cost if we used Pro for everything
        all_pro_cost = total_requests * (
            (avg_input_tokens / 1000) * config.GEMINI_PRO_INPUT_COST_PER_1K +
            (avg_output_tokens / 1000) * config.GEMINI_PRO_OUTPUT_COST_PER_1K
        )
        
        savings = all_pro_cost - actual_cost
        savings_percent = (savings / all_pro_cost * 100) if all_pro_cost > 0 else 0
        
        return {
            'actual_cost': actual_cost,
            'all_pro_cost': all_pro_cost,
            'savings': savings,
            'savings_percent': savings_percent,
            'flash_requests': flash_count,
            'pro_requests': pro_count
        }
