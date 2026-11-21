"""
Cost analysis and reporting
"""
from typing import Dict, List
from datetime import datetime, timedelta
import config
from sdk.metrics import metrics_collector

class CostAnalyzer:
    """Analyze and report on inference costs"""
    
    def __init__(self):
        self.collector = metrics_collector
    
    def generate_cost_report(self, hours: int = 24) -> Dict:
        """Generate comprehensive cost report"""
        summary = self.collector.get_metrics_summary(hours)
        breakdown = self.collector.get_cost_breakdown(hours)
        
        # Calculate cost per request
        cost_per_request = summary['avg_cost_per_request']
        
        # Calculate tokens per dollar
        tokens_per_dollar = summary['total_tokens'] / summary['total_cost'] if summary['total_cost'] > 0 else 0
        
        # Identify most expensive operations
        most_expensive_model = max(breakdown, key=lambda x: x['total_cost']) if breakdown else None
        
        report = {
            'time_window_hours': hours,
            'summary': {
                'total_cost': summary['total_cost'],
                'total_requests': summary['total_requests'],
                'avg_cost_per_request': cost_per_request,
                'total_tokens': summary['total_tokens'],
                'tokens_per_dollar': tokens_per_dollar
            },
            'by_model': breakdown,
            'most_expensive_model': most_expensive_model['model'] if most_expensive_model else None,
            'recommendations': self._generate_recommendations(summary, breakdown)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict, breakdown: List[Dict]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # Check if Pro model is being overused
        pro_usage = next((m for m in breakdown if 'pro' in m['model'].lower()), None)
        flash_usage = next((m for m in breakdown if 'flash' in m['model'].lower()), None)
        
        if pro_usage and flash_usage:
            pro_percent = (pro_usage['requests'] / summary['total_requests'] * 100)
            if pro_percent > 50:
                recommendations.append(
                    f"⚠️  High Pro model usage ({pro_percent:.1f}%). Consider routing more queries to Flash."
                )
        
        # Check avg cost per request
        if summary['avg_cost_per_request'] > 0.01:
            recommendations.append(
                f"⚠️  High average cost per request (${summary['avg_cost_per_request']:.4f}). Enable semantic caching."
            )
        
        # Check if caching is effective
        if not config.ENABLE_SEMANTIC_CACHE:
            recommendations.append(
                "💡 Enable semantic caching to reduce duplicate processing."
            )
        
        if not config.ENABLE_MODEL_ROUTING:
            recommendations.append(
                "💡 Enable model routing to automatically use Flash for simple queries."
            )
        
        # Good performance
        if not recommendations:
            recommendations.append("✓ Cost optimization looks good!")
        
        return recommendations
    
    def compare_periods(self, current_hours: int = 24, previous_hours: int = 24) -> Dict:
        """Compare current period to previous period"""
        current = self.collector.get_metrics_summary(current_hours)
        
        # Get previous period (offset by current_hours)
        # Note: This is simplified - in production, query with specific date ranges
        previous = self.collector.get_metrics_summary(previous_hours)
        
        cost_change = current['total_cost'] - previous['total_cost']
        cost_change_percent = (cost_change / previous['total_cost'] * 100) if previous['total_cost'] > 0 else 0
        
        latency_change = current['p95_latency_ms'] - previous['p95_latency_ms']
        latency_change_percent = (latency_change / previous['p95_latency_ms'] * 100) if previous['p95_latency_ms'] > 0 else 0
        
        return {
            'current_period': current,
            'previous_period': previous,
            'cost_change': cost_change,
            'cost_change_percent': cost_change_percent,
            'latency_change_ms': latency_change,
            'latency_change_percent': latency_change_percent,
            'improved': cost_change < 0 and latency_change < 0
        }

# Global instance
cost_analyzer = CostAnalyzer()
