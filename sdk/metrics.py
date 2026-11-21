"""
LLM-native performance metrics
"""
import psycopg2
from typing import Dict, Any, List
from datetime import datetime, timedelta
import config

class MetricsCollector:
    def __init__(self):
        self.connection = None
    
    def get_connection(self):
        """Get database connection"""
        if not self.connection or self.connection.closed:
            self.connection = psycopg2.connect(config.DATABASE_URL)
        return self.connection
    
    def record_trace(self, trace_id: str, operation_name: str, 
                    model_name: str, duration_ms: float, 
                    status: str, metadata: Dict = None):
        """Record LLM trace"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO llm_traces 
            (trace_id, operation_name, model_name, duration_ms, status, metadata, start_time, end_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trace_id) DO NOTHING
        """, (
            trace_id, operation_name, model_name, duration_ms, status,
            psycopg2.extras.Json(metadata or {}),
            datetime.now() - timedelta(milliseconds=duration_ms),
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
    
    def record_metric(self, trace_id: str, metric_name: str, 
                     metric_value: float, unit: str = 'count'):
        """Record individual metric"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO llm_metrics (trace_id, metric_name, metric_value, unit)
            VALUES (%s, %s, %s, %s)
        """, (trace_id, metric_name, metric_value, unit))
        
        conn.commit()
        cursor.close()
    
    def record_cost(self, trace_id: str, model_name: str,
                   input_tokens: int, output_tokens: int):
        """Calculate and record inference cost"""
        costs = config.MODEL_COSTS.get(model_name, config.MODEL_COSTS['gemini-1.5-flash'])
        
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        total_cost = input_cost + output_cost
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO llm_costs 
            (trace_id, model_name, input_tokens, output_tokens, total_tokens, 
             input_cost, output_cost, total_cost)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            trace_id, model_name, input_tokens, output_tokens,
            input_tokens + output_tokens, input_cost, output_cost, total_cost
        ))
        
        conn.commit()
        cursor.close()
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated metrics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        
        # Get trace statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_requests,
                AVG(duration_ms) as avg_latency,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_latency,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_latency
            FROM llm_traces
            WHERE created_at > %s
        """, (since,))
        
        trace_stats = cursor.fetchone()
        
        # Get cost statistics
        cursor.execute("""
            SELECT 
                SUM(total_cost) as total_cost,
                SUM(total_tokens) as total_tokens,
                AVG(total_cost) as avg_cost_per_request,
                COUNT(*) as total_cost_entries
            FROM llm_costs
            WHERE created_at > %s
        """, (since,))
        
        cost_stats = cursor.fetchone()
        
        # Get model usage
        cursor.execute("""
            SELECT model_name, COUNT(*) as request_count
            FROM llm_traces
            WHERE created_at > %s
            GROUP BY model_name
        """, (since,))
        
        model_usage = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.close()
        
        return {
            'time_window_hours': hours,
            'total_requests': trace_stats[0] or 0,
            'avg_latency_ms': float(trace_stats[1] or 0),
            'p95_latency_ms': float(trace_stats[2] or 0),
            'p99_latency_ms': float(trace_stats[3] or 0),
            'total_cost': float(cost_stats[0] or 0),
            'total_tokens': int(cost_stats[1] or 0),
            'avg_cost_per_request': float(cost_stats[2] or 0),
            'model_usage': model_usage
        }
    
    def get_cost_breakdown(self, hours: int = 24) -> List[Dict]:
        """Get cost breakdown by model"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        since = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT 
                model_name,
                COUNT(*) as request_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_cost) as total_cost,
                AVG(total_cost) as avg_cost
            FROM llm_costs
            WHERE created_at > %s
            GROUP BY model_name
            ORDER BY total_cost DESC
        """, (since,))
        
        breakdown = []
        for row in cursor.fetchall():
            breakdown.append({
                'model': row[0],
                'requests': row[1],
                'input_tokens': row[2],
                'output_tokens': row[3],
                'total_cost': float(row[4]),
                'avg_cost': float(row[5])
            })
        
        cursor.close()
        return breakdown

# Global metrics collector
metrics_collector = MetricsCollector()
