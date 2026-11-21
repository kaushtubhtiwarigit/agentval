-- AgentEval database schema

CREATE TABLE IF NOT EXISTS llm_traces (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(100) UNIQUE NOT NULL,
    parent_span_id VARCHAR(100),
    operation_name VARCHAR(200),
    model_name VARCHAR(100),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_ms DECIMAL(10,2),
    status VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_metrics (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(100) REFERENCES llm_traces(trace_id),
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,6),
    unit VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_costs (
    id SERIAL PRIMARY KEY,
    trace_id VARCHAR(100) REFERENCES llm_traces(trace_id),
    model_name VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    input_cost DECIMAL(10,6),
    output_cost DECIMAL(10,6),
    total_cost DECIMAL(10,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    run_type VARCHAR(50),
    total_cases INTEGER,
    passed_cases INTEGER,
    failed_cases INTEGER,
    avg_faithfulness DECIMAL(5,4),
    avg_recall DECIMAL(5,4),
    avg_accuracy DECIMAL(5,4),
    avg_safety DECIMAL(5,4),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50),
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES evaluation_runs(run_id),
    test_case_id VARCHAR(100),
    metric_name VARCHAR(100),
    score DECIMAL(5,4),
    passed BOOLEAN,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cache_hits (
    id SERIAL PRIMARY KEY,
    cache_key VARCHAR(200),
    similarity_score DECIMAL(5,4),
    hit_count INTEGER DEFAULT 1,
    last_hit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_traces_trace_id ON llm_traces(trace_id);
CREATE INDEX idx_traces_created ON llm_traces(created_at DESC);
CREATE INDEX idx_metrics_trace ON llm_metrics(trace_id);
CREATE INDEX idx_costs_trace ON llm_costs(trace_id);
CREATE INDEX idx_costs_model ON llm_costs(model_name);
CREATE INDEX idx_eval_runs_id ON evaluation_runs(run_id);
CREATE INDEX idx_eval_results_run ON evaluation_results(run_id);
