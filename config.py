import os
from dotenv import load_dotenv

load_dotenv()

# Google Cloud / Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', '')

# Database
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
POSTGRES_DB = os.getenv('POSTGRES_DB', 'agenteval')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'eval_user')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')

DATABASE_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'

# Redis
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:4318')
OTEL_SERVICE_NAME = os.getenv('OTEL_SERVICE_NAME', 'agenteval')
ENABLE_TRACING = os.getenv('ENABLE_TRACING', 'true').lower() == 'true'

# Model Configuration
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gemini-1.5-flash')
COMPLEX_MODEL = os.getenv('COMPLEX_MODEL', 'gemini-1.5-pro')
ENABLE_MODEL_ROUTING = os.getenv('ENABLE_MODEL_ROUTING', 'true').lower() == 'true'
ROUTING_COMPLEXITY_THRESHOLD = float(os.getenv('ROUTING_COMPLEXITY_THRESHOLD', 0.7))

# Semantic Cache
ENABLE_SEMANTIC_CACHE = os.getenv('ENABLE_SEMANTIC_CACHE', 'true').lower() == 'true'
CACHE_SIMILARITY_THRESHOLD = float(os.getenv('CACHE_SIMILARITY_THRESHOLD', 0.95))
CACHE_TTL_SECONDS = int(os.getenv('CACHE_TTL_SECONDS', 3600))

# Cost Configuration (per 1K tokens)
GEMINI_FLASH_INPUT_COST_PER_1K = float(os.getenv('GEMINI_FLASH_INPUT_COST_PER_1K', 0.00015))
GEMINI_FLASH_OUTPUT_COST_PER_1K = float(os.getenv('GEMINI_FLASH_OUTPUT_COST_PER_1K', 0.0006))
GEMINI_PRO_INPUT_COST_PER_1K = float(os.getenv('GEMINI_PRO_INPUT_COST_PER_1K', 0.00125))
GEMINI_PRO_OUTPUT_COST_PER_1K = float(os.getenv('GEMINI_PRO_OUTPUT_COST_PER_1K', 0.005))

MODEL_COSTS = {
    'gemini-1.5-flash': {
        'input': GEMINI_FLASH_INPUT_COST_PER_1K,
        'output': GEMINI_FLASH_OUTPUT_COST_PER_1K
    },
    'gemini-1.5-pro': {
        'input': GEMINI_PRO_INPUT_COST_PER_1K,
        'output': GEMINI_PRO_OUTPUT_COST_PER_1K
    }
}

# Evaluation
EVAL_JUDGE_MODEL = os.getenv('EVAL_JUDGE_MODEL', 'gemini-1.5-pro')
FAITHFULNESS_THRESHOLD = float(os.getenv('FAITHFULNESS_THRESHOLD', 0.8))
RECALL_THRESHOLD = float(os.getenv('RECALL_THRESHOLD', 0.75))
ACCURACY_THRESHOLD = float(os.getenv('ACCURACY_THRESHOLD', 0.85))
SAFETY_THRESHOLD = float(os.getenv('SAFETY_THRESHOLD', 0.95))

# API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# Embeddings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
