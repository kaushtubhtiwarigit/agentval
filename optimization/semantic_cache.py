"""
Redis-based semantic caching with embedding similarity
"""
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict
import json
import config

class SemanticCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
            decode_responses=True
        )
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.similarity_threshold = config.CACHE_SIMILARITY_THRESHOLD
        self.ttl = config.CACHE_TTL_SECONDS
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.embedding_model.encode(text)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _get_cache_keys(self, pattern: str = "cache:*") -> list:
        """Get all cache keys"""
        return self.redis_client.keys(pattern)
    
    def get(self, prompt: str) -> Optional[Dict]:
        """
        Retrieve from cache if semantically similar prompt exists
        Returns: {response, similarity} or None
        """
        query_embedding = self._embed(prompt)
        
        # Search through cache for similar prompts
        cache_keys = self._get_cache_keys()
        
        best_match = None
        best_similarity = 0.0
        
        for key in cache_keys:
            try:
                cached_data = self.redis_client.get(key)
                if not cached_data:
                    continue
                
                cached = json.loads(cached_data)
                cached_embedding = np.array(cached['embedding'])
                
                similarity = self._cosine_similarity(query_embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = {
                        'response': cached['response'],
                        'similarity': similarity,
                        'original_prompt': cached['prompt']
                    }
            
            except Exception as e:
                print(f"[Cache] Error reading key {key}: {e}")
                continue
        
        return best_match
    
    def set(self, prompt: str, response: str):
        """Store prompt-response pair with embedding"""
        embedding = self._embed(prompt)
        
        cache_data = {
            'prompt': prompt,
            'response': response,
            'embedding': embedding.tolist()
        }
        
        # Use prompt hash as key
        key = f"cache:{hash(prompt)}"
        
        self.redis_client.setex(
            key,
            self.ttl,
            json.dumps(cache_data)
        )
    
    def clear(self):
        """Clear all cache entries"""
        keys = self._get_cache_keys()
        if keys:
            self.redis_client.delete(*keys)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        cache_keys = self._get_cache_keys()
        return {
            'total_entries': len(cache_keys),
            'similarity_threshold': self.similarity_threshold,
            'ttl_seconds': self.ttl
        }
