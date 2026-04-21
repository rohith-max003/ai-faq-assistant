from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment_name: str = "gpt-4"
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
    azure_openai_api_version: str = "2024-02-01"

    app_env: str = "development"
    log_level: str = "INFO"
    confidence_threshold: float = 0.75
    max_retrieved_docs: int = 5
    conversation_memory_k: int = 6

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
