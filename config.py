"""
Configuration management for the Webtoon RAG System.
Loads environment variables and provides centralized config access.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Centralized configuration class."""
    
    # Supabase Configuration
    SUPABASE_URL: str = os.getenv(
        "SUPABASE_URL",
        "https://juodcqlsaardneojpwlp.supabase.co"
    ) or ""
    SUPABASE_SERVICE_KEY: str = os.getenv(
        "SUPABASE_SERVICE_KEY",
        ""  # Must be set in .env
    ) or ""
    SUPABASE_TABLE: str = os.getenv("SUPABASE_TABLE", "real_deal") or "real_deal"
    
    # Hugging Face Configuration
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN", None)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") or ""
    GEMINI_MODEL: str = "gemini-2.0-flash"
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.25  # Lowered from 0.3 to get more candidates
    
    # Smart Re-ranking Configuration (for content queries)
    ENABLE_SMART_RERANKING: bool = True  # Boost popular items when similarity is close
    POPULARITY_BOOST_ENABLED: bool = True
    
    # System Configuration
    MAX_INPUT_LENGTH: int = 500
    MIN_INPUT_LENGTH: int = 5
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configuration is present."""
        required_fields = {
            "SUPABASE_URL": cls.SUPABASE_URL,
            "SUPABASE_SERVICE_KEY": cls.SUPABASE_SERVICE_KEY,
            "GEMINI_API_KEY": cls.GEMINI_API_KEY,
        }
        
        missing = [k for k, v in required_fields.items() if not v]
        
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Please set these in your .env file."
            )
        
        return True


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    print(f"⚠️ Configuration Warning: {e}")