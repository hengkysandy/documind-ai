"""Configuration management for Confluence-QA Bot."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # Confluence API settings
    CONFLUENCE_EMAIL: str = os.getenv("CONFLUENCE_USERNAME", "")
    CONFLUENCE_API_KEY: str = os.getenv("CONFLUENCE_API_TOKEN", "")
    CONFLUENCE_BASE: str = os.getenv("CONFLUENCE_BASE_URL", "")
    PAGE_IDS: list[str] = os.getenv("PAGE_IDS", "").split(",")

    # DeepSeek API settings
    DEEPSEEK_API_KEY: str | None = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # MongoDB settings
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "confluence_qa")
    MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "vectors")

    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "300"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Server settings
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8080"))

    # Slack settings (optional)
    SLACK_APP_TOKEN: str | None = os.getenv("SLACK_APP_TOKEN")
    SLACK_BOT_TOKEN: str | None = os.getenv("SLACK_BOT_TOKEN")
    QA_ENDPOINT: str = os.getenv("QA_ENDPOINT", "http://127.0.0.1:8080/ask")

    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        required_fields = [
            "CONFLUENCE_EMAIL",
            "CONFLUENCE_API_KEY", 
            "CONFLUENCE_BASE",
            "MONGODB_URI",
        ]

        missing_fields = []
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required configuration: {', '.join(missing_fields)}")

        if not cls.PAGE_IDS or cls.PAGE_IDS == [""]:
            raise ValueError("PAGE_IDS must contain at least one page ID")

    @classmethod
    def get_confluence_url(cls, page_id: str) -> str:
        """Generate Confluence URL for a given page ID."""
        return f"{cls.CONFLUENCE_BASE}/pages/viewpage.action?pageId={page_id}"


# Create global config instance
config = Config()
