"""Application configuration from environment (Vertex AI, DB, Langfuse)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vertex AI + Gemini (Google Gen AI SDK)
    google_cloud_project: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT"),
    )
    google_cloud_location: str = Field(
        default="us-central1",
        validation_alias=AliasChoices("GOOGLE_CLOUD_LOCATION", "GCP_LOCATION"),
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        validation_alias=AliasChoices("GEMINI_MODEL", "GOOGLE_GEMINI_MODEL"),
    )
    # Phase 3 — NL→SQL: Vertex text embeddings for few-shot + schema routing
    text_embedding_model: str = Field(
        default="text-embedding-005",
        validation_alias=AliasChoices("TEXT_EMBEDDING_MODEL", "VERTEX_EMBEDDING_MODEL"),
    )
    nl_sql_max_detailed_schema_tables: int = Field(
        default=12,
        ge=1,
        validation_alias="NL_SQL_MAX_DETAILED_SCHEMA_TABLES",
    )
    nl_sql_few_shot_count: int = Field(
        default=4,
        ge=0,
        le=20,
        validation_alias="NL_SQL_FEW_SHOT_COUNT",
    )
    nl_sql_full_schema_table_threshold: int = Field(
        default=10,
        ge=1,
        validation_alias="NL_SQL_FULL_SCHEMA_TABLE_THRESHOLD",
        description=(
            "If the dataset has at most this many tables, include full per-table column lists "
            "for every CSV; otherwise use semantic routing + stubs."
        ),
    )
    google_genai_use_vertexai: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GENAI_USE_VERTEXAI",
        ),
    )

    # PostgreSQL
    database_url: str = Field(
        default="postgresql+psycopg://app:app@localhost:5432/app",
        validation_alias=AliasChoices("DATABASE_URL", "SQLALCHEMY_DATABASE_URI"),
    )

    # Langfuse (optional)
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_HOST",
    )
    langfuse_release: str = Field(default="", validation_alias="LANGFUSE_RELEASE")
    langfuse_tracing_environment: str = Field(
        default="",
        validation_alias="LANGFUSE_TRACING_ENVIRONMENT",
    )
    langfuse_trace_version: str = Field(
        default="",
        validation_alias="LANGFUSE_TRACE_VERSION",
    )

    def vertex_configured(self) -> bool:
        return bool(self.google_cloud_project.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
