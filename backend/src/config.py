from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # ===== 数据库（专家版 + 患者版数据层共用）=====
    database_url: str = Field(..., validation_alias="DATABASE_URL")

    # ===== API Keys =====
    ncbi_api_key: str = Field(default="", validation_alias="NCBI_API_KEY")
    langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")

    # ===== LLM（走 common/llm_adapter 的 MODEL_PROFILES 多厂商适配）=====
    llm_model_id: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL_ID")
    llm_fallback_models: str = Field(default="", validation_alias="LLM_FALLBACK_MODELS")
    llm_timeout: int = Field(default=60, validation_alias="LLM_TIMEOUT")

    # ===== LangSmith 追踪（两版共用）=====
    langchain_tracing_v2: str = Field(default="true", validation_alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="clinicalagent", validation_alias="LANGCHAIN_PROJECT")

    # ===== 嵌入（专家版 + 患者版数据层共用）=====
    embedding_model: str = Field(
        default="NeuML/pubmedbert-base-embeddings",
        validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=768, validation_alias="EMBEDDING_DIMENSION")

    # ===== HuggingFace 缓存目录（推理时模型加载路径，留空用默认 ~/.cache/huggingface）=====
    hf_home: str = Field(default="", validation_alias="HF_HOME")

    # ===== 精排（专家版）=====
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias="RERANK_MODEL"
    )

    # ===== 专家版本地微调模型（Llama-3.2-3B + LoRA，reasoning 节点用）=====
    reasoning_model_base_model: str = Field(
        default="unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit",
        validation_alias="REASONING_MODEL_BASE_MODEL"
    )
    reasoning_model_adapter_path: str = Field(
        default="expert_finetuning/medical_lora_adapter",
        validation_alias="REASONING_MODEL_ADAPTER_PATH"
    )
    reasoning_use_local: bool = Field(
        default=False,
        validation_alias="REASONING_USE_LOCAL"
    )

    # ===== 专家版再入院风险模型（XGBoost，risk_modeling notebook 训练产出）=====
    risk_model_path: str = Field(
        default="risk_modeling/readmission_model.json",
        validation_alias="RISK_MODEL_PATH"
    )

    # ===== PubMed（专家版）=====
    pubmed_max_results: int = Field(default=100, validation_alias="PUBMED_MAX_RESULTS")
    pubmed_batch_size: int = Field(default=20, validation_alias="PUBMED_BATCH_SIZE")

    # ===== 切块（专家版 + 患者版数据层共用）=====
    chunk_size: int = Field(default=512, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, validation_alias="CHUNK_OVERLAP")

    # ===== 检索（患者版；专家版用模块常量 FINAL_TOP_K，两版独立）=====
    retrieval_top_k: int = Field(default=5, validation_alias="RETRIEVAL_TOP_K")

    # ===== 患者版分诊分类器（LoRA）=====
    triage_adapter_path: str = Field(
        default="patient_finetuning/artifacts/triage-lora",
        validation_alias="TRIAGE_ADAPTER_PATH"
    )
    triage_base_model: str = Field(
        default="distilbert-base-uncased",
        validation_alias="TRIAGE_BASE_MODEL"
    )

    # ===== API 服务（两版共用入口 /api/chat）=====
    cors_origins: str = Field(default="*", validation_alias="CORS_ORIGINS")
    app_api_key: str = Field(default="", validation_alias="APP_API_KEY")
    rate_limit_per_minute: int = Field(default=30, validation_alias="RATE_LIMIT_PER_MINUTE")

    model_config = SettingsConfigDict(
        # .env 在项目根目录（backend 的上两级），显式指定路径，避免依赖 cwd
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        case_sensitive=True,
        # .env 里还有厂商 key（DEEPSEEK_API_KEY 等）非 Settings 字段，需忽略
        extra="ignore",
    )

    def cors_origins_list(self) -> list[str]:
        """逗号分隔的 CORS origin 字符串 → list"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
