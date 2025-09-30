"""
系统配置管理
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv("config.env")

class Settings(BaseSettings):
    """系统配置类"""
    
    # 基础配置
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    HOST: str = Field(default="127.0.0.1", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    SECRET_KEY: str = Field(default="osint-secure-key-2024-new-session", env="SECRET_KEY")
    
    # 数据库配置
    DATABASE_URL: str = Field(default="postgresql://osint:password@localhost/osint_db", env="DATABASE_URL")
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USERNAME: str = Field(default="neo4j", env="NEO4J_USERNAME")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # AI服务配置
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    USE_LOCAL_AI: bool = Field(default=False, env="USE_LOCAL_AI")
    
    # API密钥配置
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_ENGINE_ID")
    TWITTER_BEARER_TOKEN: Optional[str] = Field(default=None, env="TWITTER_BEARER_TOKEN")
    LINKEDIN_API_KEY: Optional[str] = Field(default=None, env="LINKEDIN_API_KEY")
    GITHUB_API_TOKEN: Optional[str] = Field(default=None, env="GITHUB_API_TOKEN")
    
    # 安全配置
    MAX_REQUESTS_PER_MINUTE: int = Field(default=100, env="MAX_REQUESTS_PER_MINUTE")
    ENABLE_RATE_LIMITING: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # 合规性配置
    RESPECT_ROBOTS_TXT: bool = Field(default=True, env="RESPECT_ROBOTS_TXT")
    DEFAULT_DELAY_SECONDS: int = Field(default=1, env="DEFAULT_DELAY_SECONDS")
    MAX_CONCURRENT_REQUESTS: int = Field(default=5, env="MAX_CONCURRENT_REQUESTS")
    ENABLE_DATA_RETENTION_POLICY: bool = Field(default=True, env="ENABLE_DATA_RETENTION_POLICY")
    DATA_RETENTION_DAYS: int = Field(default=90, env="DATA_RETENTION_DAYS")
    
    # 代理配置
    HTTP_PROXY: Optional[str] = Field(default=None, env="HTTP_PROXY")
    HTTPS_PROXY: Optional[str] = Field(default=None, env="HTTPS_PROXY")
    USE_ROTATING_PROXIES: bool = Field(default=False, env="USE_ROTATING_PROXIES")
    
    class Config:
        env_file = "config.env"
        case_sensitive = True

# 创建全局配置实例
settings = Settings()

# 合规性检查
def validate_compliance():
    """验证合规性配置"""
    compliance_issues = []
    
    if not settings.RESPECT_ROBOTS_TXT:
        compliance_issues.append("⚠️  警告: robots.txt 检查已禁用")
    
    if settings.DEFAULT_DELAY_SECONDS < 1:
        compliance_issues.append("⚠️  警告: 请求延迟小于1秒，可能违反服务条款")
    
    if settings.MAX_CONCURRENT_REQUESTS > 10:
        compliance_issues.append("⚠️  警告: 并发请求数过高，可能给目标服务器造成压力")
    
    if not settings.ENABLE_DATA_RETENTION_POLICY:
        compliance_issues.append("⚠️  警告: 数据保留政策已禁用")
    
    return compliance_issues

# 获取代理配置
def get_proxy_config():
    """获取代理配置"""
    if settings.HTTP_PROXY or settings.HTTPS_PROXY:
        return {
            "http": settings.HTTP_PROXY,
            "https": settings.HTTPS_PROXY
        }
    return None 