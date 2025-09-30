"""
人物信息数据模型
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class PersonProfile(BaseModel):
    """人物档案数据模型 - Pydantic"""
    
    # 基本信息
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="姓名")
    alternative_names: List[str] = Field(default=[], description="别名、昵称")
    age: Optional[int] = Field(None, ge=0, le=150, description="年龄")
    birth_date: Optional[datetime] = Field(None, description="出生日期")
    gender: Optional[str] = Field(None, description="性别")
    photo_url: Optional[str] = Field(default=None, description="人物照片URL")
    
    # 联系方式
    email: Optional[str] = Field(None, description="电子邮箱")
    phone: Optional[str] = Field(None, description="电话号码(主)")
    phones: List[str] = Field(default=[], description="电话号码列表")
    address: Optional[str] = Field(None, description="地址(主)")
    delivery_addresses: List[str] = Field(default=[], description="快递/收货地址列表")
    
    # 身份/户籍
    id_numbers: List[str] = Field(default=[], description="身份证号码列表")
    hukou_place: Optional[str] = Field(default=None, description="户籍地")
    hukou_address: Optional[str] = Field(default=None, description="户籍地地址")
    
    # 职业信息
    current_job: Optional[str] = Field(None, description="当前职位")
    current_company: Optional[str] = Field(None, description="当前公司")
    industry: Optional[str] = Field(None, description="行业")
    skills: List[str] = Field(default=[], description="技能")
    
    # 教育背景
    education: List[Dict[str, Any]] = Field(default=[], description="教育经历")
    
    # 社交媒体（固定字段）
    wechat_id: Optional[str] = Field(default=None, description="微信")
    qq_id: Optional[str] = Field(default=None, description="QQ")
    weibo_id: Optional[str] = Field(default=None, description="微博")
    douyin_id: Optional[str] = Field(default=None, description="抖音")
    xhs_id: Optional[str] = Field(default=None, description="小红书")
    gitee_username: Optional[str] = Field(default=None, description="Gitee 用户名")
    linkedin_url: Optional[str] = Field(default=None, description="LinkedIn 链接")
    
    # 兼容的社交媒体字典
    social_profiles: Dict[str, str] = Field(default={}, description="社交媒体资料")
    
    # 自定义属性与原始文本
    custom_attributes: Dict[str, Any] = Field(default={}, description="未预定义的自定义维度")
    raw_text: Optional[str] = Field(default=None, description="导入的原始文本")
    
    # 关系网络
    relationships: List[Dict[str, Any]] = Field(default=[], description="社会关系")
    
    # 数据来源
    data_sources: List[Dict[str, Any]] = Field(default=[], description="数据来源")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="数据可信度")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    verified: bool = Field(default=False, description="是否已验证")
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('无效的邮箱格式')
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PersonEntity(Base):
    """人物实体 - SQLAlchemy"""
    __tablename__ = "persons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    alternative_names = Column(JSON, default=[])
    age = Column(Integer)
    birth_date = Column(DateTime)
    gender = Column(String(20))
    photo_url = Column(String(512))
    
    # 联系方式
    email = Column(String(255), index=True)
    phone = Column(String(50))
    phones = Column(JSON, default=[])
    address = Column(Text)
    delivery_addresses = Column(JSON, default=[])
    
    # 身份/户籍
    id_numbers = Column(JSON, default=[])
    hukou_place = Column(String(255))
    hukou_address = Column(Text)
    
    # 职业信息
    current_job = Column(String(255))
    current_company = Column(String(255))
    industry = Column(String(100))
    skills = Column(JSON, default=[])
    
    # 教育背景
    education = Column(JSON, default=[])
    
    # 社交媒体（固定字段）
    wechat_id = Column(String(255))
    qq_id = Column(String(255))
    weibo_id = Column(String(255))
    douyin_id = Column(String(255))
    xhs_id = Column(String(255))
    gitee_username = Column(String(255))
    linkedin_url = Column(String(512))
    
    # 兼容的社交媒体字典
    social_profiles = Column(JSON, default={})
    
    # 自定义属性与原始文本
    custom_attributes = Column(JSON, default={})
    raw_text = Column(Text)
    
    # 关系网络
    relationships = Column(JSON, default=[])
    
    # 数据来源
    data_sources = Column(JSON, default=[])
    confidence_score = Column(Float, default=0.0)
    
    # 元数据
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    verified = Column(Boolean, default=False)
    
    # 合规性字段
    data_retention_date = Column(DateTime)
    collection_consent = Column(Boolean, default=False)

class SearchResult(BaseModel):
    """搜索结果数据模型"""
    
    # 基本信息
    source: str = Field(..., description="数据源")
    url: str = Field(..., description="源链接")
    title: str = Field(..., description="标题")
    content: str = Field(..., description="内容")
    snippet: str = Field(..., description="摘要")
    
    # 元数据
    timestamp: datetime = Field(default_factory=datetime.now)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # 提取的信息
    extracted_data: Dict[str, Any] = Field(default={})
    person_mentions: List[str] = Field(default=[])
    
    # 置信度评估
    reliability_score: float = Field(default=0.0, ge=0.0, le=1.0)

class Relationship(BaseModel):
    """关系数据模型"""
    
    # 关系双方
    person_id_1: str = Field(..., description="人物1 ID")
    person_id_2: str = Field(..., description="人物2 ID")
    
    # 关系类型
    relationship_type: str = Field(..., description="关系类型")
    relationship_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # 关系描述
    description: str = Field(default="", description="关系描述")
    context: str = Field(default="", description="关系背景")
    
    # 数据来源
    source: str = Field(..., description="关系来源")
    evidence: List[str] = Field(default=[], description="证据链接")
    
    # 时间信息
    start_date: Optional[datetime] = Field(None, description="关系开始时间")
    end_date: Optional[datetime] = Field(None, description="关系结束时间")
    is_current: bool = Field(default=True, description="是否当前关系")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

class DataSource(BaseModel):
    """数据源模型"""
    
    # 基本信息
    name: str = Field(..., description="数据源名称")
    type: str = Field(..., description="数据源类型")
    url: Optional[str] = Field(None, description="数据源URL")
    
    # 质量评估
    reliability_score: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # 访问信息
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_frequency: int = Field(default=0, description="访问次数")
    
    # 合规性
    robots_txt_allowed: bool = Field(default=True)
    rate_limit: Optional[int] = Field(None, description="速率限制(请求/分钟)")
    
    # 元数据
    created_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = Field(default=True) 