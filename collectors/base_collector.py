"""
基础数据收集器
所有具体收集器的父类，提供通用功能和合规性检查
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup

from app.core.config import settings, get_proxy_config
from app.core.logger import logger, log_collection, log_compliance_warning
from app.models.person import SearchResult, DataSource

class BaseCollector(ABC):
    """基础数据收集器抽象类"""
    
    def __init__(self, name: str, base_url: str = ""):
        self.name = name
        self.base_url = base_url
        self.session = requests.Session()
        self.last_request_time = 0
        self.request_count = 0
        self.proxies = get_proxy_config()
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        if self.proxies:
            self.session.proxies.update(self.proxies)
            
        logger.info(f"初始化数据收集器: {self.name}")
    
    def check_robots_txt(self, url: str) -> bool:
        """检查robots.txt是否允许访问"""
        if not settings.RESPECT_ROBOTS_TXT:
            return True
            
        try:
            rp = RobotFileParser()
            robots_url = f"{self.base_url}/robots.txt"
            rp.set_url(robots_url)
            rp.read()
            
            allowed = rp.can_fetch(self.session.headers.get('User-Agent', '*'), url)
            if not allowed:
                log_compliance_warning(f"robots.txt 禁止访问: {url}")
            return allowed
            
        except Exception as e:
            logger.warning(f"无法检查 robots.txt: {e}")
            return True
    
    def rate_limit(self):
        """实施速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < settings.DEFAULT_DELAY_SECONDS:
            sleep_time = settings.DEFAULT_DELAY_SECONDS - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # 检查每分钟请求限制
        if self.request_count > settings.MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"达到速率限制: {self.request_count} 请求/分钟")
            time.sleep(60)  # 等待一分钟
            self.request_count = 0
    
    def make_request(self, url: str, method: str = "GET", **kwargs) -> Optional[requests.Response]:
        """发送HTTP请求，包含所有合规性检查"""
        
        # 检查robots.txt
        if not self.check_robots_txt(url):
            return None
        
        # 应用速率限制
        if settings.ENABLE_RATE_LIMITING:
            self.rate_limit()
        
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            
            log_collection(
                source=self.name,
                target=url,
                data_type="web_request",
                status="success" if response.status_code == 200 else "error",
                details=f"状态码: {response.status_code}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"请求失败 {url}: {e}")
            log_collection(
                source=self.name,
                target=url,
                data_type="web_request",
                status="error",
                details=str(e)
            )
            return None
    
    def extract_text_from_html(self, html_content: str) -> str:
        """从HTML中提取纯文本"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()
            
            # 获取文本
            text = soup.get_text()
            
            # 清理文本
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"HTML文本提取失败: {e}")
            return ""
    
    def create_data_source(self, url: str) -> DataSource:
        """创建数据源记录"""
        return DataSource(
            name=self.name,
            type=self.__class__.__name__,
            url=url,
            reliability_score=self.get_reliability_score(),
        )
    
    def get_reliability_score(self) -> float:
        """获取数据源可靠性评分"""
        # 子类可以重写此方法提供特定的可靠性评分
        return 0.7
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """搜索方法 - 由子类实现"""
        pass
    
    @abstractmethod
    def can_collect_from(self, url: str) -> bool:
        """检查是否可以从指定URL收集数据"""
        pass
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'session'):
            self.session.close()

class SearchEngineCollector(BaseCollector):
    """搜索引擎收集器基类"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        super().__init__(name)
        self.api_key = api_key
    
    def build_query(self, name: str, additional_terms: List[str] = None) -> str:
        """构建搜索查询"""
        query_parts = [f'"{name}"']
        
        if additional_terms:
            query_parts.extend(additional_terms)
        
        return " ".join(query_parts)
    
    def parse_search_results(self, response_data: Dict[str, Any]) -> List[SearchResult]:
        """解析搜索结果 - 由具体子类实现"""
        return []

class SocialMediaCollector(BaseCollector):
    """社交媒体收集器基类"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        super().__init__(name)
        self.api_key = api_key
    
    def extract_profile_info(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """从社交媒体资料中提取信息"""
        return {
            "username": profile_data.get("username", ""),
            "display_name": profile_data.get("display_name", ""),
            "bio": profile_data.get("bio", ""),
            "location": profile_data.get("location", ""),
            "followers_count": profile_data.get("followers_count", 0),
            "following_count": profile_data.get("following_count", 0),
        }
    
    def calculate_profile_similarity(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> float:
        """计算两个社交媒体资料的相似度"""
        # 这里可以实现更复杂的相似度算法
        similarity_score = 0.0
        
        # 比较用户名
        if profile1.get("username") == profile2.get("username"):
            similarity_score += 0.4
        
        # 比较显示名称
        if profile1.get("display_name") == profile2.get("display_name"):
            similarity_score += 0.3
        
        # 比较地理位置
        if profile1.get("location") == profile2.get("location"):
            similarity_score += 0.2
        
        # 比较个人简介的相似度
        bio1 = profile1.get("bio", "").lower()
        bio2 = profile2.get("bio", "").lower()
        if bio1 and bio2:
            # 简单的文本相似度检查
            common_words = set(bio1.split()) & set(bio2.split())
            if common_words:
                similarity_score += 0.1
        
        return min(similarity_score, 1.0) 