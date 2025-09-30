"""
Google搜索数据收集器
使用Google Custom Search API收集公开信息
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime

from app.collectors.base_collector import SearchEngineCollector
from app.models.person import SearchResult
from app.core.config import settings
from app.core.logger import logger

class GoogleCollector(SearchEngineCollector):
    """Google自定义搜索收集器"""
    
    def __init__(self):
        super().__init__(
            name="Google Search",
            api_key=settings.GOOGLE_API_KEY
        )
        self.search_engine_id = settings.GOOGLE_SEARCH_ENGINE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    def can_collect_from(self, url: str) -> bool:
        """检查是否可以收集Google搜索数据"""
        return bool(self.api_key and self.search_engine_id)
    
    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """执行Google搜索"""
        if not self.can_collect_from(""):
            logger.warning("Google API配置不完整，跳过Google搜索")
            return []
        
        results = []
        
        try:
            # 构建搜索参数
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(limit, 10),  # Google API每次最多返回10个结果
                'safe': 'medium',
                'lr': 'lang_zh|lang_en'  # 中文和英文结果
            }
            
            response = self.make_request(self.base_url, params=params)
            
            if response and response.status_code == 200:
                data = response.json()
                results = self.parse_search_results(data, query)
                logger.info(f"Google搜索 '{query}' 返回 {len(results)} 个结果")
            else:
                logger.error(f"Google搜索失败: {response.status_code if response else 'No response'}")
                
        except Exception as e:
            logger.error(f"Google搜索异常: {e}")
        
        return results
    
    def parse_search_results(self, data: Dict[str, Any], query: str) -> List[SearchResult]:
        """解析Google搜索结果"""
        results = []
        
        items = data.get('items', [])
        
        for item in items:
            try:
                # 提取基本信息
                title = item.get('title', '')
                link = item.get('link', '')
                snippet = item.get('snippet', '')
                
                # 计算相关性评分
                relevance_score = self.calculate_relevance(query, title, snippet)
                
                # 提取结构化数据
                extracted_data = self.extract_structured_data(item)
                
                result = SearchResult(
                    source="Google Search",
                    url=link,
                    title=title,
                    content=snippet,
                    snippet=snippet,
                    relevance_score=relevance_score,
                    extracted_data=extracted_data,
                    person_mentions=self.extract_person_mentions(title + " " + snippet),
                    reliability_score=0.9  # Google搜索结果一般可靠性较高
                )
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"解析Google搜索结果失败: {e}")
                continue
        
        return results
    
    def calculate_relevance(self, query: str, title: str, snippet: str) -> float:
        """计算搜索结果相关性"""
        relevance = 0.0
        query_terms = query.lower().split()
        
        content = (title + " " + snippet).lower()
        
        # 计算查询词匹配度
        matched_terms = sum(1 for term in query_terms if term in content)
        term_ratio = matched_terms / len(query_terms) if query_terms else 0
        
        relevance += term_ratio * 0.6
        
        # 标题匹配加分
        title_matches = sum(1 for term in query_terms if term in title.lower())
        title_ratio = title_matches / len(query_terms) if query_terms else 0
        relevance += title_ratio * 0.4
        
        return min(relevance, 1.0)
    
    def extract_structured_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """从搜索结果中提取结构化数据"""
        structured_data = {}
        
        # 提取页面信息
        if 'pagemap' in item:
            pagemap = item['pagemap']
            
            # 提取网站信息
            if 'website' in pagemap:
                structured_data['website'] = pagemap['website'][0]
            
            # 提取组织信息
            if 'organization' in pagemap:
                structured_data['organization'] = pagemap['organization'][0]
            
            # 提取人物信息
            if 'person' in pagemap:
                structured_data['person'] = pagemap['person'][0]
            
            # 提取社交媒体信息
            if 'socialmediaposting' in pagemap:
                structured_data['social_media'] = pagemap['socialmediaposting'][0]
        
        return structured_data
    
    def extract_person_mentions(self, text: str) -> List[str]:
        """从文本中提取人名提及"""
        # 这里可以使用更复杂的NLP技术
        # 暂时使用简单的启发式方法
        person_mentions = []
        
        # 查找常见的人名模式
        import re
        
        # 中文姓名模式
        chinese_name_pattern = r'[\u4e00-\u9fff]{2,4}'
        chinese_names = re.findall(chinese_name_pattern, text)
        
        # 英文姓名模式
        english_name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        english_names = re.findall(english_name_pattern, text)
        
        person_mentions.extend(chinese_names)
        person_mentions.extend(english_names)
        
        # 去重并过滤
        person_mentions = list(set(person_mentions))
        person_mentions = [name for name in person_mentions if len(name) > 1]
        
        return person_mentions[:10]  # 限制返回数量
    
    async def search_with_filters(self, name: str, filters: Dict[str, str]) -> List[SearchResult]:
        """带过滤条件的搜索"""
        query_parts = [f'"{name}"']
        
        # 添加过滤条件
        if 'site' in filters:
            query_parts.append(f"site:{filters['site']}")
        
        if 'filetype' in filters:
            query_parts.append(f"filetype:{filters['filetype']}")
        
        if 'company' in filters:
            query_parts.append(f'"{filters["company"]}"')
        
        if 'location' in filters:
            query_parts.append(f'"{filters["location"]}"')
        
        query = " ".join(query_parts)
        return await self.search(query)
    
    async def search_social_profiles(self, name: str) -> List[SearchResult]:
        """搜索社交媒体资料"""
        social_sites = [
            "linkedin.com",
            "twitter.com", 
            "facebook.com",
            "instagram.com",
            "github.com",
            "weibo.com",
            "zhihu.com"
        ]
        
        all_results = []
        
        for site in social_sites:
            try:
                results = await self.search_with_filters(name, {"site": site})
                all_results.extend(results)
                
                # 添加延迟避免过快请求
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"搜索 {site} 上的 {name} 失败: {e}")
        
        return all_results
    
    async def search_professional_info(self, name: str) -> List[SearchResult]:
        """搜索专业信息"""
        professional_terms = [
            "CEO", "CTO", "管理", "总监", "经理", "工程师", "教授", "博士"
        ]
        
        all_results = []
        
        for term in professional_terms:
            try:
                query = f'"{name}" {term}'
                results = await self.search(query, limit=5)
                all_results.extend(results)
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"搜索 {name} 的专业信息失败: {e}")
        
        return all_results
    
    def get_reliability_score(self) -> float:
        """Google搜索的可靠性评分"""
        return 0.9 