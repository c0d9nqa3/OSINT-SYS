"""
演示数据收集器
用于演示系统功能，无需外部API
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
import random

from app.collectors.base_collector import BaseCollector
from app.models.person import SearchResult
from app.core.logger import logger

class DemoCollector(BaseCollector):
    """演示数据收集器"""
    
    def __init__(self):
        super().__init__(name="Demo Collector", base_url="demo://")
        
        # 预定义的演示数据模板
        self.demo_templates = [
            {
                "title": "{name} - 个人资料",
                "content": "{name}是一位{job}，目前就职于{company}。具有丰富的{skill}经验。",
                "source": "Demo Profile Site"
            },
            {
                "title": "{name}的LinkedIn资料",
                "content": "查看{name}的专业资料。{name}在{company}担任{job}职位，专业领域包括{skill}。",
                "source": "LinkedIn Demo"
            },
            {
                "title": "{name} - 项目经验",
                "content": "{name}参与了多个{skill}相关项目，在{company}工作期间表现突出。",
                "source": "Project Portfolio"
            },
            {
                "title": "关于{name}的新闻报道",
                "content": "本网站报道了{name}在{field}领域的最新动态。{name}目前在{company}工作。",
                "source": "News Demo"
            },
            {
                "title": "{name}的技术博客",
                "content": "{name}分享了关于{skill}的技术文章。作为{company}的{job}，{name}在这个领域有深入研究。",
                "source": "Tech Blog"
            }
        ]
        
        # 预定义的职业和技能
        self.job_titles = ["软件工程师", "产品经理", "数据分析师", "市场专员", "设计师", "项目经理"]
        self.companies = ["科技公司", "创新企业", "咨询公司", "金融机构", "教育机构", "研究院"]
        self.skills = ["Python", "数据分析", "项目管理", "市场营销", "用户体验", "机器学习"]
        self.fields = ["技术", "商业", "教育", "金融", "创新"]
        
    def can_collect_from(self, url: str) -> bool:
        """演示收集器总是可用"""
        return True
    
    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """生成演示搜索结果"""
        logger.info(f"演示收集器搜索: {query}")
        
        # 模拟网络延迟
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        results = []
        
        # 生成指定数量的演示结果
        for i in range(min(limit, len(self.demo_templates))):
            template = self.demo_templates[i]
            
            # 随机选择职业信息
            job = random.choice(self.job_titles)
            company = random.choice(self.companies)
            skill = random.choice(self.skills)
            field = random.choice(self.fields)
            
            # 填充模板
            title = template["title"].format(
                name=query, job=job, company=company, skill=skill, field=field
            )
            content = template["content"].format(
                name=query, job=job, company=company, skill=skill, field=field
            )
            
            # 生成虚拟URL
            url = f"https://demo-site-{i+1}.com/profile/{query.replace(' ', '-').lower()}"
            
            # 计算相关性分数（基于姓名匹配）
            relevance_score = random.uniform(0.7, 1.0)
            
            result = SearchResult(
                source=template["source"],
                url=url,
                title=title,
                content=content,
                snippet=content[:200] + "..." if len(content) > 200 else content,
                relevance_score=relevance_score,
                reliability_score=random.uniform(0.6, 0.9),
                extracted_data={
                    "job_title": job,
                    "company": company,
                    "skills": [skill],
                    "demo_data": True
                },
                person_mentions=[query] + [f"相关人员{j}" for j in range(random.randint(0, 3))]
            )
            
            results.append(result)
        
        # 添加一些变化以模拟真实搜索
        if len(results) > 3:
            # 随机添加一些"噪音"结果
            noise_result = SearchResult(
                source="General Web",
                url=f"https://example.com/search?q={query}",
                title=f"搜索结果: {query}",
                content=f"找到了与{query}相关的多个结果，但可能不是同一人。",
                snippet=f"搜索{query}的相关信息...",
                relevance_score=random.uniform(0.3, 0.6),
                reliability_score=random.uniform(0.4, 0.7),
                extracted_data={"demo_data": True, "noise": True},
                person_mentions=[query]
            )
            results.append(noise_result)
        
        logger.info(f"演示收集器返回 {len(results)} 个结果")
        return results
    
    async def search_social_profiles(self, name: str) -> List[SearchResult]:
        """搜索社交媒体资料（演示版）"""
        logger.info(f"演示社交媒体搜索: {name}")
        
        await asyncio.sleep(random.uniform(1.0, 2.0))
        
        social_platforms = [
            ("微博", "weibo.com"),
            ("知乎", "zhihu.com"), 
            ("GitHub", "github.com"),
            ("LinkedIn", "linkedin.com")
        ]
        
        results = []
        
        for platform_name, domain in social_platforms:
            # 50% 概率找到该平台的资料
            if random.random() > 0.5:
                continue
                
            url = f"https://{domain}/profile/{name.replace(' ', '')}"
            title = f"{name}的{platform_name}资料"
            
            if platform_name == "GitHub":
                content = f"{name}在GitHub上有多个开源项目，主要涉及{random.choice(self.skills)}开发。"
            elif platform_name == "LinkedIn":
                content = f"{name}的LinkedIn显示其在{random.choice(self.companies)}担任{random.choice(self.job_titles)}。"
            elif platform_name == "知乎":
                content = f"{name}在知乎上回答了关于{random.choice(self.fields)}的问题，展现了专业知识。"
            else:
                content = f"{name}在{platform_name}上分享了工作和生活动态。"
            
            result = SearchResult(
                source=f"{platform_name} Demo",
                url=url,
                title=title,
                content=content,
                snippet=content,
                relevance_score=random.uniform(0.8, 1.0),
                reliability_score=random.uniform(0.7, 0.9),
                extracted_data={
                    "platform": platform_name,
                    "social_media": True,
                    "demo_data": True
                },
                person_mentions=[name]
            )
            
            results.append(result)
        
        return results
    
    async def search_professional_info(self, name: str) -> List[SearchResult]:
        """搜索专业信息（演示版）"""
        logger.info(f"演示专业信息搜索: {name}")
        
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        professional_sources = [
            "公司官网",
            "行业报告", 
            "专业论坛",
            "会议演讲",
            "专利数据库"
        ]
        
        results = []
        
        for source in professional_sources[:random.randint(2, 4)]:
            job = random.choice(self.job_titles)
            company = random.choice(self.companies)
            skill = random.choice(self.skills)
            
            if source == "公司官网":
                title = f"{company}团队介绍 - {name}"
                content = f"{name}作为我们的{job}，在{skill}方面具有丰富经验。"
            elif source == "专利数据库":
                title = f"发明专利 - {name}等"
                content = f"{name}与团队共同申请了关于{skill}技术的发明专利。"
            elif source == "会议演讲":
                field = random.choice(self.fields)
                title = f"{name}在{field}会议上的演讲"
                content = f"{name}分享了在{company}工作中关于{skill}的实践经验。"
            else:
                field = random.choice(self.fields)
                title = f"{source}提及{name}"
                content = f"在{source}中发现{name}在{field}领域的相关信息。"
            
            result = SearchResult(
                source=f"{source} Demo",
                url=f"https://demo-{source.lower().replace(' ', '')}.com/{name}",
                title=title,
                content=content,
                snippet=content,
                relevance_score=random.uniform(0.6, 0.9),
                reliability_score=random.uniform(0.5, 0.8),
                extracted_data={
                    "professional": True,
                    "source_type": source,
                    "demo_data": True
                },
                person_mentions=[name]
            )
            
            results.append(result)
        
        return results
    
    def get_reliability_score(self) -> float:
        """演示收集器的可靠性评分"""
        return 0.5  # 演示数据，中等可靠性 