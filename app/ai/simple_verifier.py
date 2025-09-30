"""
简化版AI身份验证器
在没有外部AI服务时提供基础功能
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from app.models.person import PersonProfile, SearchResult

@dataclass
class VerificationResult:
    """身份验证结果"""
    is_same_person: bool
    confidence_score: float
    evidence: List[str]
    reasoning: str
    verification_methods: List[str]

class SimpleIdentityVerifier:
    """简化版身份验证器"""
    
    def __init__(self):
        self.name_similarity_threshold = 0.8
        
    async def verify_identity(self, 
                            target_profile: PersonProfile, 
                            search_results: List[SearchResult]) -> VerificationResult:
        """简化版身份验证"""
        
        evidence = []
        verification_methods = []
        scores = []
        
        # 1. 基础姓名匹配
        name_score = self._verify_name_matching(target_profile, search_results)
        if name_score > 0:
            scores.append(name_score)
            evidence.append(f"姓名匹配得分: {name_score:.2f}")
            verification_methods.append("name_matching")
        
        # 2. 关键词频率分析
        keyword_score = self._verify_keyword_frequency(target_profile, search_results)
        if keyword_score > 0:
            scores.append(keyword_score)
            evidence.append(f"关键词频率得分: {keyword_score:.2f}")
            verification_methods.append("keyword_frequency")
        
        # 3. 数据源一致性
        source_score = self._verify_source_consistency(search_results)
        if source_score > 0:
            scores.append(source_score)
            evidence.append(f"数据源一致性得分: {source_score:.2f}")
            verification_methods.append("source_consistency")
        
        # 计算总体得分
        final_score = sum(scores) / len(scores) if scores else 0.0
        is_same_person = final_score > 0.6
        
        reasoning = f"基于{len(verification_methods)}种验证方法，平均得分{final_score:.2f}。"
        if is_same_person:
            reasoning += "各项指标显示很可能为同一人。"
        else:
            reasoning += "证据不足，需要更多信息确认。"
        
        return VerificationResult(
            is_same_person=is_same_person,
            confidence_score=final_score,
            evidence=evidence,
            reasoning=reasoning,
            verification_methods=verification_methods
        )
    
    def _verify_name_matching(self, profile: PersonProfile, results: List[SearchResult]) -> float:
        """验证姓名匹配"""
        target_name = profile.name.lower()
        matches = 0
        total = len(results)
        
        if total == 0:
            return 0.0
        
        for result in results:
            content = f"{result.title} {result.content}".lower()
            if target_name in content:
                matches += 1
        
        return matches / total
    
    def _verify_keyword_frequency(self, profile: PersonProfile, results: List[SearchResult]) -> float:
        """关键词频率验证"""
        keywords = []
        
        if profile.current_company:
            keywords.append(profile.current_company.lower())
        if profile.current_job:
            keywords.append(profile.current_job.lower())
        
        if not keywords:
            return 0.0
        
        matches = 0
        total = len(results) * len(keywords)
        
        for result in results:
            content = f"{result.title} {result.content}".lower()
            for keyword in keywords:
                if keyword in content:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def _verify_source_consistency(self, results: List[SearchResult]) -> float:
        """数据源一致性验证"""
        if len(results) < 2:
            return 0.0
        
        # 简单检查：如果多个来源都有相似信息，提高可信度
        sources = set(result.source for result in results)
        return min(1.0, len(sources) / 3)  # 3个以上不同来源得满分

# 为了兼容性，提供别名
IdentityVerifier = SimpleIdentityVerifier 