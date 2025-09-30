"""
AI身份验证系统
使用多种AI技术验证和筛选人物身份信息
"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import re

# 导入AI相关库
try:
    import openai
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False

from app.core.config import settings
from app.core.logger import logger, log_audit
from app.models.person import PersonProfile, SearchResult

@dataclass
class VerificationResult:
    """身份验证结果"""
    is_same_person: bool
    confidence_score: float
    evidence: List[str]
    reasoning: str
    verification_methods: List[str]

class IdentityVerifier:
    """AI身份验证器"""
    
    def __init__(self):
        self.openai_client = None
        self.text_similarity_model = None
        self.face_recognition_available = False
        
        # 初始化AI模型
        self._initialize_models()
        
    def _initialize_models(self):
        """初始化AI模型"""
        try:
            # 初始化OpenAI
            if settings.OPENAI_API_KEY and not settings.USE_LOCAL_AI and HAS_AI_LIBS:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI客户端初始化成功")
            
            # 初始化本地模型
            if HAS_AI_LIBS:
                try:
                    self.text_similarity_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                    logger.info("文本相似度模型加载成功")
                except Exception as e:
                    logger.warning(f"文本相似度模型加载失败: {e}")
                
                # 检查人脸识别库
                try:
                    import face_recognition
                    self.face_recognition_available = True
                    logger.info("人脸识别功能可用")
                except ImportError:
                    logger.warning("人脸识别库未安装")
            
        except Exception as e:
            logger.error(f"AI模型初始化失败: {e}")
    
    async def verify_identity(self, 
                            target_profile: PersonProfile, 
                            search_results: List[SearchResult]) -> VerificationResult:
        """综合身份验证"""
        
        log_audit("IDENTITY_VERIFICATION", target_profile.name, details="开始身份验证")
        
        verification_scores = []
        all_evidence = []
        verification_methods = []
        
        # 1. 文本相似度验证
        text_score, text_evidence = await self._verify_by_text_similarity(target_profile, search_results)
        if text_score > 0:
            verification_scores.append(text_score)
            all_evidence.extend(text_evidence)
            verification_methods.append("text_similarity")
        
        # 2. 名称匹配验证
        name_score, name_evidence = self._verify_by_name_matching(target_profile, search_results)
        if name_score > 0:
            verification_scores.append(name_score)
            all_evidence.extend(name_evidence)
            verification_methods.append("name_matching")
        
        # 3. 社交网络验证
        social_score, social_evidence = await self._verify_by_social_networks(target_profile, search_results)
        if social_score > 0:
            verification_scores.append(social_score)
            all_evidence.extend(social_evidence)
            verification_methods.append("social_networks")
        
        # 4. 时间线一致性验证
        timeline_score, timeline_evidence = self._verify_by_timeline_consistency(target_profile, search_results)
        if timeline_score > 0:
            verification_scores.append(timeline_score)
            all_evidence.extend(timeline_evidence)
            verification_methods.append("timeline_consistency")
        
        # 5. 地理位置验证
        location_score, location_evidence = self._verify_by_location_consistency(target_profile, search_results)
        if location_score > 0:
            verification_scores.append(location_score)
            all_evidence.extend(location_evidence)
            verification_methods.append("location_consistency")
        
        # 6. AI语言模型验证 (如果可用)
        if self.openai_client:
            ai_score, ai_evidence = await self._verify_by_ai_reasoning(target_profile, search_results)
            if ai_score > 0:
                verification_scores.append(ai_score)
                all_evidence.extend(ai_evidence)
                verification_methods.append("ai_reasoning")
        
        # 计算综合评分
        if verification_scores:
            final_score = np.mean(verification_scores)
            is_same_person = final_score > 0.7  # 阈值可调整
        else:
            final_score = 0.0
            is_same_person = False
        
        reasoning = self._generate_reasoning(verification_methods, verification_scores, final_score)
        
        result = VerificationResult(
            is_same_person=is_same_person,
            confidence_score=final_score,
            evidence=all_evidence,
            reasoning=reasoning,
            verification_methods=verification_methods
        )
        
        log_audit("IDENTITY_VERIFICATION", target_profile.name, 
                 details=f"验证完成，置信度: {final_score:.2f}, 结果: {is_same_person}")
        
        return result
    
    async def _verify_by_text_similarity(self, 
                                       target_profile: PersonProfile, 
                                       search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """基于文本相似度的验证"""
        if not self.text_similarity_model:
            return 0.0, []
        
        evidence = []
        similarities = []
        
        # 构建目标人物的文本描述
        target_text = self._build_person_description(target_profile)
        
        for result in search_results:
            try:
                # 构建搜索结果的文本描述
                result_text = f"{result.title} {result.content}"
                
                # 计算文本相似度
                if target_text and result_text:
                    similarity = self._calculate_text_similarity(target_text, result_text)
                    
                    if similarity > 0.6:  # 高相似度阈值
                        similarities.append(similarity)
                        evidence.append(f"文本相似度 {similarity:.2f}: {result.url}")
                        
            except Exception as e:
                logger.warning(f"文本相似度计算失败: {e}")
                continue
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return avg_similarity, evidence
    
    def _verify_by_name_matching(self, 
                               target_profile: PersonProfile, 
                               search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """基于姓名匹配的验证"""
        evidence = []
        matches = []
        
        target_names = [target_profile.name.lower()] + [name.lower() for name in target_profile.alternative_names]
        
        for result in search_results:
            try:
                content = f"{result.title} {result.content}".lower()
                
                for name in target_names:
                    if name in content:
                        # 计算名称匹配强度
                        match_strength = self._calculate_name_match_strength(name, content)
                        matches.append(match_strength)
                        evidence.append(f"姓名匹配 '{name}' 在 {result.url}")
                        
            except Exception as e:
                logger.warning(f"姓名匹配验证失败: {e}")
                continue
        
        avg_match = np.mean(matches) if matches else 0.0
        return avg_match, evidence
    
    async def _verify_by_social_networks(self, 
                                       target_profile: PersonProfile, 
                                       search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """基于社交网络的验证"""
        evidence = []
        social_matches = []
        
        target_socials = target_profile.social_profiles
        
        for result in search_results:
            try:
                # 检查是否为社交媒体链接
                social_platform = self._identify_social_platform(result.url)
                
                if social_platform:
                    # 提取社交媒体用户名
                    username = self._extract_social_username(result.url, social_platform)
                    
                    if username and social_platform in target_socials:
                        target_username = target_socials[social_platform]
                        
                        # 计算用户名相似度
                        similarity = self._calculate_username_similarity(username, target_username)
                        
                        if similarity > 0.8:
                            social_matches.append(similarity)
                            evidence.append(f"社交媒体匹配 {social_platform}: {username}")
                            
            except Exception as e:
                logger.warning(f"社交网络验证失败: {e}")
                continue
        
        avg_social = np.mean(social_matches) if social_matches else 0.0
        return avg_social, evidence
    
    def _verify_by_timeline_consistency(self, 
                                      target_profile: PersonProfile, 
                                      search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """基于时间线一致性的验证"""
        evidence = []
        timeline_scores = []
        
        # 提取目标人物的时间信息
        target_events = self._extract_timeline_events(target_profile)
        
        for result in search_results:
            try:
                # 从搜索结果中提取时间信息
                result_events = self._extract_timeline_from_text(f"{result.title} {result.content}")
                
                # 计算时间线一致性
                consistency = self._calculate_timeline_consistency(target_events, result_events)
                
                if consistency > 0.5:
                    timeline_scores.append(consistency)
                    evidence.append(f"时间线一致性 {consistency:.2f}: {result.url}")
                    
            except Exception as e:
                logger.warning(f"时间线验证失败: {e}")
                continue
        
        avg_timeline = np.mean(timeline_scores) if timeline_scores else 0.0
        return avg_timeline, evidence
    
    def _verify_by_location_consistency(self, 
                                      target_profile: PersonProfile, 
                                      search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """基于地理位置一致性的验证"""
        evidence = []
        location_scores = []
        
        target_location = target_profile.address
        if not target_location:
            return 0.0, []
        
        for result in search_results:
            try:
                # 从搜索结果中提取位置信息
                result_locations = self._extract_locations_from_text(f"{result.title} {result.content}")
                
                for location in result_locations:
                    similarity = self._calculate_location_similarity(target_location, location)
                    
                    if similarity > 0.6:
                        location_scores.append(similarity)
                        evidence.append(f"位置匹配 '{location}' 与 '{target_location}': {result.url}")
                        
            except Exception as e:
                logger.warning(f"位置验证失败: {e}")
                continue
        
        avg_location = np.mean(location_scores) if location_scores else 0.0
        return avg_location, evidence
    
    async def _verify_by_ai_reasoning(self, 
                                    target_profile: PersonProfile, 
                                    search_results: List[SearchResult]) -> Tuple[float, List[str]]:
        """使用AI大模型进行推理验证"""
        if not self.openai_client:
            return 0.0, []
        
        evidence = []
        
        try:
            # 构建AI推理提示
            prompt = self._build_ai_verification_prompt(target_profile, search_results)
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的身份验证专家，需要分析给定的信息，判断是否为同一人。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            
            # 解析AI响应
            confidence, reasoning = self._parse_ai_response(ai_response)
            
            if confidence > 0:
                evidence.append(f"AI推理: {reasoning}")
            
            return confidence, evidence
            
        except Exception as e:
            logger.error(f"AI推理验证失败: {e}")
            return 0.0, []
    
    def _build_person_description(self, profile: PersonProfile) -> str:
        """构建人物文本描述"""
        parts = []
        
        if profile.name:
            parts.append(f"姓名: {profile.name}")
        
        if profile.current_job:
            parts.append(f"职位: {profile.current_job}")
        
        if profile.current_company:
            parts.append(f"公司: {profile.current_company}")
        
        if profile.skills:
            parts.append(f"技能: {', '.join(profile.skills)}")
        
        if profile.address:
            parts.append(f"地址: {profile.address}")
        
        return " ".join(parts)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度"""
        try:
            if not self.text_similarity_model:
                return 0.0
            
            embeddings = self.text_similarity_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return max(0.0, float(similarity))
            
        except Exception as e:
            logger.error(f"文本相似度计算错误: {e}")
            return 0.0
    
    def _calculate_name_match_strength(self, name: str, content: str) -> float:
        """计算姓名匹配强度"""
        # 精确匹配
        if name in content:
            # 检查是否为完整词匹配
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, content, re.IGNORECASE):
                return 1.0
            else:
                return 0.8
        
        # 部分匹配
        name_parts = name.split()
        matches = sum(1 for part in name_parts if part in content)
        return matches / len(name_parts) if name_parts else 0.0
    
    def _identify_social_platform(self, url: str) -> Optional[str]:
        """识别社交媒体平台"""
        platforms = {
            'linkedin.com': 'linkedin',
            'twitter.com': 'twitter',
            'facebook.com': 'facebook', 
            'instagram.com': 'instagram',
            'github.com': 'github',
            'weibo.com': 'weibo',
            'zhihu.com': 'zhihu'
        }
        
        for domain, platform in platforms.items():
            if domain in url.lower():
                return platform
        
        return None
    
    def _extract_social_username(self, url: str, platform: str) -> Optional[str]:
        """从社交媒体URL中提取用户名"""
        try:
            import re
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            
            # 不同平台的用户名提取规则
            if platform == 'linkedin':
                match = re.search(r'/in/([^/]+)', path)
                return match.group(1) if match else None
            
            elif platform in ['twitter', 'github', 'instagram']:
                parts = path.split('/')
                return parts[0] if parts else None
            
            elif platform == 'weibo':
                match = re.search(r'/u/(\d+)', path)
                return match.group(1) if match else None
            
            return None
            
        except Exception as e:
            logger.warning(f"提取用户名失败: {e}")
            return None
    
    def _calculate_username_similarity(self, username1: str, username2: str) -> float:
        """计算用户名相似度"""
        if username1.lower() == username2.lower():
            return 1.0
        
        # 使用编辑距离计算相似度
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        max_len = max(len(username1), len(username2))
        distance = levenshtein_distance(username1.lower(), username2.lower())
        similarity = 1 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _extract_timeline_events(self, profile: PersonProfile) -> List[Dict[str, Any]]:
        """从个人资料中提取时间线事件"""
        events = []
        
        if profile.birth_date:
            events.append({
                'type': 'birth',
                'date': profile.birth_date,
                'description': '出生'
            })
        
        # 从教育经历中提取
        for edu in profile.education:
            if 'start_date' in edu:
                events.append({
                    'type': 'education',
                    'date': edu['start_date'],
                    'description': f"就读于 {edu.get('institution', '')}"
                })
        
        return events
    
    def _extract_timeline_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取时间信息"""
        events = []
        
        # 使用正则表达式提取年份
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        
        for year in years:
            events.append({
                'type': 'mention',
                'year': int(year),
                'description': f"提及年份 {year}"
            })
        
        return events
    
    def _calculate_timeline_consistency(self, 
                                      target_events: List[Dict[str, Any]], 
                                      result_events: List[Dict[str, Any]]) -> float:
        """计算时间线一致性"""
        if not target_events or not result_events:
            return 0.0
        
        matches = 0
        total = len(target_events)
        
        for target_event in target_events:
            target_year = getattr(target_event.get('date'), 'year', target_event.get('year'))
            
            for result_event in result_events:
                result_year = result_event.get('year')
                
                if target_year and result_year and abs(target_year - result_year) <= 2:
                    matches += 1
                    break
        
        return matches / total if total > 0 else 0.0
    
    def _extract_locations_from_text(self, text: str) -> List[str]:
        """从文本中提取地理位置"""
        # 简化的位置提取，实际应用中可以使用NER模型
        import re
        
        # 中国城市名称模式
        chinese_cities = re.findall(r'[\u4e00-\u9fff]+(?:市|区|县|省)', text)
        
        # 英文地名模式
        english_locations = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Province|Country))\b', text)
        
        locations = chinese_cities + english_locations
        return list(set(locations))  # 去重
    
    def _calculate_location_similarity(self, location1: str, location2: str) -> float:
        """计算地理位置相似度"""
        if location1.lower() == location2.lower():
            return 1.0
        
        # 检查是否有包含关系
        if location1.lower() in location2.lower() or location2.lower() in location1.lower():
            return 0.8
        
        # 使用简单的字符串相似度
        common_chars = set(location1.lower()) & set(location2.lower())
        total_chars = set(location1.lower()) | set(location2.lower())
        
        if total_chars:
            similarity = len(common_chars) / len(total_chars)
            return similarity
        
        return 0.0
    
    def _build_ai_verification_prompt(self, 
                                    profile: PersonProfile, 
                                    search_results: List[SearchResult]) -> str:
        """构建AI验证提示"""
        prompt = f"""
请分析以下信息，判断搜索结果是否与目标人物为同一人：

目标人物信息：
- 姓名: {profile.name}
- 职位: {profile.current_job or '未知'}
- 公司: {profile.current_company or '未知'}
- 地址: {profile.address or '未知'}

搜索结果:
"""
        
        for i, result in enumerate(search_results[:5], 1):  # 限制结果数量
            prompt += f"""
{i}. 标题: {result.title}
   来源: {result.url}
   内容: {result.snippet[:200]}...
"""
        
        prompt += """
请分析这些搜索结果是否与目标人物匹配，给出0-1之间的置信度分数，并简要说明理由。
格式: 置信度: X.XX, 理由: [你的分析]
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str) -> Tuple[float, str]:
        """解析AI响应"""
        try:
            import re
            
            # 提取置信度
            confidence_match = re.search(r'置信度[：:]\s*(\d+\.?\d*)', response)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.0
            
            # 提取理由
            reason_match = re.search(r'理由[：:]\s*(.+)', response)
            reasoning = reason_match.group(1).strip() if reason_match else response
            
            return confidence, reasoning
            
        except Exception as e:
            logger.warning(f"解析AI响应失败: {e}")
            return 0.0, response
    
    def _generate_reasoning(self, 
                          methods: List[str], 
                          scores: List[float], 
                          final_score: float) -> str:
        """生成验证推理说明"""
        reasoning_parts = []
        
        method_names = {
            'text_similarity': '文本相似度分析',
            'name_matching': '姓名匹配验证',
            'social_networks': '社交网络验证',
            'timeline_consistency': '时间线一致性',
            'location_consistency': '地理位置一致性',
            'ai_reasoning': 'AI推理分析'
        }
        
        for method, score in zip(methods, scores):
            method_name = method_names.get(method, method)
            reasoning_parts.append(f"{method_name}: {score:.2f}")
        
        reasoning = f"验证方法及评分: {', '.join(reasoning_parts)}. "
        reasoning += f"综合置信度: {final_score:.2f}. "
        
        if final_score > 0.8:
            reasoning += "高度可信，很可能为同一人。"
        elif final_score > 0.6:
            reasoning += "中等可信度，可能为同一人。"
        elif final_score > 0.4:
            reasoning += "较低可信度，需要更多证据。"
        else:
            reasoning += "可信度很低，很可能不是同一人。"
        
        return reasoning 