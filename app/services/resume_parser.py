"""
履历自动整理服务
从各种数据源中提取和整理人物履历信息
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from app.models.person import PersonProfile, SearchResult
from app.core.logger import logger, log_audit
from app.core.config import settings

@dataclass
class WorkExperience:
    """工作经历"""
    company: str
    position: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_current: bool = False
    description: str = ""
    location: str = ""
    source: str = ""
    confidence: float = 0.0

@dataclass
class Education:
    """教育经历"""
    institution: str
    degree: str = ""
    major: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    gpa: Optional[float] = None
    description: str = ""
    source: str = ""
    confidence: float = 0.0

@dataclass
class Achievement:
    """成就奖项"""
    title: str
    organization: str = ""
    date: Optional[datetime] = None
    description: str = ""
    category: str = ""  # award, certification, publication, etc.
    source: str = ""
    confidence: float = 0.0

@dataclass
class Skill:
    """技能"""
    name: str
    category: str = ""  # technical, language, soft, etc.
    proficiency: str = ""  # beginner, intermediate, advanced, expert
    years_experience: Optional[int] = None
    source: str = ""
    confidence: float = 0.0

@dataclass
class ResumeData:
    """完整履历数据"""
    personal_info: Dict[str, Any]
    work_experiences: List[WorkExperience]
    education: List[Education]
    skills: List[Skill]
    achievements: List[Achievement]
    timeline: List[Dict[str, Any]]
    summary: str = ""
    confidence_score: float = 0.0

class ResumeParser:
    """履历解析器"""
    
    def __init__(self):
        # 日期模式
        self.date_patterns = [
            r'(\d{4})年(\d{1,2})月',
            r'(\d{4})-(\d{1,2})',
            r'(\d{1,2})/(\d{4})',
            r'(\d{4})\s*年',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})'
        ]
        
        # 职位关键词
        self.position_keywords = [
            'CEO', 'CTO', 'CFO', 'COO', 'VP', 'Director', 'Manager', 'Lead', 'Senior', 'Junior',
            '总监', '经理', '主管', '总裁', '副总', '首席', '总经理', '部长', '组长', '负责人',
            '工程师', '开发', '设计师', '分析师', '顾问', '专家', '架构师'
        ]
        
        # 公司关键词
        self.company_keywords = [
            'Company', 'Corp', 'Corporation', 'Inc', 'Ltd', 'LLC', 'Co.',
            '公司', '集团', '企业', '科技', '网络', '信息', '软件', '系统'
        ]
        
        # 学历关键词
        self.degree_keywords = [
            'PhD', 'Ph.D', 'Doctor', 'Master', 'Bachelor', 'MBA', 'MS', 'BS', 'BA', 'MA',
            '博士', '硕士', '学士', '本科', '研究生', '大学', '学院'
        ]
        
        # 技能类别
        self.skill_categories = {
            'programming': ['Python', 'Java', 'JavaScript', 'C++', 'Go', 'Rust', 'PHP', 'Ruby'],
            'database': ['MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server'],
            'framework': ['Django', 'Flask', 'React', 'Vue', 'Angular', 'Spring', 'Laravel'],
            'tools': ['Git', 'Docker', 'Kubernetes', 'Jenkins', 'AWS', 'Azure', 'GCP'],
            'language': ['English', 'Chinese', 'Japanese', 'Korean', 'French', 'German', 'Spanish', '英语', '中文', '日语']
        }
    
    async def parse_resume_from_results(self, 
                                      target_person: PersonProfile, 
                                      search_results: List[SearchResult]) -> ResumeData:
        """从搜索结果中解析履历信息"""
        
        log_audit("RESUME_PARSING", target_person.name, details="开始解析履历信息")
        
        # 初始化履历数据
        resume_data = ResumeData(
            personal_info=self._extract_personal_info(target_person),
            work_experiences=[],
            education=[],
            skills=[],
            achievements=[],
            timeline=[]
        )
        
        # 从每个搜索结果中提取信息
        for result in search_results:
            try:
                # 提取工作经历
                work_exp = self._extract_work_experience(result)
                resume_data.work_experiences.extend(work_exp)
                
                # 提取教育经历
                education = self._extract_education(result)
                resume_data.education.extend(education)
                
                # 提取技能
                skills = self._extract_skills(result)
                resume_data.skills.extend(skills)
                
                # 提取成就
                achievements = self._extract_achievements(result)
                resume_data.achievements.extend(achievements)
                
            except Exception as e:
                logger.warning(f"解析履历信息失败 {result.url}: {e}")
                continue
        
        # 数据清理和去重
        resume_data = self._clean_and_deduplicate(resume_data)
        
        # 构建时间线
        resume_data.timeline = self._build_timeline(resume_data)
        
        # 生成摘要
        resume_data.summary = self._generate_summary(resume_data)
        
        # 计算整体置信度
        resume_data.confidence_score = self._calculate_overall_confidence(resume_data)
        
        log_audit("RESUME_PARSING", target_person.name, 
                 details=f"履历解析完成，工作经历: {len(resume_data.work_experiences)}, "
                        f"教育经历: {len(resume_data.education)}, 置信度: {resume_data.confidence_score:.2f}")
        
        return resume_data
    
    def _extract_personal_info(self, person: PersonProfile) -> Dict[str, Any]:
        """提取个人基本信息"""
        return {
            'name': person.name,
            'alternative_names': person.alternative_names,
            'email': person.email,
            'phone': person.phone,
            'address': person.address,
            'age': person.age,
            'current_job': person.current_job,
            'current_company': person.current_company
        }
    
    def _extract_work_experience(self, result: SearchResult) -> List[WorkExperience]:
        """从搜索结果中提取工作经历"""
        experiences = []
        content = f"{result.title} {result.content}"
        
        # 查找工作经历模式
        work_patterns = [
            r'(在|任职于|工作于)\s*([^，。\s]+(?:公司|集团|企业|科技|网络))\s*(?:担任|任)?\s*([^，。\s]+)',
            r'([^，。\s]+(?:公司|集团|企业|科技|网络))\s*([^，。\s]+(?:总监|经理|主管|CEO|CTO|工程师))',
            r'(\d{4})\s*[-年]\s*(\d{4})?\s*([^，。\s]+(?:公司|集团))\s*([^，。\s]+)'
        ]
        
        for pattern in work_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    if len(match) >= 3:
                        company = self._clean_company_name(match[1] if '公司' in match[1] else match[2])
                        position = self._clean_position_name(match[2] if len(match) > 2 else match[1])
                        
                        # 提取日期
                        start_date, end_date = self._extract_dates_from_context(content, company)
                        
                        experience = WorkExperience(
                            company=company,
                            position=position,
                            start_date=start_date,
                            end_date=end_date,
                            is_current=(end_date is None and start_date is not None),
                            source=result.source,
                            confidence=result.reliability_score * 0.8
                        )
                        
                        experiences.append(experience)
                        
                except Exception as e:
                    logger.debug(f"解析工作经历失败: {e}")
                    continue
        
        return experiences
    
    def _extract_education(self, result: SearchResult) -> List[Education]:
        """从搜索结果中提取教育经历"""
        educations = []
        content = f"{result.title} {result.content}"
        
        # 查找教育经历模式
        education_patterns = [
            r'(毕业于|就读于|学习于)\s*([^，。\s]+(?:大学|学院|学校))\s*([^，。\s]*)',
            r'([^，。\s]+(?:大学|学院|学校))\s*([^，。\s]+(?:博士|硕士|学士|本科|研究生))',
            r'(\d{4})\s*[-年]\s*(\d{4})?\s*([^，。\s]+(?:大学|学院))\s*([^，。\s]*)'
        ]
        
        for pattern in education_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    if len(match) >= 2:
                        institution = self._clean_institution_name(match[1] if '大学' in match[1] else match[2])
                        degree = self._extract_degree_info(match[2] if len(match) > 2 else match[1])
                        
                        # 提取日期
                        start_date, end_date = self._extract_dates_from_context(content, institution)
                        
                        education = Education(
                            institution=institution,
                            degree=degree,
                            start_date=start_date,
                            end_date=end_date,
                            source=result.source,
                            confidence=result.reliability_score * 0.7
                        )
                        
                        educations.append(education)
                        
                except Exception as e:
                    logger.debug(f"解析教育经历失败: {e}")
                    continue
        
        return educations
    
    def _extract_skills(self, result: SearchResult) -> List[Skill]:
        """从搜索结果中提取技能"""
        skills = []
        content = f"{result.title} {result.content}"
        
        # 从已知技能列表中匹配
        for category, skill_list in self.skill_categories.items():
            for skill_name in skill_list:
                if skill_name.lower() in content.lower():
                    # 提取熟练程度
                    proficiency = self._extract_proficiency(content, skill_name)
                    
                    skill = Skill(
                        name=skill_name,
                        category=category,
                        proficiency=proficiency,
                        source=result.source,
                        confidence=result.reliability_score * 0.6
                    )
                    
                    skills.append(skill)
        
        return skills
    
    def _extract_achievements(self, result: SearchResult) -> List[Achievement]:
        """从搜索结果中提取成就奖项"""
        achievements = []
        content = f"{result.title} {result.content}"
        
        # 查找成就模式
        achievement_patterns = [
            r'(获得|荣获|得到|获奖)\s*([^，。\s]+(?:奖|证书|认证))',
            r'(发表|出版)\s*([^，。\s]+(?:论文|文章|书籍))',
            r'(专利|Patent)\s*([^，。\s]+)',
            r'(Award|Prize|Certificate)\s*([^，。\s]+)'
        ]
        
        for pattern in achievement_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    if len(match) >= 2:
                        title = match[1]
                        category = self._categorize_achievement(match[0], title)
                        
                        # 提取日期
                        date = self._extract_single_date_from_context(content, title)
                        
                        achievement = Achievement(
                            title=title,
                            category=category,
                            date=date,
                            source=result.source,
                            confidence=result.reliability_score * 0.7
                        )
                        
                        achievements.append(achievement)
                        
                except Exception as e:
                    logger.debug(f"解析成就信息失败: {e}")
                    continue
        
        return achievements
    
    def _extract_dates_from_context(self, content: str, entity: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """从上下文中提取日期范围"""
        start_date = None
        end_date = None
        
        # 在实体周围查找日期
        entity_pos = content.find(entity)
        if entity_pos == -1:
            return start_date, end_date
        
        # 搜索范围：实体前后100个字符
        search_start = max(0, entity_pos - 100)
        search_end = min(len(content), entity_pos + len(entity) + 100)
        search_text = content[search_start:search_end]
        
        # 查找日期模式
        for pattern in self.date_patterns:
            matches = re.findall(pattern, search_text)
            if matches:
                dates = []
                for match in matches:
                    try:
                        date = self._parse_date_match(match)
                        if date:
                            dates.append(date)
                    except:
                        continue
                
                if dates:
                    dates.sort()
                    start_date = dates[0]
                    if len(dates) > 1:
                        end_date = dates[-1]
                break
        
        return start_date, end_date
    
    def _extract_single_date_from_context(self, content: str, entity: str) -> Optional[datetime]:
        """从上下文中提取单个日期"""
        start_date, end_date = self._extract_dates_from_context(content, entity)
        return end_date or start_date
    
    def _parse_date_match(self, match: tuple) -> Optional[datetime]:
        """解析日期匹配结果"""
        try:
            if len(match) == 2:
                if match[0].isdigit() and match[1].isdigit():
                    year = int(match[0]) if int(match[0]) > 31 else int(match[1])
                    month = int(match[1]) if int(match[0]) > 31 else int(match[0])
                    if 1900 <= year <= 2030 and 1 <= month <= 12:
                        return datetime(year, month, 1)
                elif match[0].isdigit():
                    # 只有年份
                    year = int(match[0])
                    if 1900 <= year <= 2030:
                        return datetime(year, 1, 1)
            elif len(match) == 3:
                # 月/日/年 格式
                try:
                    month = int(match[0])
                    year = int(match[2])
                    if 1900 <= year <= 2030 and 1 <= month <= 12:
                        return datetime(year, month, 1)
                except:
                    pass
        except:
            pass
        
        return None
    
    def _clean_company_name(self, company: str) -> str:
        """清理公司名称"""
        # 移除多余空格和特殊字符
        company = re.sub(r'\s+', ' ', company.strip())
        
        # 统一公司后缀
        suffixes = ['有限公司', '股份有限公司', 'Co.', 'Corp.', 'Inc.', 'Ltd.']
        for suffix in suffixes:
            if company.endswith(suffix):
                break
        else:
            # 如果没有后缀，可能需要补充
            if any(keyword in company for keyword in self.company_keywords):
                pass  # 已经包含公司关键词
            else:
                company += '公司' if any('\u4e00' <= char <= '\u9fff' for char in company) else ' Corp.'
        
        return company
    
    def _clean_position_name(self, position: str) -> str:
        """清理职位名称"""
        # 移除多余空格
        position = re.sub(r'\s+', ' ', position.strip())
        
        # 移除常见前缀
        prefixes = ['担任', '任职', '任', '职位:', '岗位:']
        for prefix in prefixes:
            position = position.replace(prefix, '')
        
        return position.strip()
    
    def _clean_institution_name(self, institution: str) -> str:
        """清理教育机构名称"""
        institution = re.sub(r'\s+', ' ', institution.strip())
        return institution
    
    def _extract_degree_info(self, text: str) -> str:
        """提取学位信息"""
        for degree in self.degree_keywords:
            if degree.lower() in text.lower():
                return degree
        return text.strip()
    
    def _extract_proficiency(self, content: str, skill: str) -> str:
        """提取技能熟练程度"""
        skill_context = self._get_context_around_word(content, skill, 50)
        
        proficiency_keywords = {
            'expert': ['专家', '精通', 'expert', 'advanced', 'proficient'],
            'advanced': ['高级', '资深', 'senior', 'advanced'],
            'intermediate': ['中级', '熟悉', 'intermediate', 'familiar'],
            'beginner': ['初级', '入门', 'beginner', 'basic', 'junior']
        }
        
        for level, keywords in proficiency_keywords.items():
            for keyword in keywords:
                if keyword in skill_context.lower():
                    return level
        
        return 'intermediate'  # 默认中级
    
    def _get_context_around_word(self, text: str, word: str, window: int) -> str:
        """获取单词周围的上下文"""
        pos = text.lower().find(word.lower())
        if pos == -1:
            return ""
        
        start = max(0, pos - window)
        end = min(len(text), pos + len(word) + window)
        
        return text[start:end]
    
    def _categorize_achievement(self, action: str, title: str) -> str:
        """分类成就类型"""
        action_lower = action.lower()
        title_lower = title.lower()
        
        if any(word in action_lower for word in ['发表', 'publish', '出版']):
            return 'publication'
        elif any(word in title_lower for word in ['专利', 'patent']):
            return 'patent'
        elif any(word in title_lower for word in ['证书', 'certificate', '认证']):
            return 'certification'
        elif any(word in title_lower for word in ['奖', 'award', 'prize']):
            return 'award'
        else:
            return 'other'
    
    def _clean_and_deduplicate(self, resume_data: ResumeData) -> ResumeData:
        """清理和去重履历数据"""
        # 工作经历去重
        unique_work = []
        seen_work = set()
        
        for work in resume_data.work_experiences:
            key = (work.company.lower(), work.position.lower())
            if key not in seen_work:
                seen_work.add(key)
                unique_work.append(work)
        
        resume_data.work_experiences = sorted(unique_work, 
                                            key=lambda x: x.start_date or datetime.min, 
                                            reverse=True)
        
        # 教育经历去重
        unique_education = []
        seen_education = set()
        
        for edu in resume_data.education:
            key = (edu.institution.lower(), edu.degree.lower())
            if key not in seen_education:
                seen_education.add(key)
                unique_education.append(edu)
        
        resume_data.education = sorted(unique_education,
                                     key=lambda x: x.start_date or datetime.min,
                                     reverse=True)
        
        # 技能去重
        unique_skills = []
        seen_skills = set()
        
        for skill in resume_data.skills:
            key = skill.name.lower()
            if key not in seen_skills:
                seen_skills.add(key)
                unique_skills.append(skill)
        
        resume_data.skills = unique_skills
        
        # 成就去重
        unique_achievements = []
        seen_achievements = set()
        
        for achievement in resume_data.achievements:
            key = achievement.title.lower()
            if key not in seen_achievements:
                seen_achievements.add(key)
                unique_achievements.append(achievement)
        
        resume_data.achievements = sorted(unique_achievements,
                                        key=lambda x: x.date or datetime.min,
                                        reverse=True)
        
        return resume_data
    
    def _build_timeline(self, resume_data: ResumeData) -> List[Dict[str, Any]]:
        """构建时间线"""
        timeline_events = []
        
        # 添加工作经历事件
        for work in resume_data.work_experiences:
            if work.start_date:
                timeline_events.append({
                    'date': work.start_date,
                    'type': 'work_start',
                    'title': f"开始在{work.company}担任{work.position}",
                    'details': work,
                    'category': 'work'
                })
            
            if work.end_date:
                timeline_events.append({
                    'date': work.end_date,
                    'type': 'work_end',
                    'title': f"结束在{work.company}的工作",
                    'details': work,
                    'category': 'work'
                })
        
        # 添加教育经历事件
        for edu in resume_data.education:
            if edu.start_date:
                timeline_events.append({
                    'date': edu.start_date,
                    'type': 'education_start',
                    'title': f"开始在{edu.institution}学习{edu.degree}",
                    'details': edu,
                    'category': 'education'
                })
            
            if edu.end_date:
                timeline_events.append({
                    'date': edu.end_date,
                    'type': 'education_end',
                    'title': f"从{edu.institution}毕业",
                    'details': edu,
                    'category': 'education'
                })
        
        # 添加成就事件
        for achievement in resume_data.achievements:
            if achievement.date:
                timeline_events.append({
                    'date': achievement.date,
                    'type': 'achievement',
                    'title': f"获得{achievement.title}",
                    'details': achievement,
                    'category': 'achievement'
                })
        
        # 按时间排序
        timeline_events.sort(key=lambda x: x['date'])
        
        return timeline_events
    
    def _generate_summary(self, resume_data: ResumeData) -> str:
        """生成履历摘要"""
        summary_parts = []
        
        # 基本信息
        name = resume_data.personal_info.get('name', '该人员')
        summary_parts.append(f"{name}的履历信息如下：")
        
        # 当前职位
        current_job = resume_data.personal_info.get('current_job')
        current_company = resume_data.personal_info.get('current_company')
        if current_job and current_company:
            summary_parts.append(f"目前在{current_company}担任{current_job}。")
        
        # 工作经历
        if resume_data.work_experiences:
            work_count = len(resume_data.work_experiences)
            companies = [work.company for work in resume_data.work_experiences[:3]]
            summary_parts.append(f"拥有{work_count}段工作经历，曾就职于{', '.join(companies)}等公司。")
        
        # 教育背景
        if resume_data.education:
            latest_edu = resume_data.education[0]
            summary_parts.append(f"毕业于{latest_edu.institution}，获得{latest_edu.degree}学位。")
        
        # 技能
        if resume_data.skills:
            skill_categories = set(skill.category for skill in resume_data.skills)
            summary_parts.append(f"掌握{len(resume_data.skills)}项技能，涵盖{', '.join(skill_categories)}等领域。")
        
        # 成就
        if resume_data.achievements:
            achievement_count = len(resume_data.achievements)
            summary_parts.append(f"获得{achievement_count}项成就和奖项。")
        
        return ' '.join(summary_parts)
    
    def _calculate_overall_confidence(self, resume_data: ResumeData) -> float:
        """计算整体置信度"""
        all_confidences = []
        
        # 收集所有置信度分数
        for work in resume_data.work_experiences:
            all_confidences.append(work.confidence)
        
        for edu in resume_data.education:
            all_confidences.append(edu.confidence)
        
        for skill in resume_data.skills:
            all_confidences.append(skill.confidence)
        
        for achievement in resume_data.achievements:
            all_confidences.append(achievement.confidence)
        
        if all_confidences:
            return sum(all_confidences) / len(all_confidences)
        else:
            return 0.0
    
    def export_resume_to_format(self, resume_data: ResumeData, format: str = 'json') -> str:
        """导出履历为指定格式"""
        if format == 'json':
            return self._export_to_json(resume_data)
        elif format == 'markdown':
            return self._export_to_markdown(resume_data)
        elif format == 'text':
            return self._export_to_text(resume_data)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _export_to_json(self, resume_data: ResumeData) -> str:
        """导出为JSON格式"""
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj.__dict__ if hasattr(obj, '__dict__') else str(obj)
        
        return json.dumps(resume_data.__dict__, default=serialize_datetime, 
                         indent=2, ensure_ascii=False)
    
    def _export_to_markdown(self, resume_data: ResumeData) -> str:
        """导出为Markdown格式"""
        lines = []
        
        # 标题
        name = resume_data.personal_info.get('name', '履历')
        lines.append(f"# {name}的履历")
        lines.append("")
        
        # 个人信息
        lines.append("## 个人信息")
        for key, value in resume_data.personal_info.items():
            if value:
                key_cn = {
                    'name': '姓名', 'email': '邮箱', 'phone': '电话', 
                    'address': '地址', 'current_job': '当前职位', 
                    'current_company': '当前公司'
                }.get(key, key)
                lines.append(f"- **{key_cn}**: {value}")
        lines.append("")
        
        # 工作经历
        if resume_data.work_experiences:
            lines.append("## 工作经历")
            for work in resume_data.work_experiences:
                date_range = ""
                if work.start_date:
                    date_range = work.start_date.strftime("%Y年%m月")
                    if work.end_date:
                        date_range += f" - {work.end_date.strftime('%Y年%m月')}"
                    elif work.is_current:
                        date_range += " - 至今"
                
                lines.append(f"### {work.position} @ {work.company}")
                if date_range:
                    lines.append(f"**时间**: {date_range}")
                if work.description:
                    lines.append(f"**描述**: {work.description}")
                lines.append("")
        
        # 教育经历
        if resume_data.education:
            lines.append("## 教育经历")
            for edu in resume_data.education:
                date_range = ""
                if edu.start_date:
                    date_range = edu.start_date.strftime("%Y年%m月")
                    if edu.end_date:
                        date_range += f" - {edu.end_date.strftime('%Y年%m月')}"
                
                lines.append(f"### {edu.degree} @ {edu.institution}")
                if date_range:
                    lines.append(f"**时间**: {date_range}")
                if edu.major:
                    lines.append(f"**专业**: {edu.major}")
                lines.append("")
        
        # 技能
        if resume_data.skills:
            lines.append("## 技能")
            skill_by_category = {}
            for skill in resume_data.skills:
                category = skill.category or '其他'
                if category not in skill_by_category:
                    skill_by_category[category] = []
                skill_by_category[category].append(skill.name)
            
            for category, skills in skill_by_category.items():
                lines.append(f"- **{category}**: {', '.join(skills)}")
            lines.append("")
        
        # 成就奖项
        if resume_data.achievements:
            lines.append("## 成就奖项")
            for achievement in resume_data.achievements:
                date_str = achievement.date.strftime("%Y年%m月") if achievement.date else ""
                lines.append(f"- **{achievement.title}** {date_str}")
                if achievement.organization:
                    lines.append(f"  - 颁发机构: {achievement.organization}")
                if achievement.description:
                    lines.append(f"  - 描述: {achievement.description}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _export_to_text(self, resume_data: ResumeData) -> str:
        """导出为纯文本格式"""
        lines = []
        
        # 基本信息
        name = resume_data.personal_info.get('name', '履历')
        lines.append(f"{name}的履历")
        lines.append("=" * (len(name) * 2 + 4))
        lines.append("")
        
        # 摘要
        if resume_data.summary:
            lines.append("摘要:")
            lines.append(resume_data.summary)
            lines.append("")
        
        # 工作经历
        if resume_data.work_experiences:
            lines.append("工作经历:")
            for i, work in enumerate(resume_data.work_experiences, 1):
                lines.append(f"{i}. {work.position} @ {work.company}")
                if work.start_date:
                    date_str = work.start_date.strftime("%Y年%m月")
                    if work.end_date:
                        date_str += f" - {work.end_date.strftime('%Y年%m月')}"
                    elif work.is_current:
                        date_str += " - 至今"
                    lines.append(f"   时间: {date_str}")
            lines.append("")
        
        return '\n'.join(lines) 