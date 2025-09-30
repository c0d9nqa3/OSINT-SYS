"""
AI驱动的智能文本解析器
真正实现"识别到什么字段就录入什么字段"的智能解析

技术栈：
- OpenAI GPT-4 API：智能字段识别和分类
- jieba：中文分词
- 正则表达式：基础模式匹配
- 语义相似度：智能去重
"""

import re
import json
import jieba
import openai
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import difflib
from dataclasses import dataclass

from app.core.config import settings
from app.core.logger import logger

# 配置jieba
jieba.setLogLevel(jieba.logging.INFO)

@dataclass
class FieldMapping:
    """字段映射配置"""
    field_name: str
    synonyms: List[str]  # 同义词
    data_type: str  # 'string', 'date', 'phone', 'id_number', 'list'
    priority: int  # 优先级，数字越小优先级越高

# 预定义的字段映射
FIELD_MAPPINGS = [
    FieldMapping("姓名", ["name", "名字", "全名", "真实姓名"], "string", 1),
    FieldMapping("性别", ["gender", "sex"], "string", 2),
    FieldMapping("出生日期", ["生日", "birth_date", "出生年月", "birth", "birthday"], "date", 3),
    FieldMapping("身份证号", ["身份证", "id_number", "身份证号码", "证件号"], "id_number", 4),
    FieldMapping("电话", ["手机", "手机号", "联系电话", "phone", "mobile", "tel"], "phone", 5),
    FieldMapping("地址", ["住址", "address", "居住地", "家庭地址"], "string", 6),
    FieldMapping("户籍地", ["籍贯", "出生地", "户口所在地"], "string", 7),
    FieldMapping("户籍地址", ["真户籍地址", "户口地址", "户籍详细地址"], "string", 8),
    FieldMapping("生肖", ["属相"], "string", 9),
    FieldMapping("星座", ["constellation"], "string", 10),
    FieldMapping("职业", ["工作", "职位", "occupation", "job"], "string", 11),
    FieldMapping("公司", ["工作单位", "company"], "string", 12),
]

class AITextParser:
    """AI驱动的智能文本解析器"""
    
    def __init__(self):
        self.openai_client = None
        self._initialize_openai()
        
    def _initialize_openai(self):
        """初始化OpenAI客户端"""
        try:
            # 从环境变量获取API密钥
            api_key = settings.OPENAI_API_KEY
            if api_key and api_key.strip():
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("OpenAI客户端初始化成功")
            else:
                logger.warning("未配置OpenAI API密钥，将使用传统解析方法")
        except ImportError:
            logger.warning("OpenAI库未安装，将使用传统解析方法")
        except Exception as e:
            logger.warning(f"OpenAI初始化失败: {e}，将使用传统解析方法")
    
    def parse_text_intelligent(self, text: str) -> Dict[str, Any]:
        """
        智能解析文本，提取所有可能的字段
        """
        logger.info("开始AI智能文本解析...")
        
        # 1. 预处理：分离主人物和同户人
        main_section, household_section = self._split_main_and_household(text)
        
        # 2. AI增强的字段提取
        if self.openai_client:
            main_fields = self._ai_extract_fields(main_section)
        else:
            main_fields = self._traditional_extract_fields(main_section)
        
        # 3. 同户人信息解析
        household_members = []
        if household_section:
            household_members = self._parse_household_members(household_section)
        
        # 4. 字段标准化和去重
        normalized_fields = self._normalize_and_deduplicate(main_fields)
        
        # 5. 数据类型转换和验证
        validated_fields = self._validate_and_convert(normalized_fields)
        
        result = {
            'main_person': validated_fields,
            'household_members': household_members,
            'raw_text': text,
            'parsing_method': 'ai' if self.openai_client else 'traditional',
            'confidence_score': self._calculate_confidence(validated_fields)
        }
        
        logger.info(f"解析完成，提取到 {len(validated_fields)} 个字段，{len(household_members)} 个同户人")
        return result
    
    def _ai_extract_fields(self, text: str) -> Dict[str, Any]:
        """使用AI模型提取字段"""
        try:
            prompt = f"""
请从以下文本中提取所有人物相关的信息字段。要求：

1. 提取任何形如"字段名: 值"或"字段名：值"的信息
2. 识别隐含的字段信息（如文本中的性别、生肖、星座等）
3. 返回JSON格式，包含所有找到的字段
4. 字段名使用中文标准化命名
5. 如果有多个相同类型字段，保留所有值

文本内容：
{text}

请返回JSON格式的结果，例如：
{{
    "姓名": "张三",
    "性别": "男",
    "出生日期": "1990-01-01",
    "电话": ["13800138000"],
    "生肖": "马",
    "星座": "水瓶座",
    "其他字段": "对应值"
}}
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的文本信息提取专家，擅长从中文文本中提取结构化信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.warning("AI返回结果中未找到有效JSON")
                return self._traditional_extract_fields(text)
                
        except Exception as e:
            logger.error(f"AI字段提取失败: {e}")
            return self._traditional_extract_fields(text)
    
    def _traditional_extract_fields(self, text: str) -> Dict[str, Any]:
        """传统正则表达式方法提取字段"""
        fields = {}
        
        # 通用键值对模式
        kv_patterns = [
            r'([^：:\n\r]{1,15})\s*[:：]\s*([^，\n\r；;]{1,100})',  # key: value
            r'([^：:\n\r]{1,15})\s*[：:]\s*([^，\n\r；;]{1,100})',  # key：value
        ]
        
        for pattern in kv_patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                
                if len(key) >= 2 and len(value) >= 1 and value not in ['', '-', '无', '空']:
                    fields[key] = value
        
        # 特殊字段检测
        special_patterns = {
            '性别': r'[性别]?([男女])',
            '生肖': r'生肖[：:]?\s*([^，\s,；;]{1,3})',
            '星座': r'星座[：:]?\s*([^，\s,；;]{1,10})',
            '身份证号': r'\b(\d{15}|\d{18}|\d{17}[0-9Xx])\b',
            '手机号': r'\b(1[3-9]\d{9})\b',
            '农历': r'([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]年)',
        }
        
        for field_name, pattern in special_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                if field_name in ['身份证号', '手机号']:
                    fields[field_name] = matches  # 多值
                else:
                    fields[field_name] = matches[0]  # 单值
        
        return fields
    
    def _split_main_and_household(self, text: str) -> Tuple[str, str]:
        """智能分离主人物和同户人信息"""
        lines = text.split('\n')
        main_lines = []
        household_lines = []
        in_household_section = False
        
        household_keywords = ['同户人', '同户', '同住人', '家庭成员', '户主']
        separator_patterns = [r'-{3,}', r'={3,}', r'━{3,}', r'─{3,}']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测同户人关键词
            if any(keyword in line for keyword in household_keywords):
                in_household_section = True
                household_lines.append(line)
                continue
            
            # 检测分隔线
            if any(re.search(pattern, line) for pattern in separator_patterns):
                in_household_section = True
                household_lines.append(line)
                continue
            
            # 分配到对应部分
            if in_household_section:
                household_lines.append(line)
            else:
                main_lines.append(line)
        
        return '\n'.join(main_lines), '\n'.join(household_lines)
    
    def _parse_household_members(self, household_text: str) -> List[Dict[str, Any]]:
        """解析同户人信息"""
        members = []
        
        # 使用分隔线分割每个成员
        member_sections = re.split(r'-{3,}|={3,}|━{3,}', household_text)
        
        for section in member_sections:
            section = section.strip()
            if not section or '同户人' in section:
                continue
            
            # 为每个成员段落提取字段
            if self.openai_client:
                member_fields = self._ai_extract_fields(section)
            else:
                member_fields = self._traditional_extract_fields(section)
            
            if member_fields and any(key in ['姓名', 'name'] for key in member_fields.keys()):
                members.append(member_fields)
        
        return members
    
    def _normalize_and_deduplicate(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """字段标准化和去重"""
        normalized = {}
        
        for raw_key, value in fields.items():
            # 找到最佳匹配的标准字段名
            standard_key = self._find_standard_field_name(raw_key)
            
            # 智能去重：检查是否已存在相似字段
            existing_key = self._find_similar_existing_key(standard_key, normalized.keys())
            
            final_key = existing_key if existing_key else standard_key
            
            # 合并值（如果存在重复字段）
            if final_key in normalized:
                normalized[final_key] = self._merge_field_values(normalized[final_key], value)
            else:
                normalized[final_key] = value
        
        return normalized
    
    def _find_standard_field_name(self, raw_key: str) -> str:
        """根据同义词映射找到标准字段名"""
        raw_key_lower = raw_key.lower().strip()
        
        # 精确匹配
        for mapping in FIELD_MAPPINGS:
            if raw_key in [mapping.field_name] + mapping.synonyms:
                return mapping.field_name
        
        # 模糊匹配
        for mapping in FIELD_MAPPINGS:
            for synonym in [mapping.field_name] + mapping.synonyms:
                if difflib.SequenceMatcher(None, raw_key_lower, synonym.lower()).ratio() > 0.8:
                    return mapping.field_name
        
        # 返回原字段名（首字母大写）
        return raw_key.strip()
    
    def _find_similar_existing_key(self, key: str, existing_keys: List[str], threshold: float = 0.85) -> Optional[str]:
        """查找是否已存在相似的字段名"""
        for existing_key in existing_keys:
            similarity = difflib.SequenceMatcher(None, key.lower(), existing_key.lower()).ratio()
            if similarity > threshold:
                return existing_key
        return None
    
    def _merge_field_values(self, existing_value: Any, new_value: Any) -> Any:
        """智能合并字段值"""
        if existing_value == new_value:
            return existing_value
        
        # 转换为列表格式进行合并
        if not isinstance(existing_value, list):
            existing_value = [existing_value]
        if not isinstance(new_value, list):
            new_value = [new_value]
        
        # 去重合并
        merged = existing_value.copy()
        for item in new_value:
            if item not in merged:
                merged.append(item)
        
        # 如果只有一个值，返回单值
        return merged[0] if len(merged) == 1 else merged
    
    def _validate_and_convert(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """数据类型验证和转换"""
        validated = {}
        
        for key, value in fields.items():
            try:
                # 找到字段的数据类型定义
                field_mapping = next((m for m in FIELD_MAPPINGS if m.field_name == key), None)
                data_type = field_mapping.data_type if field_mapping else 'string'
                
                # 根据数据类型进行验证和转换
                converted_value = self._convert_by_type(value, data_type)
                if converted_value is not None:
                    validated[key] = converted_value
                
            except Exception as e:
                logger.warning(f"字段 {key} 验证失败: {e}，保留原值")
                validated[key] = value
        
        return validated
    
    def _convert_by_type(self, value: Any, data_type: str) -> Any:
        """根据数据类型转换值"""
        if data_type == 'date':
            return self._parse_date(str(value))
        elif data_type == 'phone':
            return self._validate_phone(str(value))
        elif data_type == 'id_number':
            return self._validate_id_number(str(value))
        elif data_type == 'list':
            return value if isinstance(value, list) else [value]
        else:  # string
            return str(value).strip()
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """解析日期字符串"""
        date_patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{4})/(\d{1,2})/(\d{1,2})',
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                year, month, day = match.groups()
                try:
                    # 验证日期有效性
                    datetime(int(year), int(month), int(day))
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    continue
        
        return date_str  # 如果无法解析，返回原字符串
    
    def _validate_phone(self, phone_str: str) -> Optional[str]:
        """验证手机号"""
        # 清理手机号（移除空格、横线等）
        clean_phone = re.sub(r'[\s\-\(\)]', '', phone_str)
        
        # 中国手机号模式
        if re.match(r'^1[3-9]\d{9}$', clean_phone):
            return clean_phone
        
        return phone_str  # 如果不匹配，返回原字符串
    
    def _validate_id_number(self, id_str: str) -> Optional[str]:
        """验证身份证号"""
        clean_id = re.sub(r'[\s\-]', '', id_str)
        
        # 18位身份证
        if re.match(r'^\d{17}[\dXx]$', clean_id):
            return clean_id.upper()
        # 15位身份证
        elif re.match(r'^\d{15}$', clean_id):
            return clean_id
        
        return id_str  # 如果不匹配，返回原字符串
    
    def _calculate_confidence(self, fields: Dict[str, Any]) -> float:
        """计算解析置信度"""
        if not fields:
            return 0.0
        
        # 基础分数
        base_score = 0.5
        
        # 关键字段加分
        key_fields = ['姓名', '性别', '出生日期', '身份证号']
        key_field_bonus = sum(0.1 for field in key_fields if field in fields)
        
        # 字段数量加分
        field_count_bonus = min(len(fields) * 0.05, 0.3)
        
        # 数据质量加分（有效的电话、身份证等）
        quality_bonus = 0
        if '身份证号' in fields and self._validate_id_number(str(fields['身份证号'])):
            quality_bonus += 0.1
        if '电话' in fields and self._validate_phone(str(fields['电话'])):
            quality_bonus += 0.1
        
        total_score = base_score + key_field_bonus + field_count_bonus + quality_bonus
        return min(total_score, 1.0)


# 创建全局解析器实例
ai_parser = AITextParser() 