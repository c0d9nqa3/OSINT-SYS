"""
智能文本解析器（基于jieba + 规则引擎）
实现"识别到什么字段就录入什么字段"的智能解析
无需外部AI API依赖
"""

import re
import jieba
import jieba.posseg as pseg
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import difflib

from app.core.logger import logger

# 配置jieba
jieba.setLogLevel(jieba.logging.INFO)

class SmartTextParser:
    """智能文本解析器"""
    
    def __init__(self):
        self.field_patterns = self._build_field_patterns()
        self.value_extractors = self._build_value_extractors()
        # 添加更多字段模式
        self._load_extended_patterns()
        self._load_custom_dict()
    
    def _load_custom_dict(self):
        """加载自定义词典"""
        custom_words = [
            # 身份信息
            '身份证号', '身份证号码', '证件号', '户籍地', '户籍地址', '真户籍地址',
            '出生地', '籍贯', '户口所在地', '星座', '生肖', '属相',
            # 联系方式
            '手机号', '联系电话', '联系方式', '邮箱地址', '电子邮箱',
            # 职业信息
            '工作单位', '职业', '职位', '岗位', '工作', '从事',
            # 地址信息
            '家庭地址', '居住地', '住址', '通讯地址', '快递地址',
            # 时间信息
            '出生年月', '出生日期', '生日', '年龄',
            # 同户人相关
            '同户人', '同户', '家庭成员', '户主'
        ]
        
        for word in custom_words:
            jieba.add_word(word)
        
        logger.info(f"已加载 {len(custom_words)} 个自定义词汇")
    
    def _load_extended_patterns(self):
        """加载扩展的字段识别模式"""
        # 扩展字段模式
        extended_fields = {
            '微博': ['微博', '微博链接', 'weibo', 'weibo_url'],
            '微信': ['微信', '微信号', 'wechat', 'wx'],
            'QQ': ['QQ', 'qq', 'QQ号'],
            '车牌': ['车牌', '车牌号', '牌照', 'license_plate'],
            '手机归属地': ['手机归属地', '归属地'],
            '区号': ['区号', 'area_code'],
            '区划代码': ['区划代码', 'region_code'],
            '运营商': ['运营商', '中国联通', '中国移动', '中国电信'],
        }
        
        # 合并到现有字段模式
        for field_name, patterns in extended_fields.items():
            if field_name not in self.field_patterns:
                self.field_patterns[field_name] = patterns
            else:
                self.field_patterns[field_name].extend(patterns)
        
        # 扩展值提取模式
        extended_extractors = {
             'URL': r'https?://[\w\.\-/\?=&%#]+|www\.[\w\.\-/\?=&%#]+',
             '车牌号': r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{4,5}[A-Z0-9挂学警港澳]',
             'QQ号': r'(?:QQ[：:]?|qq[：:]?)\s*([1-9]\d{4,10})',  # 更精确的QQ号识别
             '邮箱_enhanced': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
         }
        
        self.value_extractors.update(extended_extractors)
    
    def _build_field_patterns(self) -> Dict[str, List[str]]:
        """构建字段识别模式"""
        return {
            '姓名': ['姓名', 'name', '名字', '真实姓名', '本名'],
            '性别': ['性别', 'gender', 'sex'],
            '年龄': ['年龄', 'age', '岁'],
            '出生日期': ['出生日期', '出生年月', '生日', 'birthday', 'birth_date', '出生时间'],
            '身份证号': ['身份证号', '身份证号码', '身份证', 'id_number', '证件号', '身份证件号'],
            '电话': ['电话', '手机', '手机号', '联系电话', 'phone', 'mobile', 'tel', '电话号码'],
            '邮箱': ['邮箱', '邮件', 'email', '电子邮箱', '邮箱地址'],
            '地址': ['地址', '住址', 'address', '居住地', '家庭地址', '通讯地址'],
            '职业': ['职业', '工作', '职位', 'job', 'occupation', '岗位', '从事'],
            '公司': ['公司', '工作单位', 'company', '单位', '企业'],
            '户籍地': ['户籍地', '籍贯', '出生地', '户口所在地'],
            '户籍地址': ['户籍地址', '真户籍地址', '户口地址', '户籍详细地址'],
            '学历': ['学历', '教育', 'education', '毕业于', '就读于'],
            '星座': ['星座', 'constellation'],
            '生肖': ['生肖', '属相', 'zodiac'],
            '民族': ['民族', 'ethnicity', '族'],
            '婚姻状况': ['婚姻状况', '婚姻', '已婚', '未婚', '离异'],
            '血型': ['血型', 'blood_type'],
            '身高': ['身高', 'height', '高'],
            '体重': ['体重', 'weight', '重'],
            '备注': ['备注', '其他', '说明', 'note', 'remark', '补充'],
        }
    
    def _build_value_extractors(self) -> Dict[str, str]:
        """构建值提取模式"""
        return {
            '身份证号': r'\b(\d{15}|\d{17}[\dXx]|\d{18})\b',
            '手机号': r'\b(1[3-9]\d{9})\b',
            '座机号': r'\b(0\d{2,3}[-\s]?\d{7,8})\b',
            '邮箱': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '日期': r'(\d{4})[年\-/\.]\s*(\d{1,2})[月\-/\.]\s*(\d{1,2})[日]?',
            '年份': r'(\d{4})年',
            '性别': r'[性别]?\s*([男女])',
            '生肖': r'([鼠牛虎兔龙蛇马羊猴鸡狗猪])',
            '星座': r'(白羊座|金牛座|双子座|巨蟹座|狮子座|处女座|天秤座|天蝎座|射手座|摩羯座|水瓶座|双鱼座)',
            '农历年': r'([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]年)',
            '农历月日': r'((?:正|二|三|四|五|六|七|八|九|十|冬|腊)月\s*(?:初|十|廿|卅)?[一二三四五六七八九十]{1,3})',
        }
    
    def parse_text_intelligent(self, text: str) -> Dict[str, Any]:
        """智能解析文本"""
        logger.info("开始智能文本解析...")
        
        # 1. 预处理文本
        cleaned_text = self._preprocess_text(text)
        
        # 2. 分离主人物和同户人信息
        main_section, household_section = self._split_main_and_household(cleaned_text)
        
        # 3. 提取主人物字段
        main_fields = self._extract_all_fields(main_section)
        
        # 4. 提取同户人信息
        household_members = []
        if household_section:
            household_members = self._parse_household_members(household_section)
        
        # 5. 字段后处理和优化
        optimized_fields = self._post_process_fields(main_fields)
        
        result = {
            'main_person': optimized_fields,
            'household_members': household_members,
            'raw_text': text,
            'parsing_method': 'smart_jieba',
            'confidence_score': self._calculate_confidence(optimized_fields, household_members)
        }
        
        logger.info(f"解析完成，提取到 {len(optimized_fields)} 个字段，{len(household_members)} 个同户人")
        return result
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 标准化冒号
        text = text.replace('：', ':')
        # 移除特殊符号前后的空格，但保留换行符
        text = re.sub(r'[ \t]*([：:，。；;])[ \t]*', r'\1', text)
        # 移除行内多余空格，但保留换行符
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # 只处理行内空格，不影响换行符
            cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
            if cleaned_line:  # 只保留非空行
                cleaned_lines.append(cleaned_line)
        return '\n'.join(cleaned_lines)
    
    def _split_main_and_household(self, text: str) -> Tuple[str, str]:
        """分离主人物和同户人信息"""
        lines = text.split('\n')
        main_lines = []
        household_lines = []
        in_household_section = False
        first_person_processed = False
        
        # 同户人关键词
        household_keywords = ['同户人', '同户', '同住人', '家庭成员', '户主', '家属']
        
        # 分隔符模式
        separator_patterns = [
            r'-{3,}',      # ---
            r'={3,}',      # ===
            r'━{3,}',      # ━━━
            r'─{3,}',      # ───
            r'__{3,}',     # ___
        ]
        
        # 智能分离策略：只有明确标识的同户人才分离，其余都是主人物
        in_household_mode = False
        temp_household_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测同户人关键词 - 明确进入同户人区域
            if any(keyword in line for keyword in household_keywords):
                in_household_mode = True
                temp_household_lines.append(line)
                continue
            
            # 检测分隔符 - 可能是同户人分隔符
            is_separator = any(re.search(pattern, line) for pattern in separator_patterns)
            if is_separator:
                temp_household_lines.append(line)
                continue
            
            # 如果在同户人模式下
            if in_household_mode:
                # 检查这行是否是新的主人物信息（重复的姓名信息）
                if line.startswith('姓名:'):
                    # 提交之前收集的同户人信息到正式列表
                    household_lines.extend(temp_household_lines)
                    temp_household_lines = []  # 清空临时列表
                    
                    # 退出同户人模式，开始处理新的主人物信息
                    in_household_mode = False
                    main_lines.append(line)
                else:
                    # 继续收集同户人信息
                    temp_household_lines.append(line)
            else:
                # 主人物信息
                main_lines.append(line)
        
        # 将剩余的临时同户人信息添加到正式列表
        household_lines.extend(temp_household_lines)
        
        return '\n'.join(main_lines), '\n'.join(household_lines)
    
    def _extract_all_fields(self, text: str) -> Dict[str, Any]:
        """提取所有字段"""
        fields = {}
        
        # 1. 使用jieba分词和词性标注
        words = list(pseg.cut(text))
        
        # 2. 基于模式的键值对提取
        kv_fields = self._extract_key_value_pairs(text)
        self._merge_field_dicts(fields, kv_fields)
        
        # 3. 基于正则的特殊值提取
        regex_fields = self._extract_with_regex(text)
        self._merge_field_dicts(fields, regex_fields)
        
        # 4. 基于上下文的智能提取
        context_fields = self._extract_with_context(text, words)
        self._merge_field_dicts(fields, context_fields)
        
        # 5. 去重和标准化
        normalized_fields = self._normalize_fields(fields)
        
        return normalized_fields
    
    def _merge_field_dicts(self, target: Dict[str, Any], source: Dict[str, Any]):
        """智能合并字段字典"""
        for key, value in source.items():
            if key in target:
                # 使用已有的合并逻辑
                target[key] = self._merge_values(target[key], value)
            else:
                target[key] = value
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """提取键值对"""
        fields = {}
        
        # 多种键值对模式
        patterns = [
            r'([^：:\n\r]{1,20})\s*[:：]\s*([^，\n\r；;]{1,200})',  # 基本模式（扩大值长度）
            r'([^：:\n\r]{1,20})\s*[：:]\s*([^，\n\r；;]{1,200})',  # 变体模式
            r'([^：:\n\r]{1,20})\s*为\s*([^，\n\r；;]{1,100})',     # "为"连接
            r'([^：:\n\r]{1,20})\s*是\s*([^，\n\r；;]{1,100})',     # "是"连接
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for raw_key, raw_value in matches:
                key = raw_key.strip()
                value = raw_value.strip()
                
                # 过滤无效键值对
                if (len(key) >= 2 and len(value) >= 1 and 
                    value not in ['', '-', '无', '空', '未知', 'null', 'N/A']):
                    
                    # 标准化字段名
                    standard_key = self._standardize_field_name(key)
                    
                    # 🔥 处理多值字段（冒号分隔、分号分隔）
                    processed_value = self._process_multi_value_field(standard_key, value)
                    fields[standard_key] = processed_value
        
        return fields
    
    def _process_multi_value_field(self, field_name: str, value: str) -> Any:
        """处理多值字段"""
        # 电话字段：支持冒号分隔
        if field_name in ['电话', '手机', '手机号', 'phone', 'mobile']:
            if ':' in value:
                phones = [p.strip() for p in value.split(':') if p.strip()]
                # 验证每个电话号码
                valid_phones = []
                for phone in phones:
                    clean_phone = re.sub(r'[\s\-\(\)]', '', phone)
                    if re.match(r'^1[3-9]\d{9}$', clean_phone):
                        valid_phones.append(clean_phone)
                    elif len(clean_phone) >= 7:  # 可能是座机或其他格式
                        valid_phones.append(phone)
                return valid_phones if len(valid_phones) > 1 else (valid_phones[0] if valid_phones else value)
        
        # 地址字段：支持分号分隔
        elif field_name in ['地址', '快递地址', '收货地址', 'address']:
            if ';' in value:
                addresses = [addr.strip() for addr in value.split(';') if addr.strip()]
                return addresses if len(addresses) > 1 else (addresses[0] if addresses else value)
        
        # URL字段：识别链接
        elif field_name in ['微博链接', '链接', 'url', 'link']:
            url_pattern = r'https?://[\w\.\-/\?=&%]+|www\.[\w\.\-/\?=&%]+'
            urls = re.findall(url_pattern, value)
            return urls[0] if urls else value
        
        # 其他字段保持原值
        return value
    
    def _extract_with_regex(self, text: str) -> Dict[str, Any]:
        """使用正则表达式提取特殊字段"""
        fields = {}
        
        for field_type, pattern in self.value_extractors.items():
            matches = re.findall(pattern, text)
            if matches:
                if field_type == '日期':
                    # 日期特殊处理
                    for match in matches:
                        if isinstance(match, tuple):
                            year, month, day = match
                            date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            fields['出生日期'] = date_str
                            break
                elif field_type in ['身份证号', '手机号', '邮箱', 'URL', '车牌号', 'QQ号', '邮箱_enhanced']:
                    # 多值字段
                    if field_type == '身份证号':
                        fields['身份证号'] = list(set(matches))  # 去重
                    elif field_type == '手机号':
                        fields['电话'] = list(set(matches))
                    elif field_type in ['邮箱', '邮箱_enhanced']:
                        fields['邮箱'] = list(set(matches))
                    elif field_type == 'URL':
                        fields['链接'] = list(set(matches))
                    elif field_type == '车牌号':
                        fields['车牌'] = list(set(matches))
                    elif field_type == 'QQ号':
                        fields['QQ'] = list(set(matches))
                else:
                    # 单值字段
                    field_map = {
                        '性别': '性别',
                        '生肖': '生肖', 
                        '星座': '星座',
                        '农历年': '农历年份',
                        '农历月日': '农历日期'
                    }
                    if field_type in field_map:
                        fields[field_map[field_type]] = matches[0]
        
        return fields
    
    def _extract_with_context(self, text: str, words: List) -> Dict[str, Any]:
        """基于上下文的智能提取"""
        fields = {}
        
        # 安全处理分词结果
        try:
            # 将分词结果转换为文本窗口
            word_list = []
            word_flag_list = []
            
            for item in words:
                if hasattr(item, 'word') and hasattr(item, 'flag'):
                    # jieba.posseg返回的对象
                    word_list.append(item.word)
                    word_flag_list.append((item.word, item.flag))
                elif isinstance(item, tuple) and len(item) == 2:
                    # 元组格式
                    word_list.append(item[0])
                    word_flag_list.append(item)
                else:
                    # 字符串格式
                    word_list.append(str(item))
                    word_flag_list.append((str(item), 'n'))
            
            # 查找可能的字段指示词
            for i, (word, flag) in enumerate(word_flag_list):
                # 寻找可能的字段名
                standard_field = self._find_field_in_context(word, i, word_list)
                if standard_field:
                    # 提取该字段的值
                    value = self._extract_value_for_field(standard_field, i, word_list, text)
                    if value:
                        fields[standard_field] = value
        
        except Exception as e:
            logger.warning(f"上下文提取失败: {e}")
        
        return fields
    
    def _find_field_in_context(self, word: str, position: int, word_list: List[str]) -> Optional[str]:
        """在上下文中查找字段名"""
        # 检查当前词是否是字段名
        for field_name, patterns in self.field_patterns.items():
            if word in patterns:
                return field_name
            
            # 模糊匹配
            for pattern in patterns:
                if difflib.SequenceMatcher(None, word, pattern).ratio() > 0.8:
                    return field_name
        
        # 检查组合词（当前词+下一个词）
        if position < len(word_list) - 1:
            combined = word + word_list[position + 1]
            for field_name, patterns in self.field_patterns.items():
                if combined in patterns:
                    return field_name
        
        return None
    
    def _extract_value_for_field(self, field_name: str, position: int, word_list: List[str], full_text: str) -> Optional[str]:
        """为特定字段提取值"""
        # 在词位置附近查找值
        context_window = 3  # 前后3个词的窗口
        
        start = max(0, position - context_window)
        end = min(len(word_list), position + context_window + 1)
        
        context_words = word_list[start:end]
        context_text = ' '.join(context_words)
        
        # 根据字段类型使用不同策略
        if field_name in ['身份证号', '电话', '邮箱']:
            # 使用正则提取
            if field_name == '身份证号':
                matches = re.findall(self.value_extractors['身份证号'], context_text)
            elif field_name == '电话':
                matches = re.findall(self.value_extractors['手机号'], context_text)
            elif field_name == '邮箱':
                matches = re.findall(self.value_extractors['邮箱'], context_text)
            
            return matches[0] if matches else None
        
        # 对于其他字段，查找冒号后的内容
        colon_pattern = rf'{re.escape(field_name)}\s*[:：]\s*([^，\n\r；;]+)'
        match = re.search(colon_pattern, full_text)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _standardize_field_name(self, raw_field: str) -> str:
        """标准化字段名"""
        raw_field = raw_field.strip()
        
        # 精确匹配
        for standard_name, patterns in self.field_patterns.items():
            if raw_field in patterns:
                return standard_name
        
        # 模糊匹配
        best_match = None
        best_score = 0
        
        for standard_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                score = difflib.SequenceMatcher(None, raw_field.lower(), pattern.lower()).ratio()
                if score > best_score and score > 0.8:
                    best_score = score
                    best_match = standard_name
        
        return best_match if best_match else raw_field
    
    def _normalize_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """字段标准化和去重"""
        normalized = {}
        
        for key, value in fields.items():
            # 合并相似字段
            merged_key = self._find_similar_key(key, list(normalized.keys()))
            final_key = merged_key if merged_key else key
            
            # 值处理
            if final_key in normalized:
                # 合并值
                existing = normalized[final_key]
                merged_value = self._merge_values(existing, value)
                normalized[final_key] = merged_value
            else:
                normalized[final_key] = value
        
        return normalized
    
    def _find_similar_key(self, key: str, existing_keys: List[str], threshold: float = 0.85) -> Optional[str]:
        """查找相似的已存在键"""
        for existing_key in existing_keys:
            similarity = difflib.SequenceMatcher(None, key.lower(), existing_key.lower()).ratio()
            if similarity > threshold:
                return existing_key
        return None
    
    def _merge_values(self, existing: Any, new: Any) -> Any:
        """合并字段值"""
        if existing == new:
            return existing
        
        # 特殊处理电话字段的冒号分隔值
        if isinstance(new, str) and ':' in new:
            new_phones = [p.strip() for p in new.split(':') if p.strip() and len(p.strip()) >= 7]
            if new_phones:
                new = new_phones
        
        # 转换为列表进行合并
        if not isinstance(existing, list):
            existing = [existing] if existing else []
        if not isinstance(new, list):
            new = [new] if new else []
        
        # 合并并去重
        merged = existing.copy()
        for item in new:
            if item and str(item).strip() and item not in merged:
                merged.append(item)
        
        return merged[0] if len(merged) == 1 else merged
    
    def _parse_household_members(self, household_text: str) -> List[Dict[str, Any]]:
        """解析同户人信息"""
        members = []
        
        # 按分隔符分割成员
        sections = re.split(r'-{3,}|={3,}|\u2501{3,}|\u2014{3,}', household_text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # 检查是否包含同户人关键词但仍有实际内容
            has_household_keyword = any(keyword in section for keyword in ['同户人', '同户', '家庭成员'])
            if has_household_keyword:
                # 处理包含同户人关键词的行，提取实际信息
                lines = section.split('\n')
                content_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 检查是否是 "同户人:姓名,性别,身份证,日期" 格式
                    if any(keyword in line for keyword in ['同户人', '同户', '家庭成员']):
                        # 提取冒号后的内容
                        colon_pos = line.find(':')
                        if colon_pos != -1:
                            member_info = line[colon_pos + 1:].strip()
                            if member_info:
                                content_lines.append(member_info)
                    else:
                        content_lines.append(line)
                
                if content_lines:
                    section = '\n'.join(content_lines)
                else:
                    continue
            
            # 为每个成员提取字段
            member_fields = self._extract_all_fields(section)
            
            # 如果没有提取到姓名，尝试解析逗号分隔格式
            if not member_fields.get('姓名'):
                member_fields = self._parse_comma_separated_member(section, member_fields)
            
            # 验证是否为有效成员（至少有姓名或身份证）
            if (member_fields.get('姓名') or 
                member_fields.get('身份证号') or
                any('姓名' in str(k) for k in member_fields.keys())):
                members.append(member_fields)
        
        return members
    
    def _parse_comma_separated_member(self, section: str, existing_fields: Dict[str, Any]) -> Dict[str, Any]:
        """解析逗号分隔的成员信息格式，如：邓青松,男,512922197305011634,1973年05月01日"""
        member_fields = existing_fields.copy()
        
        # 查找可能的逗号分隔行
        for line in section.split('\n'):
            line = line.strip()
            if not line or ':' in line:  # 跳过键值对格式的行
                continue
                
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:  # 至少有姓名、性别、身份证
                # 尝试识别各个字段
                potential_name = parts[0]
                potential_gender = parts[1] if len(parts) > 1 else None
                potential_id = parts[2] if len(parts) > 2 else None
                potential_birth = parts[3] if len(parts) > 3 else None
                
                # 验证身份证格式来确定这是有效的成员信息
                if potential_id and re.match(r'^[0-9]{15}([0-9]{2}[0-9Xx])?$', potential_id):
                    # 提取姓名（如果还没有）
                    if not member_fields.get('姓名') and potential_name and len(potential_name) <= 10:
                        member_fields['姓名'] = potential_name
                    
                    # 提取性别（如果还没有）
                    if not member_fields.get('性别') and potential_gender in ['男', '女']:
                        member_fields['性别'] = potential_gender
                    
                    # 身份证号已经通过常规方法提取了，不需要重复
                    
                    # 出生日期已经通过常规方法提取了，不需要重复
                    break
        
        return member_fields
    
    def _post_process_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """字段后处理"""
        processed = fields.copy()
        
        # 数据类型转换和验证
        if '出生日期' in processed:
            processed['出生日期'] = self._normalize_date(processed['出生日期'])
        
        if '电话' in processed:
            processed['电话'] = self._normalize_phones(processed['电话'])
        
        if '身份证号' in processed:
            processed['身份证号'] = self._normalize_id_numbers(processed['身份证号'])
        
        # 移除空值和无效值（保留非空列表）
        def is_valid_value(value):
            if isinstance(value, list):
                return len(value) > 0 and any(str(item).strip() for item in value)
            return (value is not None and 
                   str(value).strip() not in ['', '-', '无', '空', '未知', 'null', 'N/A'])
        
        processed = {k: v for k, v in processed.items() if is_valid_value(v)}
        
        return processed
    
    def _normalize_date(self, date_value: Any) -> str:
        """标准化日期"""
        if isinstance(date_value, list):
            date_value = date_value[0]
        
        date_str = str(date_value).strip()
        
        # 尝试多种日期格式
        patterns = [
            r'(\d{4})[年\-/\.]\s*(\d{1,2})[月\-/\.]\s*(\d{1,2})[日]?',
            r'(\d{4})\s*(\d{1,2})\s*(\d{1,2})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                year, month, day = match.groups()
                try:
                    # 验证日期
                    datetime(int(year), int(month), int(day))
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except ValueError:
                    continue
        
        return date_str
    
    def _normalize_phones(self, phone_value: Any) -> List[str]:
        """标准化电话号码"""
        if not isinstance(phone_value, list):
            phone_value = [phone_value]
        
        normalized = []
        for phone in phone_value:
            phone_str = str(phone).strip()
            # 清理格式
            clean_phone = re.sub(r'[\s\-\(\)]', '', phone_str)
            # 验证手机号格式
            if re.match(r'^1[3-9]\d{9}$', clean_phone):
                normalized.append(clean_phone)
            else:
                normalized.append(phone_str)  # 保留原格式
        
        return list(set(normalized))  # 去重
    
    def _normalize_id_numbers(self, id_value: Any) -> List[str]:
        """标准化身份证号"""
        if not isinstance(id_value, list):
            id_value = [id_value]
        
        normalized = []
        for id_num in id_value:
            id_str = str(id_num).strip().upper()
            # 基本格式验证
            if re.match(r'^\d{15}$|^\d{17}[\dX]$|^\d{18}$', id_str):
                normalized.append(id_str)
        
        return list(set(normalized))  # 去重
    
    def _calculate_confidence(self, main_fields: Dict[str, Any], household_members: List[Dict[str, Any]]) -> float:
        """计算解析置信度"""
        score = 0.5  # 基础分数
        
        # 主要字段加分
        key_fields = ['姓名', '性别', '出生日期', '身份证号', '电话']
        for field in key_fields:
            if field in main_fields:
                score += 0.1
        
        # 字段数量加分
        score += min(len(main_fields) * 0.02, 0.2)
        
        # 同户人信息加分
        score += min(len(household_members) * 0.02, 0.1)
        
        # 数据质量加分
        if '身份证号' in main_fields:
            id_nums = main_fields['身份证号']
            if isinstance(id_nums, list) and any(re.match(r'^\d{17}[\dX]$', str(num)) for num in id_nums):
                score += 0.1
        
        return min(score, 1.0)


# 创建全局解析器实例
smart_parser = SmartTextParser() 