"""
自由文本人物信息导入服务
- 从用户输入的文本中提取人物关键信息
- 将人物档案以JSON形式存储在本地(data/persons.json)
"""

import os
import re
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.models.person import PersonProfile
from app.core.logger import logger, log_audit

try:
    import yaml
except Exception:
    yaml = None

DATA_DIR = "data"
PERSONS_FILE = os.path.join(DATA_DIR, "persons.json")

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 更严格的手机号/座机匹配：
MOBILE_REGEX = re.compile(r"(?:\+?86[- ]?)?(1[3-9]\d{9})(?!\d)")
LANDLINE_REGEX = re.compile(r"(?:\+?\d{1,3}[- ]?)?(0\d{2,3})[- ]?(\d{7,8})(?!\d)")
NAME_LABEL_REGEX = re.compile(r"(?:姓名|Name)\s*[:：]\s*([^\n，。;；]+)")
ADDRESS_LABEL_REGEX = re.compile(r"(?:地址|住址|居住地|居住地址|户籍地址|快递地址|Address)\s*[:：]\s*([^\n;；]+)")
COMPANY_REGEX = re.compile(r"(?:公司|任职于|在)\s*([^\n，。;；]+?公司)")
POSITION_REGEX = re.compile(r"(?:职位|职务|岗位|担任)\s*[:：]?\s*([^\n，。;；]+)")
EDU_REGEX = re.compile(r"(?:毕业于|就读于|学习于|学历|Education)\s*[:：]?\s*([^\n]+)")
SKILLS_REGEX = re.compile(r"(?:技能|Skills?)\s*[:：]\s*([^\n]+)")
GENDER_REGEX = re.compile(r"(?:性别|Gender)\s*[:：]\s*([^\n，。;；]+)")
BIRTH_REGEX = re.compile(r"(?:出生(?:年月|日期|时间)?|生日|Birth(?:day| date)?)\s*[:：]\s*([^\n，。;；]+)")
ID_REGEX = re.compile(r"\b(\d{6})(\d{8})(\d{3}[0-9Xx])\b")  # 粗略匹配18位身份证
ID15_REGEX = re.compile(r"\b\d{15}\b")
LICENSE_PLATE_REGEX = re.compile(r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5}[A-Z0-9挂学警港澳]?")
HUKOU_PLACE_REGEX = re.compile(r"(?:出生地|籍贯|户籍地|户口所在地)\s*[:：]\s*([^\n;；]+)")
HUKOU_ADDR_STRICT_REGEX = re.compile(r"(?:真户籍地址|户籍地地址|户口地址)\s*[:：]\s*([^\n;；]+)")
HOUSEHOLD_LABELS = ["同户人", "同户", "同住人", "家庭成员", "户主", "household", "household_members"]

SOCIAL_KEYS = {
    "微信": "wechat",
    "weixin": "wechat",
    "wechat": "wechat",
    "qq": "qq",
    "QQ": "qq",
    "微博": "weibo",
    "抖音": "douyin",
    "telegram": "telegram",
    "tg": "telegram",
    "twitter": "twitter",
    "推特": "twitter",
    "github": "github",
    "gitee": "gitee",
    "领英": "linkedin",
    "linkedin": "linkedin"
}

SOCIAL_WEIBO_REGEX = re.compile(r"https?://[\w\.]*weibo\.cn/[\w/\-?=&%]+", re.IGNORECASE)


def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PERSONS_FILE):
        with open(PERSONS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def load_all_persons() -> List[Dict[str, Any]]:
    ensure_storage()
    try:
        if not os.path.exists(PERSONS_FILE) or os.path.getsize(PERSONS_FILE) == 0:
            return []
        with open(PERSONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取persons.json失败，将使用空列表: {e}")
        return []


def save_all_persons(persons: List[Dict[str, Any]]):
    with open(PERSONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(persons, f, ensure_ascii=False, indent=2)


def parse_free_text_to_profile(text: str) -> PersonProfile:
	"""从自由文本中解析人物信息，使用AI增强的智能解析"""
	text = (text or '').strip()
	if not text:
		raise ValueError("空文本无法解析")

	# 🚀 使用智能解析器（优先使用jieba版本，备选AI版本）
	try:
		# 优先使用基于jieba的智能解析器
		from app.services.smart_text_parser import smart_parser
		smart_result = smart_parser.parse_text_intelligent(text)
		
		
		# 转换解析结果为PersonProfile格式
		profile = _convert_smart_result_to_profile(smart_result, text)
		return profile
		
	except ImportError:
		logger.warning("jieba库未安装，尝试使用AI解析器")
		try:
			from app.services.ai_text_parser import ai_parser
			ai_result = ai_parser.parse_text_intelligent(text)
			profile = _convert_ai_result_to_profile(ai_result, text)
			return profile
		except Exception as e:
			logger.warning(f"AI解析失败，回退到传统方法: {e}")
			return _parse_with_traditional_method(text)
		
	except Exception as e:
		logger.warning(f"智能解析失败，回退到传统方法: {e}")
		# 回退到传统解析方法
		return _parse_with_traditional_method(text)


def _convert_smart_result_to_profile(smart_result: Dict[str, Any], original_text: str) -> PersonProfile:
	"""将智能解析结果转换为PersonProfile对象"""
	# 智能解析器和AI解析器的结果格式相同，可以复用转换逻辑
	return _convert_ai_result_to_profile(smart_result, original_text)


def _convert_ai_result_to_profile(ai_result: Dict[str, Any], original_text: str) -> PersonProfile:
	"""将AI解析结果转换为PersonProfile对象"""
	main_fields = ai_result.get('main_person', {})
	household_members = ai_result.get('household_members', [])
	
	# 提取基本字段
	name = main_fields.get('姓名') or main_fields.get('name', '未知姓名')
	gender = main_fields.get('性别') or main_fields.get('gender')
	birth_date = None
	
	# 日期处理
	birth_info = (main_fields.get('出生日期') or main_fields.get('出生年月') or 
	             main_fields.get('birth_date') or main_fields.get('生日'))
	if birth_info:
		birth_date = try_parse_date(str(birth_info))
	
	# 联系方式处理
	emails = []
	phones = []
	
	# 电话处理
	phone_fields = ['电话', '手机', '手机号', 'phone', 'mobile']
	for field in phone_fields:
		if field in main_fields:
			phone_value = main_fields[field]
			if isinstance(phone_value, list):
				phones.extend([str(p) for p in phone_value])
			else:
				phones.append(str(phone_value))
	
	# 邮箱处理
	email_fields = ['邮箱', 'email', '邮件']
	for field in email_fields:
		if field in main_fields:
			email_value = main_fields[field]
			if isinstance(email_value, list):
				emails.extend([str(e) for e in email_value])
			else:
				emails.append(str(email_value))
	
	# 地址处理
	addresses = []
	address_fields = ['地址', '住址', 'address', '居住地']
	for field in address_fields:
		if field in main_fields:
			addr_value = main_fields[field]
			if isinstance(addr_value, list):
				addresses.extend([str(a) for a in addr_value])
			else:
				addresses.append(str(addr_value))
	
	# 身份证处理
	id_numbers = []
	id_fields = ['身份证号', '身份证', 'id_number', '证件号']
	for field in id_fields:
		if field in main_fields:
			id_value = main_fields[field]
			if isinstance(id_value, list):
				id_numbers.extend([str(i) for i in id_value])
			else:
				id_numbers.append(str(id_value))
	
	# 职业信息
	current_job = (main_fields.get('职业') or main_fields.get('职位') or 
	              main_fields.get('工作') or main_fields.get('job'))
	current_company = (main_fields.get('公司') or main_fields.get('工作单位') or 
	                  main_fields.get('company'))
	
	# 户籍信息
	hukou_place = (main_fields.get('户籍地') or main_fields.get('籍贯') or 
	              main_fields.get('出生地'))
	hukou_address = (main_fields.get('户籍地址') or main_fields.get('真户籍地址') or 
	                main_fields.get('户口地址'))
	
	# 构建自定义属性（排除已处理的标准字段）
	standard_fields = {
		'姓名', 'name', '性别', 'gender', '出生日期', '出生年月', 'birth_date', '生日',
		'电话', '手机', '手机号', 'phone', 'mobile', '邮箱', 'email', '邮件',
		'地址', '住址', 'address', '居住地', '身份证号', '身份证', 'id_number', '证件号',
		'职业', '职位', '工作', 'job', '公司', '工作单位', 'company',
		'户籍地', '籍贯', '出生地', '户籍地址', '真户籍地址', '户口地址'
	}
	
	custom_attrs = {k: v for k, v in main_fields.items() if k not in standard_fields}
	
	# 构建关系数据（同户人）
	relationships = []
	for member in household_members:
		member_name = member.get('姓名') or member.get('name')
		if member_name and member_name != name:
			relationships.append({
				"type": "household_member",
				"name": member_name,
				"relationship": "同户人",
				"details": member,
				"source": "ai_import"
			})
	
	# 创建PersonProfile对象
	profile = PersonProfile(
		id=str(uuid.uuid4()),
		name=str(name),
		email=emails[0] if emails else None,
		phone=phones[0] if phones else None,
		phones=phones,
		address=addresses[0] if addresses else None,
		delivery_addresses=addresses[1:] if len(addresses) > 1 else [],
		current_company=str(current_company) if current_company else None,
		current_job=str(current_job) if current_job else None,
		gender=str(gender) if gender else None,
		birth_date=birth_date,
		education=[],
		skills=[],
		social_profiles={},
		hukou_place=str(hukou_place) if hukou_place else None,
		hukou_address=str(hukou_address) if hukou_address else None,
		custom_attributes=custom_attrs,
		raw_text=original_text,
		id_numbers=id_numbers,
		relationships=relationships,
		data_sources=[{
			"type": "ai_import", 
			"timestamp": datetime.now().isoformat(),
			"confidence": ai_result.get('confidence_score', 0.8),
			"method": ai_result.get('parsing_method', 'ai')
		}],
		confidence_score=ai_result.get('confidence_score', 0.8),
		created_at=datetime.now(),
		updated_at=datetime.now(),
		verified=False
	)
	
	return profile


def _parse_with_traditional_method(text: str) -> PersonProfile:
	"""传统解析方法作为备选方案"""
	# 预解析：若文本是JSON或YAML，优先按结构化载入
	structured: Dict[str, Any] = {}
	if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
		try:
			structured = json.loads(text)
		except Exception:
			structured = {}
	elif yaml and any(k in text.lower() for k in [':', '：']) and '\n' in text:
		try:
			loaded = yaml.safe_load(text)
			structured = loaded if isinstance(loaded, dict) else {}
		except Exception:
			structured = {}

	# 🔍 智能分析文本结构，识别主人物和同户人信息
	main_person_section, household_members_info = _split_main_and_household_info(text)
	
	# 从主人物段落解析信息
	name = _extract(NAME_LABEL_REGEX, main_person_section) or (structured.get('name') if isinstance(structured, dict) else None)
	
	# 如果主段落没有姓名，尝试从完整文本第一个姓名行获取
	if not name:
		first_name_match = NAME_LABEL_REGEX.search(text)
		if first_name_match:
			name = first_name_match.group(1).strip()

	# 邮箱（只从主人物段落提取）
	emails = extract_emails(main_person_section)
	if isinstance(structured, dict):
		for k in ['email', 'emails', '邮箱']:
			v = structured.get(k)
			if isinstance(v, str):
				emails.extend(extract_emails(v))
			elif isinstance(v, list):
				for e in v:
					if isinstance(e, str):
						emails.extend(extract_emails(e))
	emails = deduplicate_preserve_order(emails)

	# 更智能的电话提取（只从主人物段落提取，避免把同户人电话当成主人电话）
	phone_list = extract_phones(main_person_section)
	if isinstance(structured, dict):
		for k in ['phone', 'tel', 'mobile', '手机号', '电话', '联系电话']:
			v = structured.get(k)
			if isinstance(v, str):
				phone_list.extend(extract_phones(v))
			elif isinstance(v, list):
				for item in v:
					if isinstance(item, str):
						phone_list.extend(extract_phones(item))
	phone_list = deduplicate_preserve_order(phone_list)

	# 地址（只从主人物段落提取）
	addresses = extract_addresses(main_person_section)
	if isinstance(structured, dict):
		for k in ['address', 'addresses', '地址', '住址', '居住地', '居住地址', '户籍地址', '快递地址']:
			v = structured.get(k)
			if isinstance(v, str):
				addresses.append(v)
			elif isinstance(v, list):
				for a in v:
					if isinstance(a, str):
						addresses.append(a)
	addresses = [a.strip() for a in addresses if a and a.strip()]
	addresses = deduplicate_preserve_order(addresses)

	current_company = _extract(COMPANY_REGEX, main_person_section) or (structured.get('company') or structured.get('current_company') if isinstance(structured, dict) else None)
	current_job = _extract(POSITION_REGEX, main_person_section) or (structured.get('position') or structured.get('current_job') if isinstance(structured, dict) else None)
	
	# 性别识别 - 增强识别逻辑
	gender = _extract(GENDER_REGEX, main_person_section) or (structured.get('gender') if isinstance(structured, dict) else None)
	if not gender:
		# 从"其他"字段或文本中直接搜索
		if '女' in main_person_section and '男' not in main_person_section:
			gender = '女'
		elif '男' in main_person_section and '女' not in main_person_section:
			gender = '男'

	# 户籍地/户籍地址（可能在主段落或全文）
	hukou_place = _extract(HUKOU_PLACE_REGEX, main_person_section) or _extract(HUKOU_PLACE_REGEX, text) or (structured.get('hukou_place') if isinstance(structured, dict) else None)
	hukou_address = _extract(HUKOU_ADDR_STRICT_REGEX, main_person_section) or _extract(HUKOU_ADDR_STRICT_REGEX, text) or (structured.get('hukou_address') if isinstance(structured, dict) else None)

	# 身份证与车牌（只从主人物段落提取）
	id_numbers = extract_id_numbers(main_person_section)
	plates = extract_license_plates(main_person_section)

	# 出生日期（只从主人物段落提取）
	birth_info = _extract(BIRTH_REGEX, main_person_section)
	birth_date = try_parse_date(birth_info) if birth_info else None

	# 教育（只从主人物段落提取）
	education_entries: List[Dict[str, Any]] = []
	edu_line = _extract(EDU_REGEX, main_person_section)
	if edu_line:
		education_entries.append({"text": edu_line})
	if isinstance(structured, dict) and structured.get('education'):
		if isinstance(structured['education'], list):
			education_entries.extend(structured['education'])
		elif isinstance(structured['education'], str):
			education_entries.append({"text": structured['education']})

	# 技能（只从主人物段落提取）
	skills: List[str] = []
	skills_line = _extract(SKILLS_REGEX, main_person_section)
	if skills_line:
		for token in re.split(r"[、,，/|\\]\s*", skills_line):
			if token.strip():
				skills.append(token.strip())
	if isinstance(structured, dict) and structured.get('skills'):
		if isinstance(structured['skills'], list):
			skills.extend([str(s) for s in structured['skills']])
		elif isinstance(structured['skills'], str):
			skills.extend([s.strip() for s in re.split(r"[、,，/|\\]\s*", structured['skills']) if s.strip()])
	skills = deduplicate_preserve_order(skills)

	# 识别社交账号（只从主人物段落提取）
	social_profiles: Dict[str, str] = {}
	if isinstance(structured, dict):
		for raw_key, std_key in SOCIAL_KEYS.items():
			# 优先结构化里对应键
			if raw_key in structured and isinstance(structured[raw_key], str):
				social_profiles[std_key] = str(structured[raw_key]).strip()
			elif std_key in structured and isinstance(structured[std_key], str):
				social_profiles[std_key] = str(structured[std_key]).strip()
	# weibo链接直接识别（从主段落）
	m = SOCIAL_WEIBO_REGEX.search(main_person_section)
	if m:
		social_profiles['weibo'] = m.group(0)

	# 🏠 处理同户人信息
	household_members = []
	if household_members_info:
		try:
			household_members = _parse_household_members(household_members_info)
		except Exception as e:
			logger.warning(f"解析同户人信息失败: {e}")

	# 🔍 智能提取自定义字段（从主人物段落）
	custom_attrs = _extract_dynamic_fields(main_person_section)

	# 主字段：选一个主邮箱/电话/地址
	primary_email = emails[0] if emails else None
	primary_phone = phone_list[0] if phone_list else None
	primary_address = addresses[0] if addresses else None

	# 其他多值字段
	delivery_addresses = [a for a in addresses if a != primary_address]

	# 构建关系数据（同户人关系）
	relationships = []
	for member in household_members:
		member_name = member.get('name', '').strip()
		# 过滤主人物本人和无效姓名
		if (member_name 
			and member_name != name 
			and member_name not in ['其他', '姓名', '身份证号码', '出生年月']
			and len(member_name) >= 2):
			relationships.append({
				"type": "household_member",
				"name": member_name,
				"relationship": member.get('relationship', '同户人'),
				"details": member,
				"source": "text_import"
			})

	profile = PersonProfile(
		id=str(uuid.uuid4()),
		name=name or "未知姓名",
		email=primary_email,
		phone=primary_phone,
		phones=phone_list,
		address=primary_address,
		delivery_addresses=delivery_addresses,
		current_company=current_company,
		current_job=current_job,
		gender=gender,
		birth_date=birth_date,
		education=education_entries,
		skills=skills,
		social_profiles=social_profiles,
		hukou_place=hukou_place,
		hukou_address=hukou_address,
		custom_attributes=custom_attrs,  # 🔥 保存动态检测的自定义字段
		raw_text=text,
		id_numbers=id_numbers,
		relationships=relationships,  # 🔥 保存同户人关系
		data_sources=[{"type": "manual_import", "timestamp": datetime.now().isoformat()}],
		confidence_score=0.8,
		created_at=datetime.now(),
		updated_at=datetime.now(),
		verified=False
	)
	return profile


def store_profile(profile: PersonProfile) -> Dict[str, Any]:
    """将档案写入本地JSON存储，返回保存后的字典"""
    ensure_storage()
    persons = load_all_persons()
    record = profile.model_dump(mode="json")
    persons.append(record)
    save_all_persons(persons)
    logger.info(f"已导入人物信息: {profile.name} ({profile.id})")
    log_audit("PERSON_IMPORT", profile.name, details=f"person_id={profile.id}")
    return record


def delete_person_by_id(person_id: str) -> bool:
    """按ID删除人物，返回是否删除成功"""
    ensure_storage()
    persons = load_all_persons()
    new_persons = [p for p in persons if str(p.get('id')) != str(person_id)]
    if len(new_persons) == len(persons):
        return False
    save_all_persons(new_persons)
    logger.info(f"已删除人物信息: {person_id}")
    log_audit("PERSON_DELETE", str(person_id), details="deleted from JSON store")
    return True


def _extract(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _search(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    if not m:
        return None
    return m.group(0)


def extract_emails(text: str) -> List[str]:
    return [m.group(0) for m in EMAIL_REGEX.finditer(text)]


def extract_phones(text: str) -> List[str]:
    """提取电话列表，优先返回中国大陆手机号，其次座机；排除身份证等长数字。"""
    candidates: List[str] = []
    for label in ['手机号', '手机', '电话', '联系电话', 'Phone', 'Tel']:
        for line in text.splitlines():
            if label in line:
                for m in MOBILE_REGEX.finditer(line):
                    candidates.append(m.group(1))
                for m in LANDLINE_REGEX.finditer(line):
                    area, num = m.group(1), m.group(2)
                    candidates.append(f"{area}-{num}")
    for m in MOBILE_REGEX.finditer(text):
        candidates.append(m.group(1))
    for m in LANDLINE_REGEX.finditer(text):
        area, num = m.group(1), m.group(2)
        candidates.append(f"{area}-{num}")
    filtered: List[str] = []
    for c in candidates:
        if ID_REGEX.search(c):
            continue
        if c.isdigit() and len(c) != 11 and '-' not in c:
            continue
        if c.replace('-', '').isdigit() and len(c.replace('-', '')) == 11:
            if c.replace('-', '')[0] != '1':
                continue
        if c not in filtered:
            filtered.append(c)
    filtered.sort(key=lambda s: 0 if s.replace('-', '').isdigit() and len(s.replace('-', '')) == 11 else 1)
    return filtered


def extract_addresses(text: str) -> List[str]:
    addrs: List[str] = []
    for line in text.splitlines():
        m = ADDRESS_LABEL_REGEX.search(line)
        if m:
            addrs.append(m.group(1).strip())
    return addrs


def extract_id_numbers(text: str) -> List[str]:
    ids: List[str] = []
    for m in ID_REGEX.finditer(text):
        ids.append(m.group(0))
    for m in ID15_REGEX.finditer(text):
        ids.append(m.group(0))
    return deduplicate_preserve_order(ids)


def extract_license_plates(text: str) -> List[str]:
    plates = [m.group(0) for m in LICENSE_PLATE_REGEX.finditer(text)]
    return deduplicate_preserve_order(plates)


def try_parse_date(text_value: Optional[str]) -> Optional[datetime]:
    if not text_value:
        return None
    s = text_value.strip()
    # 常见格式：1996年02月19日 / 1996年02月 / 1996-02-19 / 1996/02/19 / 1996-02
    patterns = [
        r"(\d{4})年(\d{1,2})月(\d{1,2})日",
        r"(\d{4})年(\d{1,2})月",
        r"(\d{4})-(\d{1,2})-(\d{1,2})",
        r"(\d{4})/(\d{1,2})/(\d{1,2})",
        r"(\d{4})-(\d{1,2})",
        r"(\d{4})/(\d{1,2})"
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            try:
                y = int(m.group(1))
                mon = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else 1
                d = int(m.group(3)) if m.lastindex and m.lastindex >= 3 else 1
                if 1900 <= y <= 2100 and 1 <= mon <= 12 and 1 <= d <= 31:
                    return datetime(y, mon, d)
            except Exception:
                continue
    # 仅年
    m = re.search(r"(\d{4})年", s)
    if m:
        try:
            y = int(m.group(1))
            if 1900 <= y <= 2100:
                return datetime(y, 1, 1)
        except Exception:
            pass
    return None


def add_attr(attrs: Dict[str, Any], key: str, value: Any):
    """向自定义属性里追加键值；若键已存在，则合并为列表并去重。"""
    if key is None:
        return
    k = str(key).strip()
    if not k:
        return
    v = value
    # 展开简单的容器
    if isinstance(v, list) and len(v) == 1:
        v = v[0]
    # 追加逻辑
    if k not in attrs:
        attrs[k] = v
    else:
        if not isinstance(attrs[k], list):
            attrs[k] = [attrs[k]]
        if isinstance(v, list):
            for item in v:
                if item not in attrs[k]:
                    attrs[k].append(item)
        else:
            if v not in attrs[k]:
                attrs[k].append(v)


def deduplicate_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def extract_household_members(text: str) -> List[str]:
    """从文本里解析同户/同住成员名（粗略），返回姓名列表。"""
    members: List[str] = []
    lines = text.splitlines()
    for line in lines:
        if any(lbl in line for lbl in HOUSEHOLD_LABELS):
            parts = re.split(r"[:：]", line, maxsplit=1)
            segment = parts[1] if len(parts) == 2 else line
            for name in split_names(segment):
                if name:
                    members.append(name)
    return members


def split_names(segment: str) -> List[str]:
    """将一段文本按常见分隔符拆分成候选姓名，过滤明显非姓名的项。"""
    raw = re.split(r"[、,，;；\s]+", segment)
    out: List[str] = []
    for token in raw:
        t = token.strip()
        if not t:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if len(t) > 10:
            continue
        if any(k in t for k in ['性别','男','女','户','座','星座','生肖','住址','地址','号码','证件']):
            continue
        out.append(t)
    return out


def _split_main_and_household_info(text: str) -> tuple[str, str]:
    """
    智能分离主人物信息和同户人信息
    返回: (主人物段落, 同户人段落)
    """
    lines = text.split('\n')
    main_lines = []
    household_lines = []
    in_household_section = False
    
    # 关键词检测
    household_keywords = ['同户人', '同户', '同住人', '家庭成员', '--', '═', '━']
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检测是否进入同户人区域
        if any(keyword in line for keyword in household_keywords):
            in_household_section = True
            household_lines.append(line)
            continue
            
        # 检测是否是分隔线（18个以上的-或其他符号）
        if len([c for c in line if c in '-=━═']) >= 10:
            in_household_section = True
            household_lines.append(line)
            continue
            
        # 如果包含身份证号模式，判断是否属于主人物
        id_matches = ID_REGEX.findall(line)
        if id_matches and not in_household_section:
            # 第一个遇到的身份证信息被认为是主人物
            main_lines.append(line)
        elif id_matches and in_household_section:
            # 后续的身份证信息被认为是同户人
            household_lines.append(line)
        elif in_household_section:
            household_lines.append(line)
        else:
            main_lines.append(line)
    
    main_section = '\n'.join(main_lines)
    household_section = '\n'.join(household_lines)
    
    return main_section, household_section


def _parse_household_members(household_text: str) -> List[Dict[str, Any]]:
    """
    解析同户人信息段落，返回同户人列表
    """
    members = []
    
    # 分行处理
    lines = household_text.split('\n')
    current_member = {}
    
    for line in lines:
        line = line.strip()
        if not line or any(sep in line for sep in ['--', '━', '═', '=']):
            # 分隔线，保存当前成员
            if current_member.get('name'):
                members.append(current_member)
            current_member = {}
            continue
            
        # 提取姓名 - 改进姓名提取逻辑
        name_match = re.search(r'([^,，\s：:]{2,4})\s*[,，]', line)
        if name_match:
            current_member['name'] = name_match.group(1)
        else:
            # 尝试其他格式：如果行开头是姓名，没有逗号分隔
            first_word = line.split()[0] if line.split() else ''
            if first_word and len(first_word) >= 2 and len(first_word) <= 4 and not any(c.isdigit() for c in first_word):
                # 进一步过滤冒号等标点
                clean_name = first_word.strip('：:,，。')
                if clean_name and len(clean_name) >= 2:
                    current_member['name'] = clean_name
        
        # 提取性别
        if '男' in line:
            current_member['gender'] = '男'
        elif '女' in line:
            current_member['gender'] = '女'
            
        # 提取身份证号
        id_match = ID_REGEX.search(line)
        if id_match:
            current_member['id_number'] = id_match.group(0)
            
        # 提取出生日期
        birth_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', line)
        if birth_match:
            current_member['birth_date'] = f"{birth_match.group(1)}-{birth_match.group(2).zfill(2)}-{birth_match.group(3).zfill(2)}"
            
        # 提取电话
        phone_matches = MOBILE_REGEX.findall(line)
        if phone_matches:
            current_member['phone'] = phone_matches[0]
            
        # 存储原始行信息
        if 'raw_lines' not in current_member:
            current_member['raw_lines'] = []
        current_member['raw_lines'].append(line)
    
    # 保存最后一个成员
    if current_member.get('name'):
        members.append(current_member)
    
    return members


def _extract_dynamic_fields(text: str) -> Dict[str, Any]:
    """
    智能提取动态字段（生肖、星座、其他标签等）
    """
    custom_attrs = {}
    
    # 生肖检测
    zodiac_match = re.search(r'生肖\s*[:：]\s*([^，\s,；;]+)', text)
    if zodiac_match:
        custom_attrs['生肖'] = zodiac_match.group(1)
    
    # 星座检测
    constellation_match = re.search(r'星座\s*[:：]\s*([^，\s,；;]+)', text)
    if constellation_match:
        custom_attrs['星座'] = constellation_match.group(1)
        
    # 农历信息
    lunar_match = re.search(r'([甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]年)', text)
    if lunar_match:
        custom_attrs['农历年份'] = lunar_match.group(1)
        
    # 农历日期
    lunar_date_match = re.search(r'((?:正|二|三|四|五|六|七|八|九|十|冬|腊)月\s*(?:初|十|廿|卅)?[一二三四五六七八九十]{1,2})', text)
    if lunar_date_match:
        custom_attrs['农历日期'] = lunar_date_match.group(1)
    
    # 籍贯、出生地等地理信息
    birthplace_match = re.search(r'出生地\s*[:：]\s*([^，\n；;]+)', text)
    if birthplace_match:
        custom_attrs['出生地'] = birthplace_match.group(1)
        
    # 通用键值对检测 (格式：key: value 或 key：value)
    kv_matches = re.findall(r'([^：:\n]{1,10})\s*[:：]\s*([^，\n；;]{1,50})', text)
    for key, value in kv_matches:
        key = key.strip()
        value = value.strip()
        
        # 过滤已知字段和无意义字段
        skip_keys = ['姓名', '性别', '电话', '手机', '地址', '住址', '身份证', '出生', '年月', '号码']
        if any(skip in key for skip in skip_keys):
            continue
            
        if len(key) >= 2 and len(value) >= 1 and key not in custom_attrs:
            custom_attrs[key] = value
    
    return custom_attrs 