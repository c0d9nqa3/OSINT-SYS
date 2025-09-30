"""
输入校验与规范化工具
- 提供手机号/邮箱/身份证等字段校验
- 提供人口档案字典的规范化函数（去重、裁剪、格式统一）
"""

import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
MOBILE_REGEX = re.compile(r"^(?:\+?86)?1[3-9]\d{9}$")
QQ_REGEX = re.compile(r"^[1-9]\d{4,10}$")
URL_REGEX = re.compile(r"^https?://[\w.-]+(?:/[\w\-./?%&=]*)?$", re.IGNORECASE)

# 中国大陆身份证校验
AREA_CODE_REGEX = re.compile(r"^\d{6}")
ID18_REGEX = re.compile(r"^\d{17}[0-9Xx]$")
ID15_REGEX = re.compile(r"^\d{15}$")

ID_WEIGHT = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
ID_CHECKMAP = "10X98765432"


def is_valid_email(v: str) -> bool:
	return bool(v and EMAIL_REGEX.match(v))


def is_valid_cn_mobile(v: str) -> bool:
	return bool(v and MOBILE_REGEX.match(v))


def is_valid_url(v: str) -> bool:
	return bool(v and URL_REGEX.match(v))


def is_valid_prc_id(v: str) -> bool:
	if not v:
		return False
	v = v.strip()
	if ID15_REGEX.match(v):
		# 15位基本合法（不做年代转换）
		return True
	if not ID18_REGEX.match(v):
		return False
	# 校验码
	sumv = 0
	for i, ch in enumerate(v[:17]):
		sumv += int(ch) * ID_WEIGHT[i]
	check = ID_CHECKMAP[sumv % 11]
	return check == v[-1].upper()


def dedup_keep_order(items: List[str]) -> List[str]:
	seen = set()
	out = []
	for it in items or []:
		it = (it or '').strip()
		if not it or it in seen:
			continue
		seen.add(it)
		out.append(it)
	return out


def normalize_profile_dict(profile: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
	"""对档案进行规范化与校验，返回(规范化结果, 错误列表)"""
	errors: List[str] = []
	data = dict(profile or {})
	# 修剪字符串
	for k, v in list(data.items()):
		if isinstance(v, str):
			data[k] = v.strip()

	# 多值字段去重
	for key in ['phones', 'delivery_addresses', 'id_numbers', 'skills']:
		if key in data and isinstance(data[key], list):
			data[key] = dedup_keep_order([str(x).strip() for x in data[key]])

	# 手机号
	valid_phones = []
	invalid_phones = []
	for p in data.get('phones', []):
		if is_valid_cn_mobile(p):
			valid_phones.append(p)
		else:
			invalid_phones.append(p)
	if invalid_phones:
		errors.append(f"无效手机号: {', '.join(invalid_phones)}")
	data['phones'] = valid_phones
	if data.get('phone') and not is_valid_cn_mobile(data['phone']):
		errors.append(f"主手机号无效: {data['phone']}")

	# 邮箱
	if data.get('email') and not is_valid_email(data['email']):
		errors.append(f"邮箱格式无效: {data['email']}")

	# 身份证
	valid_ids = []
	invalid_ids = []
	for i in data.get('id_numbers', []):
		if is_valid_prc_id(i):
			valid_ids.append(i.upper())
		else:
			invalid_ids.append(i)
	if invalid_ids:
		errors.append(f"无效身份证号: {', '.join(invalid_ids)}")
	data['id_numbers'] = dedup_keep_order(valid_ids)

	# 社交账号简单校验
	sp = data.get('social_profiles', {}) or {}
	qq = data.get('qq_id')
	if qq and not QQ_REGEX.match(qq):
		errors.append("QQ号应为5-11位数字")
	weibo = data.get('weibo_id')
	if weibo and not is_valid_url(weibo):
		errors.append("微博应为有效链接")
	if 'weibo' in sp and not is_valid_url(sp['weibo']):
		errors.append("social_profiles.weibo 不是有效链接")

	# 户籍相关（无需强制，但去空格）
	for k in ['hukou_place', 'hukou_address', 'address']:
		if data.get(k):
			data[k] = data[k].strip()

	# 主字段回填
	if not data.get('phone') and data.get('phones'):
		data['phone'] = data['phones'][0]

	return data, errors 