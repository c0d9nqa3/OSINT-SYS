"""
人物信息导入API
- 接收自由文本，解析人物信息并保存
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
import json
from datetime import datetime
import os
import time

from app.services.text_person_importer import parse_free_text_to_profile, store_profile, delete_person_by_id
from app.core.logger import logger
from app.services.validators import normalize_profile_dict

router = APIRouter()

UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class TextImportRequest(BaseModel):
    text: str = Field(..., description="人物自由文本资料")
    user_id: str = Field(default="anonymous", description="用户ID")

class TextImportResponse(BaseModel):
    person: Dict[str, Any]
    message: str

class PersonUpdateRequest(BaseModel):
    # 允许额外字段
    model_config = ConfigDict(extra='allow')
    
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    phones: Optional[list] = None
    address: Optional[str] = None
    delivery_addresses: Optional[list] = None
    current_job: Optional[str] = None
    current_company: Optional[str] = None
    gender: Optional[str] = None
    skills: Optional[list] = None
    education: Optional[list] = None
    social_profiles: Optional[dict] = None
    photo_url: Optional[str] = None
    id_numbers: Optional[list] = None
    hukou_place: Optional[str] = None
    hukou_address: Optional[str] = None
    # 固定社交字段（去掉telegram/twitter/github）
    wechat_id: Optional[str] = None
    qq_id: Optional[str] = None
    weibo_id: Optional[str] = None
    douyin_id: Optional[str] = None
    xhs_id: Optional[str] = None
    gitee_username: Optional[str] = None
    linkedin_url: Optional[str] = None
    # 保留字段但默认不接收自定义
    custom_attributes: Optional[dict] = None

@router.post("/persons/import_text", response_model=TextImportResponse)
async def import_person_from_text(payload: TextImportRequest):
    try:
        profile = parse_free_text_to_profile(payload.text)
        data, errors = normalize_profile_dict(profile.model_dump(mode="json"))
        if errors:
            raise HTTPException(status_code=400, detail="; ".join(errors))
        # 使用规范化后的数据覆盖并保存
        from app.models.person import PersonProfile
        profile = PersonProfile(**data)
        record = store_profile(profile)
        return TextImportResponse(person=record, message="导入成功")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except json.JSONDecodeError as jde:
        logger.exception("JSON读取失败")
        raise HTTPException(status_code=400, detail=f"数据文件损坏: {str(jde)}")
    except Exception as e:
        logger.exception("导入人物信息失败")
        raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")

@router.get("/persons", response_model=Dict[str, Any])
async def list_imported_persons():
    from app.services.text_person_importer import load_all_persons
    try:
        persons = load_all_persons()
        return {"count": len(persons), "items": persons}
    except Exception as e:
        logger.exception("获取人物列表失败")
        raise HTTPException(status_code=500, detail=f"获取人物列表失败: {str(e)}")

@router.get("/persons/{person_id}", response_model=Dict[str, Any])
async def get_person(person_id: str):
    from app.services.text_person_importer import load_all_persons
    persons = load_all_persons()
    for p in persons:
        if str(p.get('id')) == str(person_id):
            return p
    raise HTTPException(status_code=404, detail="人物不存在")

@router.put("/persons/{person_id}", response_model=Dict[str, Any])
async def update_person(person_id: str, payload: PersonUpdateRequest):
    from app.services.text_person_importer import load_all_persons, save_all_persons
    try:
        persons = load_all_persons()
        updated = None
        for i, p in enumerate(persons):
            if str(p.get('id')) == str(person_id):
                data = p.copy()
                patch_raw = payload.model_dump(exclude_unset=True)
                # 过滤空字符串，避免把已有字段清空
                patch = {k: v for k, v in patch_raw.items() if not (isinstance(v, str) and v.strip() == '')}
                # 允许的列表/字典字段整体覆盖
                list_or_dict_fields = ['skills', 'education', 'social_profiles', 'phones', 'delivery_addresses', 'id_numbers']
                scalar_fields = ['name','email','phone','address','current_job','current_company','gender','photo_url','hukou_place','hukou_address',
                                 'wechat_id','qq_id','weibo_id','douyin_id','xhs_id','gitee_username','linkedin_url']
                for key in list_or_dict_fields:
                    if key in patch and patch[key] is not None:
                        data[key] = patch[key]
                for key in scalar_fields:
                    if key in patch and patch[key] is not None:
                        data[key] = patch[key]
                # 强制不保留自定义字段，除非明确传入空对象
                data['custom_attributes'] = patch.get('custom_attributes', {})
                # 其余未知字段不写入
                # 规范化与校验
                normalized, errors = normalize_profile_dict(data)
                if errors:
                    raise HTTPException(status_code=400, detail="; ".join(errors))
                normalized['updated_at'] = datetime.now().isoformat()
                persons[i] = normalized
                updated = normalized
                break
        if not updated:
            raise HTTPException(status_code=404, detail="人物不存在")
        save_all_persons(persons)
        return updated
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("更新人物失败")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

@router.post("/persons/{person_id}/photo", response_model=Dict[str, Any])
async def upload_person_photo(person_id: str, file: UploadFile = File(...)):
    from app.services.text_person_importer import load_all_persons, save_all_persons
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/gif"]:
            raise HTTPException(status_code=400, detail="不支持的图片类型，仅支持 jpg/png/webp/gif")
        ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif"
        }.get(file.content_type, "")
        filename = f"{person_id}_{int(time.time())}{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        photo_url = f"/static/uploads/{filename}"
        persons = load_all_persons()
        found = False
        for i, p in enumerate(persons):
            if str(p.get('id')) == str(person_id):
                p = p.copy()
                p['photo_url'] = photo_url
                p['updated_at'] = datetime.now().isoformat()
                persons[i] = p
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="人物不存在")
        save_all_persons(persons)
        return {"success": True, "photo_url": photo_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("上传照片失败")
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@router.delete("/persons/{person_id}")
async def delete_person(person_id: str) -> Dict[str, Any]:
    try:
        ok = delete_person_by_id(person_id)
        if not ok:
            raise HTTPException(status_code=404, detail="人物不存在或已被删除")
        return {"success": True, "deleted_id": person_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("删除人物失败")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}") 