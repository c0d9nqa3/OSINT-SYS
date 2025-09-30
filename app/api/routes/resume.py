"""
履历解析相关API路由
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/resume/{investigation_id}")
async def get_resume_data(investigation_id: str):
    """获取履历数据"""
    return {"message": "履历解析功能"} 