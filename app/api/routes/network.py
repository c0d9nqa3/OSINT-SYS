"""
网络分析相关API路由
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/network/{investigation_id}")
async def get_network_analysis(investigation_id: str):
    """获取网络分析结果"""
    return {"message": "网络分析功能"} 