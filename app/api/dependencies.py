"""
API依赖注入模块
"""

import asyncio
from app.services.osint_service import OSINTService

# 全局OSINT服务实例
_osint_service = None
_initialized = False

def get_osint_service() -> OSINTService:
    """获取OSINT服务实例（同步版本）"""
    global _osint_service, _initialized
    if _osint_service is None:
        _osint_service = OSINTService()
        # 延迟初始化，在第一次使用时再初始化
        _initialized = False
    return _osint_service 