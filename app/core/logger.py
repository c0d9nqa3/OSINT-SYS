"""
日志管理系统
"""

import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings

# 创建日志目录
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 移除默认处理器
logger.remove()

# 控制台输出
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
           "<level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True
)

# 文件输出 - 普通日志
logger.add(
    "logs/osint_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="INFO",
    rotation="1 day",
    retention="30 days",
    compression="zip"
)

# 文件输出 - 错误日志
logger.add(
    "logs/error_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    level="ERROR",
    rotation="1 day",
    retention="30 days",
    compression="zip"
)

# 审计日志 - 用于合规性记录
audit_logger = logger.bind(audit=True)
logger.add(
    "logs/audit_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {message}",
    filter=lambda record: record["extra"].get("audit", False),
    rotation="1 day",
    retention="365 days",  # 审计日志保留一年
    compression="zip"
)

# 数据收集日志 - 记录所有数据收集活动
collection_logger = logger.bind(collection=True)
logger.add(
    "logs/collection_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | COLLECTION | {message}",
    filter=lambda record: record["extra"].get("collection", False),
    rotation="1 day",
    retention="90 days",
    compression="zip"
)

def log_audit(action: str, target: str, user: str = "system", details: str = ""):
    """记录审计日志"""
    audit_logger.info(f"ACTION:{action} | TARGET:{target} | USER:{user} | DETAILS:{details}")

def log_collection(source: str, target: str, data_type: str, status: str, details: str = ""):
    """记录数据收集活动"""
    collection_logger.info(f"SOURCE:{source} | TARGET:{target} | TYPE:{data_type} | STATUS:{status} | DETAILS:{details}")

def log_compliance_warning(message: str):
    """记录合规性警告"""
    logger.warning(f"COMPLIANCE WARNING: {message}")
    log_audit("COMPLIANCE_WARNING", "system", details=message)

# 导出主要日志器
__all__ = ["logger", "log_audit", "log_collection", "log_compliance_warning"] 