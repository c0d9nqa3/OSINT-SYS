"""
主要API应用
提供OSINT情报收集系统的REST API接口
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
from typing import List, Dict, Any, Optional
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import settings, validate_compliance
from app.core.logger import logger, log_audit
from app.api.routes import investigation, network, resume, web, person_import
from app.api.dependencies import get_osint_service

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""应用生命周期管理"""
	# 启动时
	logger.info("正在启动OSINT情报收集系统...")
	
	# 验证合规性配置
	compliance_issues = validate_compliance()
	if compliance_issues:
		for issue in compliance_issues:
			logger.warning(issue)
	
	# 初始化服务
	osint_service = get_osint_service()
	await osint_service.initialize()
	
	logger.info("OSINT系统启动完成")
	log_audit("SYSTEM_START", "system", details="系统启动")
	
	yield
	
	# 关闭时
	logger.info("正在关闭OSINT系统...")
	if osint_service:
		await osint_service.cleanup()
	log_audit("SYSTEM_STOP", "system", details="系统关闭")

def create_app() -> FastAPI:
	"""创建FastAPI应用实例"""
	
	app = FastAPI(
		title="OSINT情报收集系统",
		description="一个合法的开源情报收集工具，专门用于收集和分析公开可获取的信息",
		version="1.0.0",
		lifespan=lifespan,
		docs_url="/docs" if settings.DEBUG else None,
		redoc_url="/redoc" if settings.DEBUG else None
	)
	
	# CORS中间件
	app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	# 会话中间件（用于简单登录态）
	app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
	
	# 静态文件
	app.mount("/static", StaticFiles(directory="static"), name="static")
	
	# 包含路由
	app.include_router(web.router, tags=["Web界面"])
	app.include_router(investigation.router, prefix="/api/v1", tags=["调查"])
	app.include_router(network.router, prefix="/api/v1", tags=["网络分析"])
	app.include_router(resume.router, prefix="/api/v1", tags=["履历解析"])
	app.include_router(person_import.router, prefix="/api/v1", tags=["人物导入"])
	
	return app

 