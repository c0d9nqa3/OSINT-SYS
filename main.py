#!/usr/bin/env python3

import uvicorn
import sys
import os
import importlib
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.api.main import create_app
from app.core.config import settings
from app.core.logger import logger

# 创建应用实例供uvicorn使用
app = create_app()

def setup_development_mode():
    """设置开发模式，启用热重载和模块自动刷新"""
    # 强制启用调试模式
    settings.DEBUG = True
    settings.ENVIRONMENT = "development"
    
    # 清理可能的模块缓存
    modules_to_reload = [
        'app.services.smart_text_parser',
        'app.services.text_person_importer',
        'app.services.osint_service',
        'app.ai.identity_verifier',
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                logger.info(f"🔄 重载模块: {module_name}")
            except Exception as e:
                logger.warning(f"模块重载失败 {module_name}: {e}")
    
    logger.info("🛠️ 开发模式已启用，支持热重载")

def check_environment():
    """检查运行环境"""
    print("\n" + "="*60)
    print("🚀 OSINT情报收集系统启动检查")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python版本: {python_version}")
    
    # 检查项目路径
    print(f"📁 项目路径: {project_root}")
    
    # 检查关键文件
    critical_files = [
        "app/api/main.py",
        "app/core/config.py", 
        "templates/index.html",
        "templates/login.html",
        "static/js/main.js"
    ]
    
    missing_files = []
    for file_path in critical_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 文件不存在！")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  警告：发现 {len(missing_files)} 个缺失文件")
        return False
    
    # 检查配置
    print(f"\n🔧 运行配置:")
    print(f"   环境: {settings.ENVIRONMENT}")
    print(f"   调试模式: {settings.DEBUG}")
    print(f"   主机: {settings.HOST}")
    print(f"   端口: {settings.PORT}")
    
    print(f"\n🔐 安全配置:")
    secret_preview = settings.SECRET_KEY[:10] + "..." if len(settings.SECRET_KEY) > 10 else settings.SECRET_KEY
    print(f"   会话密钥: {secret_preview}")
    
    print(f"\n📱 访问信息:")
    print(f"   主页: http://{settings.HOST}:{settings.PORT}")
    print(f"   登录: http://{settings.HOST}:{settings.PORT}/login")
    print(f"   API文档: http://{settings.HOST}:{settings.PORT}/docs")
    print(f"   登录密码: 66642012")
    
    print("="*60)
    return True

def main():
    """启动应用程序"""
    
    # 设置开发模式（强制启用热重载）
    setup_development_mode()
    
    # 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请修复上述问题后重试")
        sys.exit(1)
    
    print("🎯 开始启动服务器...")
    logger.info("正在启动OSINT情报收集系统...")
    logger.info(f"运行环境: {settings.ENVIRONMENT}")
    logger.info(f"服务地址: {settings.HOST}:{settings.PORT}")
    
    try:
        # 启动服务器
        uvicorn.run(
            "main:app",  # 使用导入字符串以支持热重载
            host=settings.HOST,
            port=settings.PORT,
            reload=True,  # 强制启用热重载
            reload_dirs=[str(project_root)],  # 监控整个项目目录
            reload_includes=["*.py"],  # 监控Python文件
            log_level="info",
            access_log=True,  # 显示访问日志
            )
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在关闭服务器...")
        logger.info("用户中断，服务器已关闭")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        logger.error(f"服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 