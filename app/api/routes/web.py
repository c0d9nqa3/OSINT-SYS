"""
Web界面路由
"""

import os
import secrets
from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 从环境变量读取密码，如果未设置则使用默认值
DEFAULT_PASSWORD = "66642012"
PASSWORD = os.getenv("OSINT_PASSWORD", DEFAULT_PASSWORD)

# 生成会话密钥（如果未在环境变量中设置）
SESSION_SECRET = os.getenv("SESSION_SECRET", secrets.token_hex(32))

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
	"""登录页面"""
	return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def do_login(request: Request, password: str = Form(...)):
	"""处理登录"""
	if password == PASSWORD:
		request.session['logged_in'] = True
		request.session['user_id'] = 'admin'
		print("✅ 登录成功")
		return RedirectResponse(url='/', status_code=303)
	
	# 记录失败的登录尝试
	client_ip = request.client.host if request.client else "unknown"
	print(f"⚠️  登录失败尝试 - IP: {client_ip}")
	
	return templates.TemplateResponse("login.html", {"request": request, "error": "密码错误"}, status_code=401)

@router.get("/logout")
async def do_logout(request: Request):
	"""注销"""
	print("🚪 用户注销")
	request.session.clear()
	return RedirectResponse(url='/login', status_code=303)

@router.get("/clear-session")
async def clear_session(request: Request):
	"""清理会话（调试用）"""
	print("🧹 强制清理会话")
	request.session.clear()
	return RedirectResponse(url='/login', status_code=303)

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
	"""主页，已登录显示主页，未登录跳转登录"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		print("🔄 未登录，重定向到登录页")
		return RedirectResponse(url='/login', status_code=303)
	
	print("✅ 已登录，显示主页")
	return templates.TemplateResponse("index.html", {"request": request})

@router.get("/main", response_class=HTMLResponse)
async def main_page(request: Request):
	"""登录后的主页"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	print(f"🔍 访问主页 - 登录状态: {logged_in}, 用户ID: {user_id}")
	
	if not logged_in or not user_id:
		print("🔄 未登录，重定向到登录页")
		return RedirectResponse(url='/login', status_code=303)
	
	print("✅ 已登录，显示主页")
	return templates.TemplateResponse("index.html", {"request": request})

@router.get("/arsenal", response_class=HTMLResponse)
async def arsenal_page(request: Request):
	"""信息武器库页面"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		return RedirectResponse(url='/login', status_code=303)
	
	return templates.TemplateResponse("arsenal.html", {"request": request})

@router.get("/investigation", response_class=HTMLResponse)
async def investigation_page(request: Request):
	"""公开资源调查页面"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		return RedirectResponse(url='/login', status_code=303)
	
	return templates.TemplateResponse("investigation.html", {"request": request})

@router.get("/network", response_class=HTMLResponse)
async def network_page(request: Request):
	"""关系网络构建分析页面"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		return RedirectResponse(url='/login', status_code=303)
	
	return templates.TemplateResponse("network.html", {"request": request})

@router.get("/social_engineering", response_class=HTMLResponse)
async def social_engineering_page(request: Request):
	"""社工库查询页面"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		return RedirectResponse(url='/login', status_code=303)
	
	return templates.TemplateResponse("social_engineering.html", {"request": request})

@router.get("/attack", response_class=HTMLResponse)
async def attack_page(request: Request):
	"""信息攻击页面"""
	logged_in = request.session.get('logged_in', False)
	user_id = request.session.get('user_id', None)
	
	if not logged_in or not user_id:
		return RedirectResponse(url='/login', status_code=303)
	
	return templates.TemplateResponse("attack.html", {"request": request}) 

# 安全检查中间件函数
def check_auth(request: Request):
	"""检查用户是否已登录"""
	return request.session.get('logged_in', False)

def get_current_user(request: Request):
	"""获取当前用户信息"""
	if check_auth(request):
		return request.session.get('user_id', 'anonymous')
	return None 