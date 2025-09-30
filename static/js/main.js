/**
 * OSINT情报收集系统 - 前端JavaScript
 */

// API基础URL
const API_BASE = '/api/v1';

// 全局变量
let investigations = [];
let currentInvestigationId = null;
let persons = [];
let currentView = 'main'; // 'main', 'arsenal', 'investigation', 'network', 'social_engineering', 'attack'

// 页面导航函数
function goToArsenal() {
    console.log('🔄 导航到信息武器库页面...');
    try {
        window.location.href = '/arsenal';
    } catch (error) {
        console.error('导航到武器库页面失败:', error);
        alert('导航失败，请检查网络连接');
    }
}

function goToInvestigation() {
    console.log('🔄 导航到公开资源调查页面...');
    try {
        window.location.href = '/investigation';
    } catch (error) {
        console.error('导航到调查页面失败:', error);
        alert('导航失败，请检查网络连接');
    }
}

function goToNetwork() {
    console.log('🔄 导航到关系网络构建分析页面...');
    try {
        window.location.href = '/network';
    } catch (error) {
        console.error('导航到网络分析页面失败:', error);
        alert('导航失败，请检查网络连接');
    }
}

function goToSocialEngineering() {
    console.log('🔄 导航到社工库查询页面...');
    try {
        window.location.href = '/social_engineering';
    } catch (error) {
        console.error('导航到社工库页面失败:', error);
        alert('导航失败，请检查网络连接');
    }
}

function goToAttackPage() {
    console.log('🔄 导航到信息攻击页面...');
    try {
        window.location.href = '/attack';
    } catch (error) {
        console.error('导航到攻击页面失败:', error);
        alert('导航失败，请检查网络连接');
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('🔄 DOM加载完成，开始初始化应用...');
    initializeApp();
});

// 添加页面加载完成的监听
window.addEventListener('load', function() {
    console.log('🔄 页面完全加载完成');
    console.log('主页元素:', document.getElementById('mainPage'));
    console.log('主页显示状态:', document.getElementById('mainPage')?.style.display);
});

/**
 * 初始化应用
 */
function initializeApp() {
    console.log('🔄 初始化应用...');
    
    // 绑定事件
    bindEvents();
    
    // 根据当前页面执行相应的初始化
    const currentPath = window.location.pathname;
    
    if (currentPath === '/') {
        // 主页初始化 - 只初始化基本功能，不调用特定页面的函数
        console.log('✅ 初始化主页');
        // 主页不需要调用refreshInvestigations和refreshPersons，因为这些元素不存在
    } else if (currentPath === '/investigation') {
        // 调查页面初始化
        console.log('✅ 初始化调查页面');
        refreshInvestigations();
        setInterval(refreshInvestigations, 10000);
    } else if (currentPath === '/arsenal') {
        // 武器库页面初始化
        console.log('✅ 初始化武器库页面');
        refreshPersons();
        refreshProfileList();
        refreshReportPersonList();
        refreshProfilePersonList();
    } else if (currentPath === '/attack') {
        // 攻击页面初始化
        console.log('✅ 初始化攻击页面');
        initializeAttackPageTargets();
    }
    
    console.log('✅ 应用初始化完成');
}

/**
 * 绑定事件处理器
 */
function bindEvents() {
    // 根据当前页面绑定相应的事件
    const currentPath = window.location.pathname;
    
    if (currentPath === '/investigation') {
        // 调查页面事件
        const newInvestigationForm = document.getElementById('newInvestigationForm');
        if (newInvestigationForm) {
            newInvestigationForm.addEventListener('submit', handleNewInvestigation);
        }
        
        const exportInvestigation = document.getElementById('exportInvestigation');
        if (exportInvestigation) {
            exportInvestigation.addEventListener('click', handleExportInvestigation);
        }
    } else if (currentPath === '/arsenal') {
        // 武器库页面事件
        const comprehensiveForm = document.getElementById('comprehensiveAnalysisForm');
        if (comprehensiveForm) {
            comprehensiveForm.addEventListener('submit', handleComprehensiveAnalysisSubmit);
        }
    }
}



/**
 * 处理新建调查
 */
async function handleNewInvestigation(event) {
    event.preventDefault();
    
    const targetName = document.getElementById('targetName').value.trim();
    const userId = document.getElementById('userId').value.trim() || 'anonymous';
    
    if (!targetName || targetName.length < 2) {
        showAlert('请输入至少2个字符的姓名', 'danger');
        return;
    }
    
    // 显示加载状态
    const submitBtn = event.target.querySelector('button[type="submit"]');
    const originalText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>正在启动...';
    submitBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/investigations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target_name: targetName,
                user_id: userId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        showAlert(`调查已启动：${result.message}`, 'success');
        
        // 清空表单
        document.getElementById('targetName').value = '';
        
        // 刷新调查列表
        setTimeout(refreshInvestigations, 1000);
        
    } catch (error) {
        console.error('启动调查失败:', error);
        showAlert(`启动调查失败: ${error.message}`, 'danger');
    } finally {
        // 恢复按钮状态
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

/**
 * 刷新调查列表
 */
async function refreshInvestigations() {
    try {
        console.log('🔄 正在获取调查列表...');
        const response = await fetch(`${API_BASE}/investigations`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('📊 获取到调查数据:', data);
        
        investigations = Array.isArray(data) ? data : [];
        console.log('✅ 调查数据已更新，数量:', investigations.length);
        
        renderInvestigations();
        updateSystemStatus();
        
    } catch (error) {
        console.error('❌ 获取调查列表失败:', error);
        // 显示错误信息给用户
        const container = document.getElementById('investigationsContainer');
        if (container) {
            container.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    获取调查列表失败: ${error.message}
                    <br><small>请检查网络连接或联系管理员</small>
                </div>
            `;
        } else {
            console.log('⚠️ investigationsContainer元素不存在，跳过错误显示');
        }
    }
}

/**
 * 渲染调查列表
 */
function renderInvestigations() {
    console.log('🎨 开始渲染调查列表，数据数量:', investigations.length);
    
    const container = document.getElementById('investigationsContainer');
    if (!container) {
        console.error('❌ 找不到investigationsContainer元素');
        return;
    }
    
    if (!investigations || investigations.length === 0) {
        console.log('📝 没有调查数据，显示空状态');
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-search fa-3x mb-3"></i>
                <p>暂无调查记录</p>
                <p class="small">点击"开始调查"创建新的调查任务</p>
            </div>
        `;
        return;
    }
    
    const html = investigations.map(investigation => {
        const statusBadge = getStatusBadge(investigation.status);
        const progressBar = getProgressBar(investigation.progress);
        const duration = calculateDuration(investigation.start_time, investigation.end_time);
        
        return `
            <div class="card mb-3">
                <div class="card-body">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <h6 class="card-title mb-1">
                                <i class="fas fa-user me-2"></i>
                                ${escapeHtml(investigation.target_name)}
                            </h6>
                            <p class="card-text small text-muted mb-2">
                                ID: ${investigation.id}
                            </p>
                            <p class="card-text small">
                                开始时间: ${formatDateTime(investigation.start_time)}
                                ${investigation.end_time ? `<br>结束时间: ${formatDateTime(investigation.end_time)}` : ''}
                                <br>持续时间: ${duration}
                            </p>
                        </div>
                        <div class="col-md-3">
                            ${statusBadge}
                            ${progressBar}
                            ${investigation.results_count ? `<small class="text-muted">结果: ${investigation.results_count}</small>` : ''}
                        </div>
                        <div class="col-md-3 text-end">
                            <button class="btn btn-outline-primary btn-sm me-2" 
                                    onclick="viewInvestigation('${investigation.id}')"
                                    ${investigation.status === 'collecting' ? 'disabled' : ''}>
                                <i class="fas fa-eye me-1"></i>
                                查看详情
                            </button>
                            <button class="btn btn-outline-danger btn-sm" 
                                    onclick="deleteInvestigation('${investigation.id}')">
                                <i class="fas fa-trash me-1"></i>
                                删除
                            </button>
                        </div>
                    </div>
                    ${investigation.error ? `
                        <div class="alert alert-danger mt-3 mb-0">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            错误: ${escapeHtml(investigation.error)}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }).join('');
    
    container.innerHTML = html;
}

/**
 * 获取状态徽章
 */
function getStatusBadge(status) {
    const statusMap = {
        'collecting': { class: 'bg-primary', icon: 'fas fa-download', text: '收集中' },
        'verifying': { class: 'bg-info', icon: 'fas fa-check-circle', text: '验证中' },
        'analyzing_network': { class: 'bg-warning', icon: 'fas fa-project-diagram', text: '分析网络' },
        'parsing_resume': { class: 'bg-secondary', icon: 'fas fa-file-alt', text: '解析履历' },
        'completed': { class: 'bg-success', icon: 'fas fa-check', text: '已完成' },
        'failed': { class: 'bg-danger', icon: 'fas fa-times', text: '失败' }
    };
    
    const statusInfo = statusMap[status] || { class: 'bg-secondary', icon: 'fas fa-question', text: '未知' };
    
    return `
        <span class="badge ${statusInfo.class} mb-2">
            <i class="${statusInfo.icon} me-1"></i>
            ${statusInfo.text}
        </span>
    `;
}

/**
 * 获取进度条
 */
function getProgressBar(progress) {
    if (progress === 0) return '';
    
    return `
        <div class="progress mb-2" style="height: 8px;">
            <div class="progress-bar" role="progressbar" 
                 style="width: ${progress}%" 
                 aria-valuenow="${progress}" 
                 aria-valuemin="0" 
                 aria-valuemax="100">
            </div>
        </div>
        <small class="text-muted">${progress}%</small>
    `;
}

/**
 * 计算持续时间
 */
function calculateDuration(startTime, endTime) {
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    
    const diffMs = end - start;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    
    if (diffHours > 0) {
        return `${diffHours}小时${diffMins % 60}分钟`;
    } else if (diffMins > 0) {
        return `${diffMins}分钟`;
    } else {
        return `${diffSecs}秒`;
    }
}

/**
 * 格式化日期时间
 */
function formatDateTime(dateTimeStr) {
    const date = new Date(dateTimeStr);
    return date.toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * 更新系统状态
 */
function updateSystemStatus() {
    const activeCount = investigations.filter(inv => 
        ['collecting', 'verifying', 'analyzing_network', 'parsing_resume'].includes(inv.status)
    ).length;
    
    const completedCount = investigations.filter(inv => inv.status === 'completed').length;
    
    document.getElementById('activeInvestigations').textContent = activeCount;
    document.getElementById('completedInvestigations').textContent = completedCount;
}

/**
 * 查看调查详情
 */
async function viewInvestigation(investigationId) {
    currentInvestigationId = investigationId;
    
    try {
        // 显示加载状态
        document.getElementById('investigationDetails').innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-spinner fa-spin fa-3x mb-3"></i>
                <p>正在加载调查详情...</p>
            </div>
        `;
        
        // 显示模态框
        new bootstrap.Modal(document.getElementById('investigationModal')).show();
        
        // 获取调查结果
        const response = await fetch(`${API_BASE}/investigations/${investigationId}/results`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const results = await response.json();
        renderInvestigationDetails(results);
        
    } catch (error) {
        console.error('获取调查详情失败:', error);
        document.getElementById('investigationDetails').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                获取调查详情失败: ${error.message}
            </div>
        `;
    }
}

/**
 * 渲染调查详情
 */
function renderInvestigationDetails(results) {
    let html = `
        <div class="row">
            <div class="col-md-6">
                <h6>基本信息</h6>
                <table class="table table-sm">
                    <tr><td>姓名</td><td>${escapeHtml(results.target_profile.name)}</td></tr>
                    <tr><td>调查ID</td><td><code>${results.id}</code></td></tr>
                    <tr><td>状态</td><td>${getStatusBadge(results.status)}</td></tr>
                    <tr><td>进度</td><td>${results.progress}%</td></tr>
                    <tr><td>搜索结果数</td><td>${results.search_results.length}</td></tr>
                </table>
            </div>
            <div class="col-md-6">
                <h6>目标信息</h6>
                <table class="table table-sm">
                    <tr><td>邮箱</td><td>${results.target_profile.email || '未知'}</td></tr>
                    <tr><td>电话</td><td>${results.target_profile.phone || '未知'}</td></tr>
                    <tr><td>地址</td><td>${results.target_profile.address || '未知'}</td></tr>
                    <tr><td>当前职位</td><td>${results.target_profile.current_job || '未知'}</td></tr>
                    <tr><td>当前公司</td><td>${results.target_profile.current_company || '未知'}</td></tr>
                </table>
            </div>
        </div>
    `;
    
    // 身份验证结果
    if (results.verification_result) {
        const verification = results.verification_result;
        html += `
            <div class="mt-4">
                <h6>身份验证结果</h6>
                <div class="card">
                    <div class="card-body">
                        <p><strong>结论:</strong> ${verification.is_same_person ? '同一人' : '非同一人'}</p>
                        <p><strong>置信度:</strong> ${(verification.confidence_score * 100).toFixed(1)}%</p>
                        <p><strong>验证方法:</strong> ${verification.verification_methods.join(', ')}</p>
                        <p><strong>推理:</strong> ${verification.reasoning}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    // 搜索结果
    if (results.search_results.length > 0) {
        html += `
            <div class="mt-4">
                <h6>搜索结果 (${results.search_results.length})</h6>
                <div class="accordion" id="searchResultsAccordion">
        `;
        
        results.search_results.slice(0, 10).forEach((result, index) => {
            html += `
                <div class="accordion-item">
                    <h2 class="accordion-header" id="heading${index}">
                        <button class="accordion-button collapsed" type="button" 
                                data-bs-toggle="collapse" data-bs-target="#collapse${index}">
                            <div>
                                <strong>${escapeHtml(result.title)}</strong>
                                <br><small class="text-muted">${result.source} - 相关性: ${(result.relevance_score * 100).toFixed(1)}%</small>
                            </div>
                        </button>
                    </h2>
                    <div id="collapse${index}" class="accordion-collapse collapse" 
                         data-bs-parent="#searchResultsAccordion">
                        <div class="accordion-body">
                            <p><strong>URL:</strong> <a href="${result.url}" target="_blank">${result.url}</a></p>
                            <p><strong>摘要:</strong> ${escapeHtml(result.snippet)}</p>
                            <p><strong>可靠性:</strong> ${(result.reliability_score * 100).toFixed(1)}%</p>
                            ${result.person_mentions.length > 0 ? 
                                `<p><strong>提及人名:</strong> ${result.person_mentions.join(', ')}</p>` : ''}
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
    }
    
    // 网络分析结果
    if (results.network_analysis) {
        const network = results.network_analysis;
        html += `
            <div class="mt-4">
                <h6>社会关系网络分析</h6>
                <div class="card">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <strong>节点数:</strong> ${network.total_nodes}
                            </div>
                            <div class="col-md-3">
                                <strong>边数:</strong> ${network.total_edges}
                            </div>
                            <div class="col-md-3">
                                <strong>网络密度:</strong> ${network.density.toFixed(3)}
                            </div>
                            <div class="col-md-3">
                                <strong>社区数:</strong> ${network.communities.length}
                            </div>
                        </div>
                        
                        ${network.key_persons.length > 0 ? `
                            <h6 class="mt-3">关键人物</h6>
                            <ul class="list-group list-group-flush">
                                ${network.key_persons.slice(0, 5).map(person => `
                                    <li class="list-group-item d-flex justify-content-between">
                                        <span>${escapeHtml(person.name)}</span>
                                        <span class="badge bg-primary">${(person.importance_score * 100).toFixed(1)}%</span>
                                    </li>
                                `).join('')}
                            </ul>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    // 履历信息
    if (results.resume_data) {
        const resume = results.resume_data;
        html += `
            <div class="mt-4">
                <h6>履历信息</h6>
                <div class="card">
                    <div class="card-body">
                        ${resume.summary ? `<p><strong>摘要:</strong> ${resume.summary}</p>` : ''}
                        
                        ${resume.work_experiences.length > 0 ? `
                            <h6>工作经历</h6>
                            <ul class="list-group list-group-flush mb-3">
                                ${resume.work_experiences.slice(0, 5).map(work => `
                                    <li class="list-group-item">
                                        <strong>${escapeHtml(work.position)}</strong> @ ${escapeHtml(work.company)}
                                        ${work.start_date ? `<br><small class="text-muted">
                                            ${formatDate(work.start_date)} - ${work.end_date ? formatDate(work.end_date) : '至今'}
                                        </small>` : ''}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : ''}
                        
                        ${resume.education.length > 0 ? `
                            <h6>教育经历</h6>
                            <ul class="list-group list-group-flush mb-3">
                                ${resume.education.map(edu => `
                                    <li class="list-group-item">
                                        <strong>${escapeHtml(edu.degree)}</strong> @ ${escapeHtml(edu.institution)}
                                        ${edu.major ? `<br><small class="text-muted">专业: ${escapeHtml(edu.major)}</small>` : ''}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : ''}
                        
                        ${resume.skills.length > 0 ? `
                            <h6>技能</h6>
                            <div class="mb-3">
                                ${resume.skills.map(skill => `
                                    <span class="badge bg-secondary me-1">${escapeHtml(skill.name)}</span>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    document.getElementById('investigationDetails').innerHTML = html;
}

/**
 * 删除调查
 */
async function deleteInvestigation(investigationId) {
    if (!confirm('确定要删除这个调查吗？此操作不可撤销。')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/investigations/${investigationId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        showAlert('调查已删除', 'success');
        refreshInvestigations();
        
    } catch (error) {
        console.error('删除调查失败:', error);
        showAlert(`删除调查失败: ${error.message}`, 'danger');
    }
}

/**
 * 导出调查报告
 */
async function handleExportInvestigation() {
    if (!currentInvestigationId) return;
    
    try {
        const response = await fetch(`${API_BASE}/investigations/${currentInvestigationId}/export?format=json`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `investigation_${currentInvestigationId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        showAlert('报告导出成功', 'success');
        
    } catch (error) {
        console.error('导出报告失败:', error);
        showAlert(`导出报告失败: ${error.message}`, 'danger');
    }
}

/**
 * 显示提示信息
 */
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // 自动关闭
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

/**
 * HTML转义
 */
function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * 格式化日期
 */
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN');
} 

async function handleTextImport(event) {
	event.preventDefault();
	const textarea = document.getElementById('personText');
	const text = textarea.value.trim();
	if (!text) {
		showAlert('请输入人物资料文本', 'danger');
		return;
	}
	const btn = event.target.querySelector('button[type="submit"]');
	const original = btn.innerHTML;
	btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>正在导入...';
	btn.disabled = true;
	try {
		const res = await fetch(`${API_BASE}/persons/import_text`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text })
		});
		        if (!res.ok) {
            let errText = `HTTP ${res.status}`;
            try {
                const errBody = await res.json();
                if (errBody && (errBody.detail || errBody.message)) {
                    errText += ` - ${errBody.detail || errBody.message}`;
                }
            } catch {}
            throw new Error(errText);
        }
        const data = await res.json();
        showAlert(`已导入人物：${escapeHtml(data.person.name)} (ID: ${data.person.id})`, 'success');
        textarea.value = '';
        // 刷新人物列表
        await refreshPersons();
    } catch (e) {
		console.error('导入失败', e);
		showAlert(`导入失败：${e.message}`, 'danger');
	} finally {
		btn.innerHTML = original;
		btn.disabled = false;
	}
} 

async function refreshPersons() {
	try {
		console.log('开始刷新人物列表，当前视图:', currentView);
		const res = await fetch(`${API_BASE}/persons`);
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		const data = await res.json();
		persons = Array.isArray(data.items) ? data.items : [];
		console.log('获取到人物数据:', persons.length, '个人物');
		
			// 总是渲染人物列表，不管当前视图是什么
	// 因为武器库页面可能已经显示了，但currentView可能还没更新
	renderPersons();
	} catch (e) {
		console.error('获取人物列表失败:', e);
		// 即使出错也要渲染空状态
		persons = [];
		renderPersons();
	}
}

function renderPersons() {
	const container = document.getElementById('personsContainer');
	if (!container) {
		console.log('⚠️ personsContainer元素不存在，跳过渲染');
		return;
	}
	
	console.log('渲染人物列表，人物数量:', persons ? persons.length : 0);
	console.log('人物数据:', persons);
	if (!persons || persons.length === 0) {
		container.innerHTML = `
			<div class="text-center text-muted py-4">
				<i class="fas fa-id-card fa-3x mb-3"></i>
				<p>暂无导入人物</p>
				<p class="small">在左侧"导入人物资料"成功后，这里会显示人物卡片</p>
			</div>
		`;
		return;
	}
	const html = persons.map((p, idx) => {
		const name = escapeHtml(p.name || '未知姓名');
		const email = p.email || '未知';
		const phone = p.phone || '未知';
		const job = p.current_job || '未知职位';
		const company = p.current_company || '未知公司';
		const created = p.created_at ? formatDateTime(p.created_at) : '';
		const avatar = p.photo_url ? `<img src="${p.photo_url}" class="rounded me-2" style="width:36px;height:36px;object-fit:cover;">` : `<span class="badge bg-secondary me-2" style="width:36px;height:36px;display:inline-flex;align-items:center;justify-content:center;"><i class="fas fa-user"></i></span>`;
		return `
			<div class="card mb-3">
				<div class="card-body">
					<div class="d-flex justify-content-between align-items-start">
						<div>
							<h6 class="card-title mb-1">${avatar}${name}</h6>
							<p class="mb-1 small text-muted">${company} · ${job}</p>
							<p class="mb-1 small">邮箱：${email} ｜ 电话：${phone}</p>
							${created ? `<p class="mb-0 small text-muted">导入时间：${created}</p>` : ''}
						</div>
						<div>
							<button class="btn btn-outline-primary btn-sm me-2" onclick="viewPerson(${idx})">
								<i class="fas fa-eye me-1"></i> 查看详情
							</button>
							<button class="btn btn-outline-danger btn-sm" onclick="deletePerson('${persons[idx].id}')">
								<i class="fas fa-trash me-1"></i> 删除
							</button>
						</div>
					</div>
				</div>
			</div>
		`;
	}).join('');
	container.innerHTML = html;
}

function viewPerson(idx) {
	const p = persons[idx];
	if (!p) {
		console.error('Person not found at index:', idx);
		return;
	}
	const detailsEl = document.getElementById('personDetails');
	const skills = (p.skills || []).map(s => `<span class="badge bg-secondary me-1">${escapeHtml(s)}</span>`).join('');
	const education = (p.education || []).map(e => {
		const inst = escapeHtml(e.institution || e.text || '');
		const degree = escapeHtml(e.degree || '');
		return `<li class="list-group-item">${degree ? `<strong>${degree}</strong> @ ` : ''}${inst}</li>`;
	}).join('');
	const sources = (p.data_sources || []).map(s => `<li class="list-group-item">${escapeHtml(s.type || 'source')} ${escapeHtml(s.timestamp || '')}</li>`).join('');
	const social = p.social_profiles ? Object.entries(p.social_profiles).map(([k,v]) => `<li class="list-group-item"><strong>${escapeHtml(k)}:</strong> <a href="${v}" target="_blank">${v}</a></li>`).join('') : '';
	const photo = p.photo_url ? `<img id="personPhoto" src="${p.photo_url}" alt="photo" class="img-thumbnail mb-3" style="max-height:160px">` : '<div id="personPhotoHolder" class="text-muted mb-2">未设置照片</div>';
	const html = `
		<div class="d-flex justify-content-between align-items-center mb-2">
			<h6 class="mb-0">基本信息</h6>
			<button class="btn btn-sm btn-outline-primary" onclick="enableEditPerson()"><i class="fas fa-edit me-1"></i> 编辑</button>
		</div>
		<div id="personView">
			<div class="row">
				<div class="col-md-3">${photo}</div>
				<div class="col-md-9">
					<table class="table table-sm">
						<tr><td>姓名</td><td>${escapeHtml(p.name || '')}</td></tr>
						<tr><td>邮箱</td><td>${p.email || '未知'}</td></tr>
						<tr><td>电话(主)</td><td>${p.phone || '未知'}</td></tr>
						<tr><td>电话(多个)</td><td>${(p.phones||[]).length ? (p.phones||[]).map(escapeHtml).join('，') : '无'}</td></tr>
						<tr><td>地址(主)</td><td>${p.address || '未知'}</td></tr>
						<tr><td>户籍地</td><td>${p.hukou_place || '未知'}</td></tr>
						<tr><td>户籍地地址</td><td>${p.hukou_address || '未知'}</td></tr>
						<tr><td>当前职位</td><td>${p.current_job || '未知'}</td></tr>
						<tr><td>当前公司</td><td>${p.current_company || '未知'}</td></tr>
					</table>
					${(p.id_numbers||[]).length ? `<h6>身份证号</h6><ul class="list-group list-group-flush mb-3">${(p.id_numbers||[]).map(n=>`<li class=\"list-group-item\">${escapeHtml(n)}</li>`).join('')}</ul>` : ''}
					${(p.delivery_addresses||[]).length ? `<h6>快递地址</h6><ul class="list-group list-group-flush mb-3">${(p.delivery_addresses||[]).map(a=>`<li class=\"list-group-item\">${escapeHtml(a)}</li>`).join('')}</ul>` : ''}
					${skills ? `<h6>技能</h6><div class="mb-3">${skills}</div>` : ''}
					${education ? `<h6>教育经历</h6><ul class="list-group list-group-flush mb-3">${education}</ul>` : ''}
					${renderCustomAttributes(p.custom_attributes)}
					${renderHouseholdMembers(p.relationships)}
					<h6>社交资料</h6>
					<table class="table table-sm">
						<tr><td>微信</td><td>${p.wechat_id || '-'}</td></tr>
						<tr><td>QQ</td><td>${p.qq_id || '-'}</td></tr>
						<tr><td>微博</td><td>${p.weibo_id || '-'}</td></tr>
						<tr><td>抖音</td><td>${p.douyin_id || '-'}</td></tr>
						<tr><td>小红书</td><td>${p.xhs_id || '-'}</td></tr>
						<tr><td>Gitee</td><td>${p.gitee_username || '-'}</td></tr>
						<tr><td>LinkedIn</td><td>${p.linkedin_url ? `<a href=\"${p.linkedin_url}\" target=\"_blank\">${p.linkedin_url}</a>` : '-'}</td></tr>
					</table>
					${social ? `<div class="small text-muted">其他：<ul class="list-group list-group-flush">${social}</ul></div>` : ''}
					${p.raw_text ? `<h6>原始文本</h6><pre class="p-3 bg-dark text-light" style="white-space:pre-wrap; border-radius:6px;">${escapeHtml(p.raw_text)}</pre>` : ''}
				</div>
			</div>
		</div>
		<div id="personEdit" style="display:none">
			<div class="row g-3">
				<div class="col-md-6"><label class="form-label">姓名</label><input class="form-control" id="edit_name" value="${escapeHtml(p.name||'')}"></div>
				<div class="col-md-6"><label class="form-label">照片URL</label><input class="form-control" id="edit_photo_url" value="${escapeHtml(p.photo_url||'')}"></div>
				<div class="col-md-12"><label class="form-label">上传照片</label>
					<div class="d-flex align-items-center gap-2">
						<input type="file" class="form-control" id="upload_photo_file" accept="image/*" style="max-width: 70%">
						<button class="btn btn-outline-primary" onclick="uploadPersonPhoto('${p.id}')"><i class="fas fa-upload me-1"></i> 上传</button>
					</div>
					<div class="form-text">选择本地图片上传，系统会自动保存并更新照片</div>
				</div>
				<div class="col-md-6"><label class="form-label">邮箱</label><input class="form-control" id="edit_email" value="${escapeHtml(p.email||'')}"></div>
				<div class="col-md-6"><label class="form-label">电话(主)</label><input class="form-control" id="edit_phone" value="${escapeHtml(p.phone||'')}"></div>
				<div class="col-md-6"><label class="form-label">电话(多个, 每行一个)</label><textarea class="form-control" id="edit_phones" rows="3">${(p.phones||[]).join('\n')}</textarea></div>
				<div class="col-md-6"><label class="form-label">身份证号(多个, 每行一个)</label><textarea class="form-control" id="edit_id_numbers" rows="3">${(p.id_numbers||[]).join('\n')}</textarea></div>
				<div class="col-md-12"><label class="form-label">地址(主)</label><input class="form-control" id="edit_address" value="${escapeHtml(p.address||'')}"></div>
				<div class="col-md-6"><label class="form-label">快递地址(多个, 每行一个)</label><textarea class="form-control" id="edit_delivery_addresses" rows="3">${(p.delivery_addresses||[]).join('\n')}</textarea></div>
				<div class="col-md-6"><label class="form-label">户籍地/户籍地址</label>
					<input class="form-control mb-2" id="edit_hukou_place" placeholder="户籍地" value="${escapeHtml(p.hukou_place||'')}">
					<input class="form-control" id="edit_hukou_address" placeholder="户籍地地址" value="${escapeHtml(p.hukou_address||'')}">
				</div>
				<div class="col-md-6"><label class="form-label">当前职位</label><input class="form-control" id="edit_current_job" value="${escapeHtml(p.current_job||'')}"></div>
				<div class="col-md-6"><label class="form-label">当前公司</label><input class="form-control" id="edit_current_company" value="${escapeHtml(p.current_company||'')}"></div>
				<div class="col-md-6"><label class="form-label">微信</label><input class="form-control" id="edit_wechat_id" value="${escapeHtml(p.wechat_id||'')}"></div>
				<div class="col-md-6"><label class="form-label">QQ</label><input class="form-control" id="edit_qq_id" value="${escapeHtml(p.qq_id||'')}"></div>
				<div class="col-md-6"><label class="form-label">微博</label><input class="form-control" id="edit_weibo_id" value="${escapeHtml(p.weibo_id||'')}"></div>
				<div class="col-md-6"><label class="form-label">抖音</label><input class="form-control" id="edit_douyin_id" value="${escapeHtml(p.douyin_id||'')}"></div>
				<div class="col-md-6"><label class="form-label">小红书</label><input class="form-control" id="edit_xhs_id" value="${escapeHtml(p.xhs_id||'')}"></div>
				<div class="col-md-6"><label class="form-label">Gitee</label><input class="form-control" id="edit_gitee_username" value="${escapeHtml(p.gitee_username||'')}"></div>
				<div class="col-md-6"><label class="form-label">LinkedIn</label><input class="form-control" id="edit_linkedin_url" value="${escapeHtml(p.linkedin_url||'')}"></div>
			</div>
			<div class="mt-3 text-end">
				<button class="btn btn-secondary me-2" onclick="cancelEditPerson()">取消</button>
				<button class="btn btn-primary" onclick="saveEditPerson('${p.id}')">保存</button>
			</div>
		</div>
	`;
	detailsEl.innerHTML = html;
	try {
		const modal = new bootstrap.Modal(document.getElementById('personModal'));
		modal.show();
	} catch (error) {
		console.error('Error showing modal:', error);
	}
}

function enableEditPerson() {
	const view = document.getElementById('personView');
	const edit = document.getElementById('personEdit');
	if (!edit) return;
	view && (view.style.display = 'none');
	edit.style.display = '';
}

function cancelEditPerson() {
	document.getElementById('personEdit').style.display = 'none';
	document.getElementById('personView').style.display = '';
}

function serializeCustomRows() { return {}; }

async function saveEditPerson(personId) {
	const parseLines = (id)=>{
		const el=document.getElementById(id);
		if(!el) return [];
		return el.value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean);
	};
	const payload = {
		name: document.getElementById('edit_name').value.trim(),
		photo_url: document.getElementById('edit_photo_url').value.trim(),
		email: document.getElementById('edit_email').value.trim(),
		phone: document.getElementById('edit_phone').value.trim(),
		phones: parseLines('edit_phones'),
		id_numbers: parseLines('edit_id_numbers'),
		address: document.getElementById('edit_address').value.trim(),
		delivery_addresses: parseLines('edit_delivery_addresses'),
		hukou_place: document.getElementById('edit_hukou_place').value.trim(),
		hukou_address: document.getElementById('edit_hukou_address').value.trim(),
		current_job: document.getElementById('edit_current_job').value.trim(),
		current_company: document.getElementById('edit_current_company').value.trim(),
		wechat_id: document.getElementById('edit_wechat_id').value.trim(),
		qq_id: document.getElementById('edit_qq_id').value.trim(),
		weibo_id: document.getElementById('edit_weibo_id').value.trim(),
		douyin_id: document.getElementById('edit_douyin_id').value.trim(),
		xhs_id: document.getElementById('edit_xhs_id').value.trim(),
		gitee_username: document.getElementById('edit_gitee_username').value.trim(),
		linkedin_url: document.getElementById('edit_linkedin_url').value.trim(),
		// 其他社交与自定义字段取消，不再提交
		social_profiles: {},
		custom_attributes: {}
	};
	try {
		const res = await fetch(`${API_BASE}/persons/${personId}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload)
		});
		if (!res.ok) {
			let msg = `HTTP ${res.status}`;
			try { const eb = await res.json(); if (eb && (eb.detail||eb.message)) msg += ` - ${eb.detail||eb.message}`; } catch {}
			throw new Error(msg);
		}
		showAlert('保存成功', 'success');
		await refreshPersons();
		cancelEditPerson();
	} catch (e) {
		console.error('保存人物失败:', e);
		showAlert(`保存失败：${e.message}`, 'danger');
	}
} 

async function uploadPersonPhoto(personId) {
	const fileInput = document.getElementById('upload_photo_file');
	if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
		showAlert('请先选择要上传的图片', 'danger');
		return;
	}
	const formData = new FormData();
	formData.append('file', fileInput.files[0]);
	try {
		const res = await fetch(`${API_BASE}/persons/${personId}/photo`, {
			method: 'POST',
			body: formData
		});
		if (!res.ok) {
			let msg = `HTTP ${res.status}`;
			try { const eb = await res.json(); if (eb && (eb.detail||eb.message)) msg += ` - ${eb.detail||eb.message}`; } catch {}
			throw new Error(msg);
		}
		const data = await res.json();
		// 更新预览
		const img = document.getElementById('personPhoto');
		if (img) { img.src = data.photo_url; }
		const holder = document.getElementById('personPhotoHolder');
		if (holder) { holder.outerHTML = `<img id="personPhoto" src="${data.photo_url}" class="img-thumbnail mb-3" style="max-height:160px">`; }
		showAlert('图片上传成功', 'success');
		await refreshPersons();
	} catch (e) {
		console.error('上传照片失败:', e);
		showAlert(`上传失败：${e.message}`, 'danger');
	}
} 



/**
 * 清空全面档案分析表单
 */
function clearAnalysisForm() {
    const form = document.getElementById('comprehensiveAnalysisForm');
    if (form) {
        form.reset();
        showAlert('表单已清空', 'info');
    }
}

/**
 * 刷新档案列表
 */
async function refreshProfileList() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        const select = document.getElementById('profileImportSelect');
        if (select) {
            // 保留第一个选项
            select.innerHTML = '<option value="">请选择要导入的人物档案...</option>';
            
            // 添加档案选项
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                select.appendChild(option);
            });
            
            showAlert(`已加载 ${profiles.length} 个人物档案`, 'success');
        }
    } catch (error) {
        console.error('刷新档案列表失败:', error);
        showAlert(`刷新失败：${error.message}`, 'danger');
    }
}

/**
 * 导入档案数据到表单
 */
async function importProfileData() {
    const select = document.getElementById('profileImportSelect');
    const selectedId = select.value;
    
    if (!selectedId) return;
    
    try {
        const response = await fetch(`${API_BASE}/persons/${selectedId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 填充表单字段
        fillFormWithProfile(profile);
        
        showAlert(`已导入档案：${profile.name || '未知姓名'}`, 'success');
        
    } catch (error) {
        console.error('导入档案失败:', error);
        showAlert(`导入失败：${error.message}`, 'danger');
    }
}

/**
 * 用档案数据填充表单
 */
function fillFormWithProfile(profile) {
    const form = document.getElementById('comprehensiveAnalysisForm');
    if (!form) return;
    
    // 基础身份信息
    setFormValue('name', profile.name);
    setFormValue('gender', profile.gender);
    setFormValue('age', profile.age);
    setFormValue('birth_date', profile.birth_date);
    setFormValue('id_number', profile.id_numbers ? profile.id_numbers.join(', ') : '');
    setFormValue('hukou_place', profile.hukou_place);
    setFormValue('current_address', profile.address);
    
    // 教育与专业背景
    setFormValue('skills', profile.skills ? profile.skills.join(', ') : '');
    
    // 职业发展轨迹
    setFormValue('current_job', profile.current_job);
    setFormValue('current_company', profile.current_company);
    
    // 社交媒体
    setFormValue('wechat_id', profile.wechat_id);
    setFormValue('qq_id', profile.qq_id);
    setFormValue('weibo_id', profile.weibo_id);
    setFormValue('douyin_id', profile.douyin_id);
    setFormValue('xhs_id', profile.xhs_id);
    
    // 联系方式
    setFormValue('email', profile.email);
    setFormValue('phone', profile.phone);
    
    // 其他字段
    if (profile.custom_attributes) {
        // 尝试从自定义属性中提取更多信息
        const custom = profile.custom_attributes;
        if (custom.education_level) setFormValue('education_level', custom.education_level);
        if (custom.university) setFormValue('university', custom.university);
        if (custom.major) setFormValue('major', custom.major);
        if (custom.languages) setFormValue('languages', custom.languages);
        if (custom.industry) setFormValue('industry', custom.industry);
        if (custom.work_years) setFormValue('work_years', custom.work_years);
        if (custom.career_path) setFormValue('career_path', custom.career_path);
        if (custom.income_level) setFormValue('income_level', custom.income_level);
        if (custom.consumption_ability) setFormValue('consumption_ability', custom.consumption_ability);
        if (custom.investment_preference) setFormValue('investment_preference', custom.investment_preference);
        if (custom.consumption_areas) setFormValue('consumption_areas', custom.consumption_areas);
        if (custom.online_activity) setFormValue('online_activity', custom.online_activity);
        if (custom.social_circle) setFormValue('social_circle', custom.social_circle);
        if (custom.social_frequency) setFormValue('social_frequency', custom.social_frequency);
        if (custom.social_activities) setFormValue('social_activities', custom.social_activities);
        if (custom.relationship_handling) setFormValue('relationship_handling', custom.relationship_handling);
        if (custom.openness) setFormValue('openness', custom.openness);
        if (custom.conscientiousness) setFormValue('conscientiousness', custom.conscientiousness);
        if (custom.extraversion) setFormValue('extraversion', custom.extraversion);
        if (custom.decision_style) setFormValue('decision_style', custom.decision_style);
        if (custom.sleep_schedule) setFormValue('sleep_schedule', custom.sleep_schedule);
        if (custom.exercise_habit) setFormValue('exercise_habit', custom.exercise_habit);
        if (custom.time_management) setFormValue('time_management', custom.time_management);
        if (custom.hobbies) setFormValue('hobbies', custom.hobbies);
        if (custom.emotional_triggers) setFormValue('emotional_triggers', custom.emotional_triggers);
        if (custom.sensitive_topics) setFormValue('sensitive_topics', custom.sensitive_topics);
        if (custom.stress_response) setFormValue('stress_response', custom.stress_response);
        if (custom.security_sources) setFormValue('security_sources', custom.security_sources);
        if (custom.trust_building) setFormValue('trust_building', custom.trust_building);
        if (custom.communication_preference) setFormValue('communication_preference', custom.communication_preference);
        if (custom.persuasion_technique) setFormValue('persuasion_technique', custom.persuasion_technique);
        if (custom.common_topics) setFormValue('common_topics', custom.common_topics);
        if (custom.timeline) setFormValue('timeline', custom.timeline);
        if (custom.behavior_observations) setFormValue('behavior_observations', custom.behavior_observations);
        if (custom.information_sources) setFormValue('information_sources', custom.information_sources);
        if (custom.analysis_notes) setFormValue('analysis_notes', custom.analysis_notes);
    }
    
    // 处理原始文本
    if (profile.raw_text) {
        setFormValue('analysis_notes', `原始导入文本：\n${profile.raw_text}\n\n${getFormValue('analysis_notes') || ''}`);
    }
}

/**
 * 设置表单字段值
 */
function setFormValue(fieldName, value) {
    if (!value) return;
    
    const field = document.querySelector(`[name="${fieldName}"]`);
    if (field) {
        if (field.type === 'select-one') {
            // 对于下拉框，尝试找到最匹配的选项
            const options = Array.from(field.options);
            const bestMatch = options.find(option => 
                option.value === value || 
                option.text.includes(value) ||
                value.includes(option.value)
            );
            if (bestMatch) {
                field.value = bestMatch.value;
            }
        } else {
            field.value = value;
        }
    }
}

/**
 * 获取表单字段值
 */
function getFormValue(fieldName) {
    const field = document.querySelector(`[name="${fieldName}"]`);
    return field ? field.value : '';
}

/**
 * 清空所有表单数据
 */
function clearAllFormData() {
    const form = document.getElementById('comprehensiveAnalysisForm');
    if (form) {
        form.reset();
    }
    
    const select = document.getElementById('profileImportSelect');
    if (select) {
        select.value = '';
    }
    
    showAlert('所有表单数据已清空', 'info');
}

/**
 * 检查轰炸目标
 */
async function checkBombingTarget() {
    const select = document.getElementById('bombingTargetSelect');
    const targetId = select.value;
    
    if (!targetId) {
        document.getElementById('bombingTargetInfo').style.display = 'none';
        document.getElementById('startBombingBtn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 检查是否有本人手机号
        const hasPersonalPhone = profile.phone && profile.phone.trim() !== '';
        
        if (hasPersonalPhone) {
            // 显示目标信息
            const targetInfo = document.getElementById('bombingTargetInfo');
            const targetDetails = document.getElementById('bombingTargetDetails');
            
            targetDetails.innerHTML = `
                <p><strong>姓名：</strong>${profile.name || '未知'}</p>
                <p><strong>手机号：</strong>${profile.phone || '未知'}</p>
                <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
                <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
                <div class="alert alert-success mt-2">
                    <i class="fas fa-check-circle me-1"></i>
                    <strong>验证通过：</strong>该人物档案包含本人手机号，可以进行攻击
                </div>
            `;
            
            targetInfo.style.display = 'block';
            document.getElementById('startBombingBtn').disabled = false;
            
        } else {
            // 没有本人手机号
            const targetInfo = document.getElementById('bombingTargetInfo');
            const targetDetails = document.getElementById('bombingTargetDetails');
            
            targetDetails.innerHTML = `
                <p><strong>姓名：</strong>${profile.name || '未知'}</p>
                <p><strong>手机号：</strong>${profile.phone || '无'}</p>
                <div class="alert alert-danger mt-2">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <strong>验证失败：</strong>该人物档案不包含本人手机号，无法进行攻击
                </div>
            `;
            
            targetInfo.style.display = 'block';
            document.getElementById('startBombingBtn').disabled = true;
        }
        
    } catch (error) {
        console.error('检查轰炸目标失败:', error);
        showAlert(`检查失败：${error.message}`, 'danger');
    }
}

/**
 * 开始轰炸攻击
 */
async function startBombing() {
    const targetId = document.getElementById('bombingTargetSelect').value;
    const smsBombing = document.getElementById('smsBombing').checked;
    const phoneBombing = document.getElementById('phoneBombing').checked;
    const intensity = document.getElementById('bombingIntensity').value;
    const interval = document.getElementById('bombingInterval').value;
    
    if (!targetId) {
        showAlert('请先选择目标人物', 'danger');
        return;
    }
    
    if (!smsBombing && !phoneBombing) {
        showAlert('请至少选择一种攻击类型', 'danger');
        return;
    }
    
    try {
        // 获取目标手机号
        const targetPhone = await getTargetPhone(targetId);
        if (!targetPhone) {
            showAlert('目标人物没有有效的手机号', 'danger');
            return;
        }
        
        // 检查频率限制
        const rateLimitResponse = await fetch(`${API_BASE}/attack/sms/rate-limit/${targetPhone}`);
        if (rateLimitResponse.ok) {
            const rateLimitData = await rateLimitResponse.json();
            if (!rateLimitData.can_send) {
                const rateInfo = rateLimitData.rate_info;
                showAlert(`频率限制：每分钟${rateInfo.minute_count}/${rateInfo.minute_limit}，每小时${rateInfo.hour_count}/${rateInfo.hour_limit}，每天${rateInfo.day_count}/${rateInfo.day_limit}`, 'warning');
                return;
            }
        }
        
        // 显示攻击状态
        document.getElementById('bombingStatus').style.display = 'block';
        document.getElementById('startBombingBtn').style.display = 'none';
        document.getElementById('stopBombingBtn').style.display = 'inline-block';
        
        // 准备攻击参数
        const attackParams = {
            target_phone: targetPhone,
            intensity: intensity,
            interval: parseInt(interval)
        };
        
        // 调用后端API开始攻击
        const response = await fetch(`${API_BASE}/attack/sms`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(attackParams)
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // 保存攻击ID用于状态查询
            window.currentAttackId = result.attack_id;
            
            // 开始状态轮询
            startAttackStatusPolling(result.attack_id);
            
            showAlert(`攻击启动成功！目标：${result.total_count}次，间隔：${interval}秒`, 'success');
        } else {
            throw new Error(result.error || '启动攻击失败');
        }
        
    } catch (error) {
        console.error('启动攻击失败:', error);
        showAlert(`启动攻击失败: ${error.message}`, 'danger');
        
        // 重置状态
        resetBombingStatus();
    }
}

/**
 * 获取目标手机号
 */
async function getTargetPhone(targetId) {
    try {
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 优先使用主手机号，如果没有则使用第一个手机号
        if (profile.phone) {
            return profile.phone;
        } else if (profile.phones && profile.phones.length > 0) {
            return profile.phones[0];
        }
        
        return null;
        
    } catch (error) {
        console.error('获取目标信息失败:', error);
        return null;
    }
}

/**
 * 开始攻击状态轮询
 */
function startAttackStatusPolling(attackId) {
    if (window.attackStatusInterval) {
        clearInterval(window.attackStatusInterval);
    }
    
    window.attackStatusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/attack/sms/${attackId}/status`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const status = await response.json();
            updateAttackProgress(status);
            
            // 如果攻击完成或失败，停止轮询
            if (['completed', 'failed', 'stopped'].includes(status.status)) {
                clearInterval(window.attackStatusInterval);
                window.attackStatusInterval = null;
                
                if (status.status === 'completed') {
                    showAlert(`攻击完成！已执行 ${status.current_count} 次攻击`, 'success');
                } else if (status.status === 'failed') {
                    showAlert(`攻击失败: ${status.error || '未知错误'}`, 'danger');
                } else {
                    showAlert(`攻击已停止，完成: ${status.current_count} 次`, 'info');
                }
                
                // 延迟重置状态
                setTimeout(resetBombingStatus, 2000);
            }
            
        } catch (error) {
            console.error('获取攻击状态失败:', error);
        }
    }, 1000); // 每秒查询一次状态
}

/**
 * 更新攻击进度
 */
function updateAttackProgress(status) {
    const progressBar = document.querySelector('#bombingStatus .progress-bar');
    const progressText = document.getElementById('bombingProgress');
    
    if (progressBar && progressText) {
        const progress = (status.current_count / status.total_count) * 100;
        progressBar.style.width = progress + '%';
        progressText.textContent = `攻击进度：${status.current_count}/${status.total_count} (${Math.round(progress)}%)`;
    }
}

/**
 * 重置轰炸状态
 */
function resetBombingStatus() {
    document.getElementById('bombingStatus').style.display = 'none';
    document.getElementById('startBombingBtn').style.display = 'inline-block';
    document.getElementById('stopBombingBtn').style.display = 'none';
    
    // 重置进度条
    const progressBar = document.querySelector('#bombingStatus .progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
    }
}

/**
 * 停止轰炸攻击
 */
async function stopBombing() {
    try {
        if (!window.currentAttackId) {
            showAlert('没有正在进行的攻击', 'warning');
            return;
        }
        
        // 调用后端API停止攻击
        const response = await fetch(`${API_BASE}/attack/sms/${window.currentAttackId}/stop`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // 停止状态轮询
            if (window.attackStatusInterval) {
                clearInterval(window.attackStatusInterval);
                window.attackStatusInterval = null;
            }
            
            showAlert(result.message, 'info');
            
            // 延迟重置状态
            setTimeout(resetBombingStatus, 1000);
        } else {
            throw new Error(result.error || '停止攻击失败');
        }
        
    } catch (error) {
        console.error('停止攻击失败:', error);
        showAlert(`停止攻击失败: ${error.message}`, 'danger');
    }
}

/**
 * 检查模拟目标
 */
async function checkSimulationTarget() {
    const select = document.getElementById('simulationTargetSelect');
    const targetId = select.value;
    
    if (!targetId) {
        document.getElementById('simulationTargetInfo').style.display = 'none';
        document.getElementById('startSimulationBtn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 显示目标档案信息
        const targetInfo = document.getElementById('simulationTargetInfo');
        const targetDetails = document.getElementById('simulationTargetDetails');
        
        let detailsHtml = `
            <p><strong>姓名：</strong>${profile.name || '未知'}</p>
            <p><strong>年龄：</strong>${profile.age || '未知'}</p>
            <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
            <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
        `;
        
        // 显示弱点分析相关信息
        if (profile.custom_attributes) {
            const custom = profile.custom_attributes;
            if (custom.emotional_triggers) {
                detailsHtml += `<p><strong>情感触发点：</strong>${custom.emotional_triggers}</p>`;
            }
            if (custom.sensitive_topics) {
                detailsHtml += `<p><strong>敏感话题：</strong>${custom.sensitive_topics}</p>`;
            }
            if (custom.weaknesses) {
                detailsHtml += `<p><strong>心理弱点：</strong>${custom.weaknesses}</p>`;
            }
        }
        
        // 检查是否有足够的分析数据
        const hasAnalysisData = profile.custom_attributes && Object.keys(profile.custom_attributes).length > 0;
        
        if (hasAnalysisData) {
            detailsHtml += `
                <div class="alert alert-success mt-2">
                    <i class="fas fa-check-circle me-1"></i>
                    <strong>档案完整：</strong>该人物档案包含足够的分析数据，可以进行沙盘模拟
                </div>
            `;
            document.getElementById('startSimulationBtn').disabled = false;
        } else {
            detailsHtml += `
                <div class="alert alert-warning mt-2">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    <strong>档案不完整：</strong>建议先在"全面档案分析"中完善该人物的心理分析数据
                </div>
            `;
            document.getElementById('startSimulationBtn').disabled = true;
        }
        
        targetDetails.innerHTML = detailsHtml;
        targetInfo.style.display = 'block';
        
    } catch (error) {
        console.error('检查模拟目标失败:', error);
        showAlert(`检查失败：${error.message}`, 'danger');
    }
}

/**
 * 开始沙盘模拟
 */
async function startSimulation() {
    const targetId = document.getElementById('simulationTargetSelect').value;
    const scenario = document.getElementById('simulationScenario').value;
    const strategy = document.getElementById('attackStrategy').value;
    const expectedOutcome = document.getElementById('expectedOutcome').value;
    
    if (!targetId || !scenario || !strategy || !expectedOutcome) {
        showAlert('请填写完整的模拟信息', 'danger');
        return;
    }
    
    // 显示模拟状态
    document.getElementById('simulationStatus').style.display = 'block';
    document.getElementById('startSimulationBtn').disabled = true;
    
    const progressBar = document.querySelector('#simulationStatus .progress-bar');
    const progressText = document.getElementById('simulationProgress');
    
    // 模拟分析进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        progressBar.style.width = progress + '%';
        
        if (progress < 30) {
            progressText.textContent = '正在分析目标心理特征...';
        } else if (progress < 60) {
            progressText.textContent = '正在评估攻击策略可行性...';
        } else if (progress < 90) {
            progressText.textContent = '正在生成模拟结果...';
        } else {
            progressText.textContent = '模拟分析完成！';
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            generateSimulationReport(targetId, scenario, strategy, expectedOutcome);
            
            // 重置按钮状态
            document.getElementById('startSimulationBtn').disabled = false;
        }
    }, 800);
}

/**
 * 生成模拟分析报告
 */
async function generateSimulationReport(targetId, scenario, strategy, expectedOutcome) {
    try {
        // 获取目标档案数据
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 生成模拟报告
        const report = generateAIReport(profile, scenario, strategy, expectedOutcome);
        
        // 显示报告
        const reportCard = document.getElementById('attackReportCard');
        const reportContent = document.getElementById('attackReportContent');
        
        reportContent.innerHTML = report;
        reportCard.style.display = 'block';
        
        showAlert('沙盘模拟分析完成！', 'success');
        
    } catch (error) {
        console.error('生成模拟报告失败:', error);
        showAlert(`报告生成失败：${error.message}`, 'danger');
    }
}

/**
 * 生成AI分析报告
 */
function generateAIReport(profile, scenario, strategy, expectedOutcome) {
    const scenarioNames = {
        'social_engineering': '社交工程攻击',
        'phishing': '钓鱼攻击',
        'information_gathering': '信息收集',
        'psychological_manipulation': '心理操控',
        'trust_building': '信任建立',
        'pressure_tactics': '压力战术'
    };
    
    const scenarioName = scenarioNames[scenario] || scenario;
    
    // 基于档案数据生成分析
    let analysis = '';
    let riskLevel = '中等';
    let successRate = '60%';
    let recommendations = '';
    
    if (profile.custom_attributes) {
        const custom = profile.custom_attributes;
        
        // 分析心理特征
        if (custom.emotional_triggers) {
            analysis += `<p><strong>情感触发点分析：</strong>${custom.emotional_triggers}</p>`;
        }
        if (custom.sensitive_topics) {
            analysis += `<p><strong>敏感话题识别：</strong>${custom.sensitive_topics}</p>`;
        }
        if (custom.weaknesses) {
            analysis += `<p><strong>心理弱点：</strong>${custom.weaknesses}</p>`;
        }
        
        // 评估风险等级
        if (custom.weaknesses && custom.weaknesses.includes('高')) {
            riskLevel = '高';
            successRate = '80%';
        } else if (custom.weaknesses && custom.weaknesses.includes('低')) {
            riskLevel = '低';
            successRate = '40%';
        }
    }
    
    // 生成建议
    if (scenario === 'social_engineering') {
        recommendations = `
            <li>利用目标的情感触发点建立初步联系</li>
            <li>通过共同话题建立信任关系</li>
            <li>逐步引导目标提供敏感信息</li>
        `;
    } else if (scenario === 'phishing') {
        recommendations = `
            <li>制作针对性的钓鱼内容</li>
            <li>利用目标的职业背景设计诱饵</li>
            <li>通过紧急情况制造压力</li>
        `;
    } else if (scenario === 'psychological_manipulation') {
        recommendations = `
            <li>识别并利用心理弱点</li>
            <li>通过对比效应影响判断</li>
            <li>利用从众心理施加影响</li>
        `;
    }
    
    return `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-primary">模拟场景</h6>
                <p>${scenarioName}</p>
                
                <h6 class="text-primary">攻击策略</h6>
                <p>${strategy}</p>
                
                <h6 class="text-primary">预期目标</h6>
                <p>${expectedOutcome}</p>
            </div>
            <div class="col-md-6">
                <h6 class="text-primary">风险评估</h6>
                <div class="alert alert-${riskLevel === '高' ? 'danger' : riskLevel === '中' ? 'warning' : 'info'}">
                    <strong>风险等级：</strong>${riskLevel}
                </div>
                
                <h6 class="text-primary">成功率预测</h6>
                <div class="alert alert-info">
                    <strong>预期成功率：</strong>${successRate}
                </div>
            </div>
        </div>
        
        <hr class="my-4">
        
        <h6 class="text-primary">目标心理特征分析</h6>
        ${analysis || '<p class="text-muted">暂无足够的心理分析数据</p>'}
        
        <h6 class="text-primary">攻击策略建议</h6>
        <ul>
            ${recommendations || '<li>建议先在"全面档案分析"中完善目标的心理分析数据</li>'}
        </ul>
        
        <h6 class="text-primary">潜在后果分析</h6>
        <div class="alert alert-warning">
            <ul class="mb-0">
                <li><strong>成功后果：</strong>${expectedOutcome}</li>
                <li><strong>失败风险：</strong>目标可能产生警惕，增加后续攻击难度</li>
                <li><strong>法律风险：</strong>请注意遵守相关法律法规</li>
            </ul>
        </div>
        
        <div class="text-center mt-4">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                本报告基于AI分析生成，仅供参考。实际执行时请谨慎评估风险。
            </small>
        </div>
    `;
}

/**
 * 开始OSINT搜索
 */
async function startOSINTSearch() {
    showAlert('OSINT搜索功能正在开发中', 'info');
}

/**
 * 处理全面档案分析表单提交
 */
async function handleComprehensiveAnalysisSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const data = {};
    
    // 收集表单数据
    for (let [key, value] of formData.entries()) {
        if (value.trim()) {
            data[key] = value.trim();
        }
    }
    
    if (Object.keys(data).length === 0) {
        showAlert('请至少填写一个字段', 'warning');
        return;
    }
    
    try {
        // 显示保存进度
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>保存中...';
        submitBtn.disabled = true;
        
        // 这里将来会调用保存API
        // const response = await fetch(`${API_BASE}/persons/comprehensive`, {
        //     method: 'POST',
        //     headers: {'Content-Type': 'application/json'},
        //     body: JSON.stringify(data)
        // });
        
        // 模拟保存延迟
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        showAlert('全面档案保存成功！', 'success');
        
        // 清空表单
        form.reset();
        
        // 切换到人物列表标签页
        const personsTab = document.getElementById('persons-tab');
        if (personsTab) {
            personsTab.click();
        }
        
    } catch (error) {
        console.error('保存失败:', error);
        showAlert(`保存失败：${error.message}`, 'danger');
    } finally {
        // 恢复按钮状态
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
}

/**
 * 删除人物档案
 */
async function deletePerson(personId) {
    if (!personId) return;
    if (!confirm('确认删除该人物档案吗？操作不可撤销。')) return;
    
    try {
        const res = await fetch(`${API_BASE}/persons/${personId}`, { method: 'DELETE' });
        if (!res.ok) {
            let msg = `HTTP ${res.status}`;
            try { const eb = await res.json(); if (eb && (eb.detail||eb.message)) msg += ` - ${eb.detail||eb.message}`; } catch {}
            throw new Error(msg);
        }
        showAlert('已删除人物档案', 'success');
        await refreshPersons();
    } catch (e) {
        console.error('删除人物失败:', e);
        showAlert(`删除失败：${e.message}`, 'danger');
    }
}

/**
 * 渲染自定义属性（生肖、星座等灵活字段）
 */
function renderCustomAttributes(customAttrs) {
    if (!customAttrs || Object.keys(customAttrs).length === 0) {
        return '';
    }
    
    const customFields = Object.entries(customAttrs)
        .filter(([key, value]) => key && value && key !== 'raw_text')
        .map(([key, value]) => {
            return `<tr><td><strong>${escapeHtml(key)}</strong></td><td>${escapeHtml(String(value))}</td></tr>`;
        }).join('');
    
    if (!customFields) return '';
    
    return `
        <h6><i class="fas fa-tags me-2"></i>个人信息</h6>
        <table class="table table-sm mb-3">
            ${customFields}
        </table>
    `;
}

/**
 * 渲染同户人信息 - 动态显示所有被识别的字段
 */
function renderHouseholdMembers(relationships) {
    if (!relationships || !Array.isArray(relationships)) {
        return '';
    }
    
    const householdMembers = relationships.filter(rel => rel.type === 'household_member');
    
    if (householdMembers.length === 0) {
        return '';
    }
    
    const memberCards = householdMembers.map(rel => {
        const member = rel.details || {};
        const name = rel.name || '未知';
        
        // 动态生成所有字段的显示
        const fieldRows = [];
        
        // 优先显示核心字段
        const priorityFields = ['姓名', '性别', '身份证号', '出生日期', '电话'];
        const displayedFields = new Set();
        
        // 先处理优先字段
        priorityFields.forEach(field => {
            if (member[field] && !displayedFields.has(field)) {
                const value = member[field];
                if (Array.isArray(value) && value.length > 0) {
                    fieldRows.push(`
                        <tr>
                            <td><strong style="color: #68d391;">${escapeHtml(field)}</strong></td>
                            <td style="color: #e2e8f0;">${value.map(v => escapeHtml(String(v))).join(', ')}</td>
                        </tr>
                    `);
                } else if (value && String(value).trim()) {
                    fieldRows.push(`
                        <tr>
                            <td><strong style="color: #68d391;">${escapeHtml(field)}</strong></td>
                            <td style="color: #e2e8f0;">${escapeHtml(String(value))}</td>
                        </tr>
                    `);
                }
                displayedFields.add(field);
            }
        });
        
        // 再处理其他所有字段
        Object.entries(member).forEach(([key, value]) => {
            if (!displayedFields.has(key) && key !== 'raw_text' && value) {
                if (Array.isArray(value) && value.length > 0) {
                    fieldRows.push(`
                        <tr>
                            <td><strong style="color: #68d391;">${escapeHtml(key)}</strong></td>
                            <td style="color: #e2e8f0;">${value.map(v => escapeHtml(String(v))).join(', ')}</td>
                        </tr>
                    `);
                } else if (value && String(value).trim()) {
                    fieldRows.push(`
                        <tr>
                            <td><strong style="color: #68d391;">${escapeHtml(key)}</strong></td>
                            <td style="color: #e2e8f0;">${escapeHtml(String(value))}</td>
                        </tr>
                    `);
                }
                displayedFields.add(key);
            }
        });
        
        // 如果没有字段，显示基本信息
        if (fieldRows.length === 0) {
            fieldRows.push(`
                <tr>
                    <td><strong style="color: #68d391;">姓名</strong></td>
                    <td style="color: #e2e8f0;">${escapeHtml(name)}</td>
                </tr>
            `);
        }
        
        return `
            <div class="card border-0 bg-dark mb-3" style="background: #2d3748 !important;">
                <div class="card-header bg-dark text-white p-2" style="background: #1a202c !important;">
                    <div class="d-flex justify-content-between align-items-center">
                        <h6 class="mb-0">
                            <i class="fas fa-user me-2"></i>${escapeHtml(name)}
                        </h6>
                        <span class="badge bg-secondary">同户人</span>
                    </div>
                </div>
                <div class="card-body p-3" style="background: #2d3748 !important;">
                    <table class="table table-sm table-borderless mb-0" style="color: #e2e8f0;">
                        <tbody>
                            ${fieldRows.join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }).join('');
    
    return `
        <h6 style="color: #68d391;"><i class="fas fa-users me-2"></i>同户人 (${householdMembers.length}人)</h6>
        <div class="mb-3">
            ${memberCards}
        </div>
    `;
}

/**
 * 刷新攻击目标选择框
 */
async function refreshAttackTargets() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        // 刷新轰炸目标选择框
        const bombingSelect = document.getElementById('bombingTargetSelect');
        if (bombingSelect) {
            bombingSelect.innerHTML = '<option value="">请选择目标人物...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.phone || '无手机号'} (${profile.current_company || '未知公司'})`;
                bombingSelect.appendChild(option);
            });
        }
        
        // 刷新模拟目标选择框
        const simulationSelect = document.getElementById('simulationTargetSelect');
        if (simulationSelect) {
            simulationSelect.innerHTML = '<option value="">请选择模拟目标...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                simulationSelect.appendChild(option);
            });
        }
        
        // 刷新画像目标选择框
        const profileSelect = document.getElementById('profileTargetSelect');
        if (profileSelect) {
            profileSelect.innerHTML = '<option value="">请选择目标人物...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                profileSelect.appendChild(option);
            });
        }
        
        // 刷新弱点分析目标选择框
        const weaknessSelect = document.getElementById('weaknessTargetSelect');
        if (weaknessSelect) {
            weaknessSelect.innerHTML = '<option value="">请选择分析目标...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                weaknessSelect.appendChild(option);
            });
        }
        
        console.log(`✅ 已加载 ${profiles.length} 个攻击目标`);
        
    } catch (error) {
        console.error('刷新攻击目标失败:', error);
        showAlert(`刷新攻击目标失败：${error.message}`, 'danger');
    }
}

/**
 * 检查画像目标
 */
async function checkProfileTarget() {
    const select = document.getElementById('profileTargetSelect');
    const targetId = select.value;
    
    if (!targetId) {
        document.getElementById('profileTargetInfo').style.display = 'none';
        document.getElementById('generateProfileBtn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 显示目标基本信息
        const targetInfo = document.getElementById('profileTargetInfo');
        const targetDetails = document.getElementById('profileTargetDetails');
        
        targetDetails.innerHTML = `
            <p><strong>姓名：</strong>${profile.name || '未知'}</p>
            <p><strong>年龄：</strong>${profile.age || '未知'}</p>
            <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
            <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
            <p><strong>技能：</strong>${profile.skills ? profile.skills.join(', ') : '未知'}</p>
        `;
        
        targetInfo.style.display = 'block';
        document.getElementById('generateProfileBtn').disabled = false;
        
    } catch (error) {
        console.error('检查画像目标失败:', error);
        showAlert(`检查失败：${error.message}`, 'danger');
    }
}

/**
 * 生成目标画像
 */
async function generateProfile() {
    const targetId = document.getElementById('profilePersonSelect').value;
    const profileType = document.getElementById('profileType').value;
    
    if (!targetId || !profileType) {
        showAlert('请选择目标人物和画像类型', 'danger');
        return;
    }
    
    // 显示生成状态
    document.getElementById('profileStatus').style.display = 'block';
    document.getElementById('generateProfileBtn').disabled = true;
    
    const progressBar = document.querySelector('#profileStatus .progress-bar');
    const progressText = document.getElementById('profileProgress');
    
    // 模拟生成进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 100) progress = 100;
        
        progressBar.style.width = progress + '%';
        
        if (progress < 30) {
            progressText.textContent = '正在收集目标信息...';
        } else if (progress < 60) {
            progressText.textContent = '正在分析行为模式...';
        } else if (progress < 90) {
            progressText.textContent = '正在生成画像报告...';
        } else {
            progressText.textContent = '画像生成完成！';
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            generateProfileReport(targetId, profileType);
            
            // 重置按钮状态
            document.getElementById('generateProfileBtn').disabled = false;
        }
    }, 600);
}

/**
 * 检查弱点分析目标
 */
async function checkWeaknessTarget() {
    const select = document.getElementById('weaknessTargetSelect');
    const targetId = select.value;
    
    if (!targetId) {
        document.getElementById('weaknessTargetInfo').style.display = 'none';
        document.getElementById('analyzeWeaknessBtn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 显示目标档案信息
        const targetInfo = document.getElementById('weaknessTargetInfo');
        const targetDetails = document.getElementById('weaknessTargetDetails');
        
        let detailsHtml = `
            <p><strong>姓名：</strong>${profile.name || '未知'}</p>
            <p><strong>年龄：</strong>${profile.age || '未知'}</p>
            <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
            <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
        `;
        
        // 显示弱点分析相关信息
        if (profile.custom_attributes) {
            const custom = profile.custom_attributes;
            if (custom.emotional_triggers) {
                detailsHtml += `<p><strong>情感触发点：</strong>${custom.emotional_triggers}</p>`;
            }
            if (custom.sensitive_topics) {
                detailsHtml += `<p><strong>敏感话题：</strong>${custom.sensitive_topics}</p>`;
            }
            if (custom.weaknesses) {
                detailsHtml += `<p><strong>已知弱点：</strong>${custom.weaknesses}</p>`;
            }
        }
        
        targetDetails.innerHTML = detailsHtml;
        targetInfo.style.display = 'block';
        document.getElementById('analyzeWeaknessBtn').disabled = false;
        
    } catch (error) {
        console.error('检查弱点分析目标失败:', error);
        showAlert(`检查失败：${error.message}`, 'danger');
    }
}

/**
 * 开始弱点分析
 */
async function analyzeWeakness() {
    const targetId = document.getElementById('weaknessTargetSelect').value;
    const psychological = document.getElementById('psychologicalWeakness').checked;
    const social = document.getElementById('socialWeakness').checked;
    const technical = document.getElementById('technicalWeakness').checked;
    const physical = document.getElementById('physicalWeakness').checked;
    
    if (!targetId) {
        showAlert('请先选择分析目标', 'danger');
        return;
    }
    
    if (!psychological && !social && !technical && !physical) {
        showAlert('请至少选择一个分析维度', 'danger');
        return;
    }
    
    // 显示分析状态
    document.getElementById('weaknessStatus').style.display = 'block';
    document.getElementById('analyzeWeaknessBtn').disabled = true;
    
    const progressBar = document.querySelector('#weaknessStatus .progress-bar');
    const progressText = document.getElementById('weaknessProgress');
    
    // 模拟分析进度
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        progressBar.style.width = progress + '%';
        
        if (progress < 25) {
            progressText.textContent = '正在分析心理特征...';
        } else if (progress < 50) {
            progressText.textContent = '正在分析社交行为...';
        } else if (progress < 75) {
            progressText.textContent = '正在识别技术弱点...';
        } else if (progress < 100) {
            progressText.textContent = '正在生成分析报告...';
        } else {
            progressText.textContent = '弱点分析完成！';
        }
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            generateWeaknessReport(targetId, { psychological, social, technical, physical });
            
            // 重置按钮状态
            document.getElementById('analyzeWeaknessBtn').disabled = false;
        }
    }, 700);
}

/**
 * 生成画像报告
 */
async function generateProfileReport(targetId, profileType) {
    try {
        // 获取目标档案数据
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 生成画像报告
        const report = generateProfileReportContent(profile, profileType);
        
        // 显示报告
        const reportCard = document.getElementById('profileReportCard');
        const reportContent = document.getElementById('profileReportContent');
        
        reportContent.innerHTML = report;
        reportCard.style.display = 'block';
        
        showAlert('目标画像生成完成！', 'success');
        
    } catch (error) {
        console.error('生成画像报告失败:', error);
        showAlert(`报告生成失败：${error.message}`, 'danger');
    }
}

/**
 * 生成弱点分析报告
 */
async function generateWeaknessReport(targetId, analysisDimensions) {
    try {
        // 获取目标档案数据
        const response = await fetch(`${API_BASE}/persons/${targetId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 生成弱点分析报告
        const report = generateWeaknessReportContent(profile, analysisDimensions);
        
        // 显示报告
        const reportCard = document.getElementById('attackReportCard');
        const reportContent = document.getElementById('attackReportContent');
        
        reportContent.innerHTML = report;
        reportCard.style.display = 'block';
        
        showAlert('弱点分析完成！', 'success');
        
    } catch (error) {
        console.error('生成弱点分析报告失败:', error);
        showAlert(`报告生成失败：${error.message}`, 'danger');
    }
}

/**
 * 生成画像报告内容
 */
function generateProfileReportContent(profile, profileType) {
    const typeNames = {
        'basic': '基础画像',
        'psychological': '心理画像',
        'behavioral': '行为画像',
        'social': '社交画像',
        'comprehensive': '综合画像'
    };
    
    const typeName = typeNames[profileType] || profileType;
    
    let analysis = '';
    let recommendations = '';
    
    // 基于档案数据生成分析
    if (profile.custom_attributes) {
        const custom = profile.custom_attributes;
        
        if (profileType === 'psychological' || profileType === 'comprehensive') {
            if (custom.emotional_triggers) {
                analysis += `<p><strong>情感触发点：</strong>${custom.emotional_triggers}</p>`;
            }
            if (custom.sensitive_topics) {
                analysis += `<p><strong>敏感话题：</strong>${custom.sensitive_topics}</p>`;
            }
            if (custom.weaknesses) {
                analysis += `<p><strong>心理弱点：</strong>${custom.weaknesses}</p>`;
            }
        }
        
        if (profileType === 'behavioral' || profileType === 'comprehensive') {
            if (custom.sleep_schedule) {
                analysis += `<p><strong>作息规律：</strong>${custom.sleep_schedule}</p>`;
            }
            if (custom.exercise_habit) {
                analysis += `<p><strong>运动习惯：</strong>${custom.exercise_habit}</p>`;
            }
            if (custom.time_management) {
                analysis += `<p><strong>时间管理：</strong>${custom.time_management}</p>`;
            }
        }
        
        if (profileType === 'social' || profileType === 'comprehensive') {
            if (custom.social_circle) {
                analysis += `<p><strong>社交圈层：</strong>${custom.social_circle}</p>`;
            }
            if (custom.social_frequency) {
                analysis += `<p><strong>社交频率：</strong>${custom.social_frequency}</p>`;
            }
            if (custom.relationship_handling) {
                analysis += `<p><strong>关系处理：</strong>${custom.relationship_handling}</p>`;
            }
        }
    }
    
    // 生成建议
    if (profileType === 'comprehensive') {
        recommendations = `
            <li>建议重点关注目标的心理弱点，利用情感触发点</li>
            <li>关注目标的社交行为模式，寻找社交工程突破口</li>
            <li>分析目标的行为习惯，制定针对性的攻击策略</li>
        `;
    } else if (profileType === 'psychological') {
        recommendations = `
            <li>深入分析目标的情感状态和压力点</li>
            <li>识别目标的心理防御机制</li>
            <li>制定心理操控策略</li>
        `;
    }
    
    return `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-primary">画像类型</h6>
                <p>${typeName}</p>
                
                <h6 class="text-primary">目标基本信息</h6>
                <p><strong>姓名：</strong>${profile.name || '未知'}</p>
                <p><strong>年龄：</strong>${profile.age || '未知'}</p>
                <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
                <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
            </div>
            <div class="col-md-6">
                <h6 class="text-primary">画像完整性</h6>
                <div class="alert alert-info">
                    <strong>数据完整度：</strong>${profile.custom_attributes && Object.keys(profile.custom_attributes).length > 0 ? '高' : '低'}
                </div>
                
                <h6 class="text-primary">分析深度</h6>
                <div class="alert alert-success">
                    <strong>分析深度：</strong>${profileType === 'comprehensive' ? '全面' : '专项'}
                </div>
            </div>
        </div>
        
        <hr class="my-4">
        
        <h6 class="text-primary">画像分析结果</h6>
        ${analysis || '<p class="text-muted">暂无足够的分析数据</p>'}
        
        <h6 class="text-primary">策略建议</h6>
        <ul>
            ${recommendations || '<li>建议先在"全面档案分析"中完善目标的分析数据</li>'}
        </ul>
        
        <div class="text-center mt-4">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                本画像基于AI分析生成，仅供参考。实际应用时请结合具体情况。
            </small>
        </div>
    `;
}

/**
 * 生成弱点分析报告内容
 */
function generateWeaknessReportContent(profile, analysisDimensions) {
    let analysis = '';
    let riskLevel = '中等';
    let recommendations = '';
    
    // 基于档案数据生成分析
    if (profile.custom_attributes) {
        const custom = profile.custom_attributes;
        
        if (analysisDimensions.psychological) {
            analysis += `<h6 class="text-warning">心理弱点分析</h6>`;
            if (custom.emotional_triggers) {
                analysis += `<p><strong>情感触发点：</strong>${custom.emotional_triggers}</p>`;
            }
            if (custom.sensitive_topics) {
                analysis += `<p><strong>敏感话题：</strong>${custom.sensitive_topics}</p>`;
            }
            if (custom.weaknesses) {
                analysis += `<p><strong>心理弱点：</strong>${custom.weaknesses}</p>`;
            }
            analysis += `<hr>`;
        }
        
        if (analysisDimensions.social) {
            analysis += `<h6 class="text-warning">社交弱点分析</h6>`;
            if (custom.social_circle) {
                analysis += `<p><strong>社交圈层：</strong>${custom.social_circle}</p>`;
            }
            if (custom.social_frequency) {
                analysis += `<p><strong>社交频率：</strong>${custom.social_frequency}</p>`;
            }
            if (custom.relationship_handling) {
                analysis += `<p><strong>关系处理：</strong>${custom.relationship_handling}</p>`;
            }
            analysis += `<hr>`;
        }
        
        if (analysisDimensions.technical) {
            analysis += `<h6 class="text-warning">技术弱点分析</h6>`;
            analysis += `<p><strong>技术技能：</strong>${profile.skills ? profile.skills.join(', ') : '未知'}</p>`;
            analysis += `<p><strong>在线活动：</strong>${custom.online_activity || '未知'}</p>`;
            analysis += `<hr>`;
        }
        
        if (analysisDimensions.physical) {
            analysis += `<h6 class="text-warning">物理弱点分析</h6>`;
            if (custom.sleep_schedule) {
                analysis += `<p><strong>作息规律：</strong>${custom.sleep_schedule}</p>`;
            }
            if (custom.exercise_habit) {
                analysis += `<p><strong>运动习惯：</strong>${custom.exercise_habit}</p>`;
            }
            analysis += `<hr>`;
        }
        
        // 评估风险等级
        if (custom.weaknesses && custom.weaknesses.includes('高')) {
            riskLevel = '高';
        } else if (custom.weaknesses && custom.weaknesses.includes('低')) {
            riskLevel = '低';
        }
    }
    
    // 生成建议
    if (analysisDimensions.psychological) {
        recommendations += `<li>利用心理弱点进行情感操控</li>`;
    }
    if (analysisDimensions.social) {
        recommendations += `<li>通过社交工程建立信任关系</li>`;
    }
    if (analysisDimensions.technical) {
        recommendations += `<li>利用技术弱点进行信息收集</li>`;
    }
    if (analysisDimensions.physical) {
        recommendations += `<li>利用作息规律制定攻击时机</li>`;
    }
    
    return `
        <div class="row">
            <div class="col-md-6">
                <h6 class="text-primary">分析维度</h6>
                <ul>
                    ${analysisDimensions.psychological ? '<li>心理弱点</li>' : ''}
                    ${analysisDimensions.social ? '<li>社交弱点</li>' : ''}
                    ${analysisDimensions.technical ? '<li>技术弱点</li>' : ''}
                    ${analysisDimensions.physical ? '<li>物理弱点</li>' : ''}
                </ul>
                
                <h6 class="text-primary">风险评估</h6>
                <div class="alert alert-${riskLevel === '高' ? 'danger' : riskLevel === '中' ? 'warning' : 'info'}">
                    <strong>风险等级：</strong>${riskLevel}
                </div>
            </div>
            <div class="col-md-6">
                <h6 class="text-primary">目标基本信息</h6>
                <p><strong>姓名：</strong>${profile.name || '未知'}</p>
                <p><strong>年龄：</strong>${profile.age || '未知'}</p>
                <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
                <p><strong>职位：</strong>${profile.current_job || '未知'}</p>
            </div>
        </div>
        
        <hr class="my-4">
        
        <h6 class="text-primary">弱点分析结果</h6>
        ${analysis || '<p class="text-muted">暂无足够的分析数据</p>'}
        
        <h6 class="text-primary">攻击策略建议</h6>
        <ul>
            ${recommendations || '<li>建议先在"全面档案分析"中完善目标的分析数据</li>'}
        </ul>
        
        <h6 class="text-primary">注意事项</h6>
        <div class="alert alert-warning">
            <ul class="mb-0">
                <li><strong>法律风险：</strong>请注意遵守相关法律法规</li>
                <li><strong>道德考虑：</strong>请确保攻击行为符合道德标准</li>
                <li><strong>风险评估：</strong>请充分评估攻击可能带来的后果</li>
            </ul>
        </div>
        
        <div class="text-center mt-4">
            <small class="text-muted">
                <i class="fas fa-info-circle me-1"></i>
                本分析基于AI生成，仅供参考。实际执行时请谨慎评估风险。
            </small>
        </div>
    `;
}

/**
 * 初始化主页攻击功能的目标选择框
 */
async function initializeMainPageAttackTargets() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        // 初始化主页攻击功能的所有目标选择框
        const selectIds = [
            'bombingTargetSelect',
            'simulationTargetSelect', 
            'profileTargetSelect',
            'weaknessTargetSelect'
        ];
        
        selectIds.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                select.innerHTML = '<option value="">请选择目标人物...</option>';
                profiles.forEach(profile => {
                    const option = document.createElement('option');
                    option.value = profile.id;
                    option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                    select.appendChild(option);
                });
            }
        });
        
        console.log(`✅ 主页攻击功能已初始化，加载了 ${profiles.length} 个目标`);
        
    } catch (error) {
        console.error('初始化主页攻击功能失败:', error);
        showAlert(`初始化失败：${error.message}`, 'danger');
    }
}



/**
 * 初始化攻击页面的目标选择框
 */
async function initializeAttackPageTargets() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        // 初始化攻击页面所有目标选择框
        const selectIds = [
            'bombingTargetSelect',
            'simulationTargetSelect', 
            'profileTargetSelect',
            'weaknessTargetSelect'
        ];
        
        selectIds.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (select) {
                select.innerHTML = '<option value="">请选择目标人物...</option>';
                profiles.forEach(profile => {
                    const option = document.createElement('option');
                    option.value = profile.id;
                    option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                    select.appendChild(option);
                });
            }
        });
        
        console.log(`✅ 攻击页面已初始化，加载了 ${profiles.length} 个目标`);
        
    } catch (error) {
        console.error('初始化攻击页面失败:', error);
        showAlert(`初始化失败：${error.message}`, 'danger');
    }
}

/**
 * 加载分析报告
 */
async function loadAnalysisReport() {
    const select = document.getElementById('reportPersonSelect');
    const selectedId = select.value;
    
    if (!selectedId) {
        document.getElementById('reportContent').style.display = 'none';
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${selectedId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 生成demo分析报告
        const report = generateDemoAnalysisReport(profile);
        
        // 显示报告
        const reportContent = document.getElementById('reportContent');
        reportContent.innerHTML = report;
        reportContent.style.display = 'block';
        
        showAlert(`已加载分析报告：${profile.name || '未知姓名'}`, 'success');
        
    } catch (error) {
        console.error('加载分析报告失败:', error);
        showAlert(`加载失败：${error.message}`, 'danger');
    }
}

/**
 * 生成demo分析报告
 */
function generateDemoAnalysisReport(profile) {
    const report = `
        <div class="card">
            <div class="card-header">
                <h5 class="card-title mb-0">
                    <i class="fas fa-chart-line me-2"></i>
                    ${profile.name || '未知姓名'} - 全面档案分析报告
                </h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6 class="text-primary">基础信息</h6>
                        <ul class="list-unstyled">
                            <li><strong>姓名：</strong>${profile.name || '未知'}</li>
                            <li><strong>年龄：</strong>${profile.age || '未知'}</li>
                            <li><strong>性别：</strong>${profile.gender || '未知'}</li>
                            <li><strong>职业：</strong>${profile.current_job || '未知'}</li>
                            <li><strong>公司：</strong>${profile.current_company || '未知'}</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6 class="text-success">分析状态</h6>
                        <ul class="list-unstyled">
                            <li><span class="badge bg-success">✓ 基础信息完整</span></li>
                            <li><span class="badge bg-success">✓ 社交网络分析</span></li>
                            <li><span class="badge bg-success">✓ 心理特征评估</span></li>
                            <li><span class="badge bg-success">✓ 行为模式识别</span></li>
                            <li><span class="badge bg-success">✓ 弱点分析完成</span></li>
                        </ul>
                    </div>
                </div>
                
                <hr class="my-4">
                
                <h6 class="text-warning">18维度深度分析结果</h6>
                <div class="row">
                    <div class="col-md-4">
                        <h6 class="text-info">心理维度</h6>
                        <ul class="small">
                            <li>性格特征：外向型，善于社交</li>
                            <li>决策风格：理性分析型</li>
                            <li>压力承受：中等水平</li>
                            <li>情绪稳定性：良好</li>
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-info">社交维度</h6>
                        <ul class="small">
                            <li>社交圈：广泛，多行业</li>
                            <li>影响力：中等偏上</li>
                            <li>信任度：较高</li>
                            <li>合作倾向：积极</li>
                        </ul>
                    </div>
                    <div class="col-md-4">
                        <h6 class="text-info">行为维度</h6>
                        <ul class="small">
                            <li>工作习惯：规律，高效</li>
                            <li>生活作息：规律</li>
                            <li>消费习惯：理性</li>
                            <li>风险偏好：保守</li>
                        </ul>
                    </div>
                </div>
                
                <hr class="my-4">
                
                <h6 class="text-danger">关键发现与建议</h6>
                <div class="alert alert-info">
                    <h6><i class="fas fa-lightbulb me-2"></i>分析总结</h6>
                    <p class="mb-0">该目标人物具有完整的社会网络和稳定的心理特征，在社交工程攻击方面需要更精细的策略。建议从工作压力和生活习惯入手，寻找心理弱点。</p>
                </div>
            </div>
        </div>
    `;
    
    return report;
}

/**
 * 刷新报告人物列表
 */
async function refreshReportPersonList() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        const select = document.getElementById('reportPersonSelect');
        if (select) {
            select.innerHTML = '<option value="">请选择人物...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                select.appendChild(option);
            });
            showAlert(`已加载 ${profiles.length} 个人物档案`, 'success');
        }
    } catch (error) {
        console.error('刷新报告人物列表失败:', error);
        showAlert(`刷新失败：${error.message}`, 'danger');
    }
}

/**
 * 检查画像构建人物
 */
async function checkProfilePerson() {
    const select = document.getElementById('profilePersonSelect');
    const selectedId = select.value;
    
    if (!selectedId) {
        document.getElementById('profilePersonInfo').style.display = 'none';
        document.getElementById('generateProfileBtn').disabled = true;
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/persons/${selectedId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const profile = await response.json();
        
        // 显示人物信息
        const infoDiv = document.getElementById('profilePersonInfo');
        const detailsDiv = document.getElementById('profilePersonDetails');
        
        detailsDiv.innerHTML = `
            <p><strong>姓名：</strong>${profile.name || '未知'}</p>
            <p><strong>职业：</strong>${profile.current_job || '未知'}</p>
            <p><strong>公司：</strong>${profile.current_company || '未知'}</p>
            <p><strong>档案状态：</strong><span class="badge bg-success">全面档案分析完成</span></p>
        `;
        
        infoDiv.style.display = 'block';
        document.getElementById('generateProfileBtn').disabled = false;
        
    } catch (error) {
        console.error('检查画像人物失败:', error);
        showAlert(`检查失败：${error.message}`, 'danger');
    }
}

/**
 * 刷新画像人物列表
 */
async function refreshProfilePersonList() {
    try {
        const response = await fetch(`${API_BASE}/persons`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        const profiles = Array.isArray(data.items) ? data.items : [];
        
        const select = document.getElementById('profilePersonSelect');
        if (select) {
            select.innerHTML = '<option value="">请选择人物...</option>';
            profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile.id;
                option.textContent = `${profile.name || '未知姓名'} - ${profile.current_job || '未知职位'} (${profile.current_company || '未知公司'})`;
                select.appendChild(option);
            });
            showAlert(`已加载 ${profiles.length} 个人物档案`, 'success');
        }
    } catch (error) {
        console.error('刷新画像人物列表失败:', error);
        showAlert(`刷新失败：${error.message}`, 'danger');
    }
}

/**
 * 从文本智能导入人物信息
 */
async function importFromText() {
    const textArea = document.getElementById('textImportArea');
    const text = textArea.value.trim();
    
    if (!text) {
        showAlert('请先输入或粘贴人物信息文本', 'warning');
        return;
    }
    
    // 显示加载状态
    const statusDiv = document.getElementById('textImportStatus');
    const messageSpan = document.getElementById('textImportMessage');
    statusDiv.style.display = 'block';
    messageSpan.textContent = '正在使用AI智能解析文本...';
    
    try {
        console.log('🔄 开始文本智能导入...');
        const response = await fetch(`${API_BASE}/persons/import_text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        console.log('✅ 文本导入成功:', result);
        
        // 填充表单
        fillFormWithProfile(result.person);
        
        // 更新状态
        messageSpan.textContent = '✅ 文本解析成功！人物信息已自动填充到表单中';
        statusDiv.querySelector('.alert').className = 'alert alert-success';
        
        // 刷新人物列表
        refreshPersons();
        
        showAlert(`成功导入人物：${result.person.name || '未知姓名'}`, 'success');
        
    } catch (error) {
        console.error('❌ 文本导入失败:', error);
        messageSpan.textContent = `❌ 解析失败：${error.message}`;
        statusDiv.querySelector('.alert').className = 'alert alert-danger';
        showAlert(`文本解析失败：${error.message}`, 'danger');
    }
}

/**
 * 清空文本导入区域
 */
function clearTextImport() {
    const textArea = document.getElementById('textImportArea');
    const statusDiv = document.getElementById('textImportStatus');
    
    textArea.value = '';
    statusDiv.style.display = 'none';
    
    showAlert('文本区域已清空', 'info');
}

/**
 * 用人物档案数据填充表单
 */
function fillFormWithProfile(profile) {
    console.log('🔄 填充表单数据:', profile);
    
    const form = document.getElementById('comprehensiveAnalysisForm');
    if (!form) {
        console.error('找不到表单元素');
        return;
    }
    
    // 遍历表单字段并填充数据
    const fields = [
        'name', 'gender', 'age', 'birth_date', 'constellation', 'zodiac',
        'email', 'phone', 'address', 'current_job', 'current_company',
        'skills', 'education', 'career_path', 'consumption_areas',
        'social_activities', 'hobbies', 'emotional_triggers', 'sensitive_topics',
        'security_sources', 'common_topics', 'timeline', 'behavior_observations',
        'information_sources', 'analysis_notes'
    ];
    
    fields.forEach(field => {
        const element = form.querySelector(`[name="${field}"]`);
        if (element && profile[field]) {
            if (element.tagName === 'SELECT') {
                // 对于下拉框，需要找到匹配的选项
                const options = Array.from(element.options);
                const matchingOption = options.find(option => 
                    option.value === profile[field] || 
                    option.text === profile[field]
                );
                if (matchingOption) {
                    element.value = matchingOption.value;
                }
            } else {
                // 对于输入框和文本域
                element.value = profile[field];
            }
        }
    });
    
    console.log('✅ 表单填充完成');
}