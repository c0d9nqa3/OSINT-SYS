"""
核心OSINT服务
整合所有功能模块，提供统一的调查接口
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from app.core.logger import logger, log_audit, log_collection
from app.core.config import settings
from app.models.person import PersonProfile, SearchResult
from app.collectors.demo_collector import DemoCollector
try:
    from app.collectors.google_collector import GoogleCollector
except ImportError:
    GoogleCollector = None

try:
    from app.ai.identity_verifier import IdentityVerifier, VerificationResult
except ImportError:
    try:
        from app.ai.simple_verifier import SimpleIdentityVerifier as IdentityVerifier, VerificationResult
    except ImportError:
        IdentityVerifier = None
        VerificationResult = None
from app.services.network_analyzer import SocialNetworkAnalyzer, NetworkAnalysisResult
from app.services.resume_parser import ResumeParser, ResumeData

class OSINTService:
    """核心OSINT服务类"""
    
    def __init__(self):
        self.collectors = {}
        self.identity_verifier = None
        self.network_analyzer = None
        self.resume_parser = None
        self.active_investigations = {}
        
    async def initialize(self):
        """初始化服务"""
        logger.info("正在初始化OSINT服务...")
        
        # 初始化数据收集器
        await self._initialize_collectors()
        
        # 初始化AI服务
        if IdentityVerifier:
            self.identity_verifier = IdentityVerifier()
            logger.info("AI身份验证器已启用")
        else:
            logger.warning("AI身份验证器不可用，将跳过身份验证")
            self.identity_verifier = None
        
        # 初始化分析器
        self.network_analyzer = SocialNetworkAnalyzer()
        self.resume_parser = ResumeParser()
        
        logger.info("OSINT服务初始化完成")
    
    async def _initialize_collectors(self):
        """初始化数据收集器"""
        # 演示收集器（总是可用）
        demo_collector = DemoCollector()
        self.collectors['demo'] = demo_collector
        logger.info("演示数据收集器已启用")
        
        # Google搜索收集器（如果可用）
        if GoogleCollector:
            try:
                google_collector = GoogleCollector()
                if google_collector.can_collect_from(""):
                    self.collectors['google'] = google_collector
                    logger.info("Google搜索收集器已启用")
                else:
                    logger.warning("Google搜索收集器未配置，跳过")
            except Exception as e:
                logger.warning(f"Google搜索收集器初始化失败: {e}")
        else:
            logger.info("Google搜索收集器不可用，使用演示模式")
        
        # 这里可以添加更多收集器
        # self.collectors['linkedin'] = LinkedInCollector()
        # self.collectors['twitter'] = TwitterCollector()
        
        logger.info(f"已初始化 {len(self.collectors)} 个数据收集器")
    
    async def start_investigation(self, target_name: str, user_id: str = "system") -> str:
        """开始调查"""
        investigation_id = str(uuid.uuid4())
        
        log_audit("INVESTIGATION_START", target_name, user=user_id, 
                 details=f"调查ID: {investigation_id}")
        
        # 创建目标人物档案
        target_profile = PersonProfile(
            id=investigation_id,
            name=target_name,
            created_at=datetime.now()
        )
        
        # 存储调查信息
        self.active_investigations[investigation_id] = {
            'target_profile': target_profile,
            'status': 'collecting',
            'progress': 0,
            'start_time': datetime.now(),
            'user_id': user_id,
            'search_results': [],
            'verification_result': None,
            'network_analysis': None,
            'resume_data': None
        }
        
        # 启动后台调查任务
        asyncio.create_task(self._run_investigation(investigation_id))
        
        return investigation_id
    
    async def _run_investigation(self, investigation_id: str):
        """执行调查任务"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation:
            return
        
        try:
            target_profile = investigation['target_profile']
            target_name = target_profile.name
            
            # 阶段1: 数据收集
            logger.info(f"开始收集 {target_name} 的信息...")
            investigation['status'] = 'collecting'
            investigation['progress'] = 10
            
            search_results = await self._collect_data(target_name)
            investigation['search_results'] = search_results
            investigation['progress'] = 40
            
            logger.info(f"收集到 {len(search_results)} 条搜索结果")
            
            # 阶段2: 身份验证
            logger.info(f"开始验证 {target_name} 的身份...")
            investigation['status'] = 'verifying'
            investigation['progress'] = 50
            
            if self.identity_verifier:
                verification_result = await self.identity_verifier.verify_identity(
                    target_profile, search_results
                )
                investigation['verification_result'] = verification_result
            else:
                # 创建简单的验证结果
                from app.ai.simple_verifier import VerificationResult
                verification_result = VerificationResult(
                    is_same_person=True,
                    confidence_score=0.7,
                    evidence=["演示模式 - 基础验证"],
                    reasoning="演示模式下使用基础验证方法",
                    verification_methods=["demo_verification"]
                )
                investigation['verification_result'] = verification_result
            
            investigation['progress'] = 60
            
            # 阶段3: 关系网络分析
            logger.info(f"开始分析 {target_name} 的社会关系网络...")
            investigation['status'] = 'analyzing_network'
            investigation['progress'] = 70
            
            network_analysis = await self.network_analyzer.build_network(
                target_profile, search_results
            )
            investigation['network_analysis'] = network_analysis
            investigation['progress'] = 80
            
            # 阶段4: 履历解析
            logger.info(f"开始解析 {target_name} 的履历信息...")
            investigation['status'] = 'parsing_resume'
            investigation['progress'] = 90
            
            resume_data = await self.resume_parser.parse_resume_from_results(
                target_profile, search_results
            )
            investigation['resume_data'] = resume_data
            
            # 完成
            investigation['status'] = 'completed'
            investigation['progress'] = 100
            investigation['end_time'] = datetime.now()
            
            logger.info(f"调查 {target_name} 完成")
            log_audit("INVESTIGATION_COMPLETE", target_name, 
                     details=f"调查ID: {investigation_id}, 结果数: {len(search_results)}")
            
        except Exception as e:
            logger.error(f"调查执行失败: {e}")
            investigation['status'] = 'failed'
            investigation['error'] = str(e)
            investigation['end_time'] = datetime.now()
            
            log_audit("INVESTIGATION_FAILED", target_profile.name, 
                     details=f"调查ID: {investigation_id}, 错误: {str(e)}")
    
    async def _collect_data(self, target_name: str) -> List[SearchResult]:
        """收集数据"""
        all_results = []
        
        # 并行执行所有收集器
        tasks = []
        
        for collector_name, collector in self.collectors.items():
            task = asyncio.create_task(
                self._collect_from_source(collector, target_name, collector_name)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整合结果
        for results in results_list:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                logger.warning(f"数据收集任务失败: {results}")
        
        return all_results
    
    async def _collect_from_source(self, collector, target_name: str, source_name: str) -> List[SearchResult]:
        """从单个数据源收集信息"""
        try:
            logger.info(f"开始从 {source_name} 收集 {target_name} 的信息")
            
            # 基础搜索
            basic_results = await collector.search(target_name, limit=10)
            
            # 如果是Google收集器，进行更深入的搜索
            if hasattr(collector, 'search_social_profiles'):
                social_results = await collector.search_social_profiles(target_name)
                basic_results.extend(social_results)
            
            if hasattr(collector, 'search_professional_info'):
                professional_results = await collector.search_professional_info(target_name)
                basic_results.extend(professional_results)
            
            logger.info(f"从 {source_name} 收集到 {len(basic_results)} 条结果")
            return basic_results
            
        except Exception as e:
            logger.error(f"从 {source_name} 收集数据失败: {e}")
            return []
    
    def get_investigation_status(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """获取调查状态"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation:
            return None
        
        return {
            'id': investigation_id,
            'target_name': investigation['target_profile'].name,
            'status': investigation['status'],
            'progress': investigation['progress'],
            'start_time': investigation['start_time'],
            'end_time': investigation.get('end_time'),
            'error': investigation.get('error'),
            'results_count': len(investigation['search_results'])
        }
    
    def get_investigation_results(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """获取调查结果"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation:
            return None
        
        return {
            'id': investigation_id,
            'target_profile': investigation['target_profile'],
            'search_results': investigation['search_results'],
            'verification_result': investigation.get('verification_result'),
            'network_analysis': investigation.get('network_analysis'),
            'resume_data': investigation.get('resume_data'),
            'status': investigation['status'],
            'progress': investigation['progress'],
            'start_time': investigation['start_time'],
            'end_time': investigation.get('end_time')
        }
    
    def list_investigations(self, user_id: str = None) -> List[Dict[str, Any]]:
        """列出调查"""
        investigations = []
        
        for inv_id, inv_data in self.active_investigations.items():
            if user_id is None or inv_data.get('user_id') == user_id:
                investigations.append({
                    'id': inv_id,
                    'target_name': inv_data['target_profile'].name,
                    'status': inv_data['status'],
                    'progress': inv_data['progress'],
                    'start_time': inv_data['start_time'],
                    'end_time': inv_data.get('end_time')
                })
        
        # 按开始时间排序
        investigations.sort(key=lambda x: x['start_time'], reverse=True)
        return investigations
    
    def delete_investigation(self, investigation_id: str, user_id: str = "system") -> bool:
        """删除调查"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation:
            return False
        
        target_name = investigation['target_profile'].name
        
        # 删除调查数据
        del self.active_investigations[investigation_id]
        
        log_audit("INVESTIGATION_DELETE", target_name, user=user_id,
                 details=f"调查ID: {investigation_id}")
        
        return True
    
    async def get_network_visualization(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """获取网络可视化数据"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation or not investigation.get('network_analysis'):
            return None
        
        # 重新构建网络以获取可视化数据
        target_profile = investigation['target_profile']
        search_results = investigation['search_results']
        
        await self.network_analyzer.build_network(target_profile, search_results)
        return self.network_analyzer.get_network_visualization_data()
    
    async def export_investigation(self, investigation_id: str, format: str = 'json') -> Optional[str]:
        """导出调查结果"""
        investigation = self.active_investigations.get(investigation_id)
        if not investigation:
            return None
        
        export_data = {
            'investigation_id': investigation_id,
            'target_name': investigation['target_profile'].name,
            'export_time': datetime.now().isoformat(),
            'status': investigation['status'],
            'search_results_count': len(investigation['search_results']),
            'verification_result': investigation.get('verification_result'),
            'network_analysis': investigation.get('network_analysis'),
            'resume_data': investigation.get('resume_data')
        }
        
        if format == 'json':
            import json
            return json.dumps(export_data, default=str, indent=2, ensure_ascii=False)
        elif format == 'markdown':
            return self._export_to_markdown(export_data)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    def _export_to_markdown(self, data: Dict[str, Any]) -> str:
        """导出为Markdown格式"""
        lines = []
        
        lines.append(f"# OSINT调查报告: {data['target_name']}")
        lines.append("")
        lines.append(f"**调查ID**: {data['investigation_id']}")
        lines.append(f"**导出时间**: {data['export_time']}")
        lines.append(f"**调查状态**: {data['status']}")
        lines.append(f"**搜索结果数量**: {data['search_results_count']}")
        lines.append("")
        
        # 身份验证结果
        if data.get('verification_result'):
            verification = data['verification_result']
            lines.append("## 身份验证结果")
            lines.append(f"**置信度**: {verification.confidence_score:.2f}")
            lines.append(f"**结论**: {'同一人' if verification.is_same_person else '非同一人'}")
            lines.append(f"**推理**: {verification.reasoning}")
            lines.append("")
        
        # 网络分析结果
        if data.get('network_analysis'):
            network = data['network_analysis']
            lines.append("## 社会关系网络分析")
            lines.append(f"**节点数**: {network.total_nodes}")
            lines.append(f"**边数**: {network.total_edges}")
            lines.append(f"**网络密度**: {network.density:.3f}")
            
            if network.key_persons:
                lines.append("### 关键人物")
                for person in network.key_persons[:5]:
                    lines.append(f"- {person['name']} (重要性: {person['importance_score']:.2f})")
            lines.append("")
        
        # 履历信息
        if data.get('resume_data'):
            resume = data['resume_data']
            lines.append("## 履历信息")
            if resume.summary:
                lines.append(resume.summary)
            lines.append("")
        
        return '\n'.join(lines)
    
    async def cleanup(self):
        """清理资源"""
        logger.info("正在清理OSINT服务资源...")
        
        # 清理收集器
        for collector in self.collectors.values():
            if hasattr(collector, 'cleanup'):
                await collector.cleanup()
        
        # 清理调查数据 (可选择性保留)
        if settings.ENABLE_DATA_RETENTION_POLICY:
            await self._cleanup_old_investigations()
        
        logger.info("OSINT服务资源清理完成")
    
    async def _cleanup_old_investigations(self):
        """清理过期的调查数据"""
        cutoff_time = datetime.now() - timedelta(days=settings.DATA_RETENTION_DAYS)
        
        expired_investigations = []
        for inv_id, inv_data in self.active_investigations.items():
            if inv_data['start_time'] < cutoff_time:
                expired_investigations.append(inv_id)
        
        for inv_id in expired_investigations:
            target_name = self.active_investigations[inv_id]['target_profile'].name
            del self.active_investigations[inv_id]
            logger.info(f"已清理过期调查: {target_name} ({inv_id})")
        
        if expired_investigations:
            log_audit("DATA_CLEANUP", "system", 
                     details=f"清理了 {len(expired_investigations)} 条过期调查记录") 