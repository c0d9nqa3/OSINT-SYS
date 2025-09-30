"""
调查相关API路由
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Response
from fastapi.responses import PlainTextResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio

from app.api.dependencies import get_osint_service
from app.core.logger import logger, log_audit

router = APIRouter()

# 请求模型
class InvestigationRequest(BaseModel):
    """调查请求模型"""
    target_name: str = Field(..., description="目标人物姓名")
    user_id: str = Field(default="anonymous", description="用户ID")

class InvestigationResponse(BaseModel):
    """调查响应模型"""
    investigation_id: str = Field(..., description="调查ID")
    message: str = Field(..., description="响应消息")

class InvestigationStatus(BaseModel):
    """调查状态模型"""
    id: str
    target_name: str
    status: str
    progress: int
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None
    results_count: int

@router.post("/investigations", response_model=InvestigationResponse)
async def start_investigation(
    request: InvestigationRequest,
    osint_service: Any = Depends(get_osint_service)
):
    """
    开始新的调查
    
    - **target_name**: 目标人物的姓名
    - **user_id**: 发起调查的用户ID（可选）
    """
    try:
        # 验证输入
        if not request.target_name or len(request.target_name.strip()) < 2:
            raise HTTPException(
                status_code=400,
                detail="目标姓名长度至少需要2个字符"
            )
        
        # 开始调查
        investigation_id = await osint_service.start_investigation(
            target_name=request.target_name.strip(),
            user_id=request.user_id
        )
        
        logger.info(f"开始调查 {request.target_name}，调查ID: {investigation_id}")
        
        return InvestigationResponse(
            investigation_id=investigation_id,
            message=f"已开始调查 {request.target_name}，调查ID: {investigation_id}"
        )
        
    except Exception as e:
        logger.error(f"开始调查失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"开始调查失败: {str(e)}"
        )

@router.get("/investigations", response_model=List[InvestigationStatus])
async def list_investigations(
    user_id: Optional[str] = Query(None, description="用户ID，为空则显示所有调查"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    列出所有调查
    
    - **user_id**: 可选，指定用户ID来过滤调查结果
    """
    try:
        investigations = osint_service.list_investigations(user_id=user_id)
        
        # 转换为响应模型
        response_data = []
        for inv in investigations:
            response_data.append(InvestigationStatus(
                id=inv['id'],
                target_name=inv['target_name'],
                status=inv['status'],
                progress=inv['progress'],
                start_time=inv['start_time'].isoformat(),
                end_time=inv['end_time'].isoformat() if inv.get('end_time') else None,
                error=inv.get('error'),
                results_count=inv.get('results_count', 0)
            ))
        
        return response_data
        
    except Exception as e:
        logger.error(f"列出调查失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"列出调查失败: {str(e)}"
        )

@router.get("/investigations/{investigation_id}/status", response_model=InvestigationStatus)
async def get_investigation_status(
    investigation_id: str = Path(..., description="调查ID"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    获取特定调查的状态
    """
    try:
        status = osint_service.get_investigation_status(investigation_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在"
            )
        
        return InvestigationStatus(
            id=status['id'],
            target_name=status['target_name'],
            status=status['status'],
            progress=status['progress'],
            start_time=status['start_time'].isoformat(),
            end_time=status['end_time'].isoformat() if status.get('end_time') else None,
            error=status.get('error'),
            results_count=status.get('results_count', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取调查状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取调查状态失败: {str(e)}"
        )

@router.get("/investigations/{investigation_id}/results")
async def get_investigation_results(
    investigation_id: str = Path(..., description="调查ID"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    获取调查结果的详细信息
    """
    try:
        logger.info(f"正在获取调查结果: {investigation_id}")
        results = osint_service.get_investigation_results(investigation_id)
        logger.info(f"获取到调查结果: {type(results)}")
        
        if not results:
            logger.warning(f"调查 {investigation_id} 不存在")
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在"
            )
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在"
            )
        
        # 将结果转换为可序列化的格式
        logger.info("开始序列化调查结果")
        serializable_results = {
            'id': results['id'],
            'status': results['status'],
            'progress': results['progress'],
            'target_profile': {
                'id': results['target_profile'].id,
                'name': results['target_profile'].name,
                'email': results['target_profile'].email or '',
                'phone': results['target_profile'].phone or '',
                'address': results['target_profile'].address or '',
                'current_job': results['target_profile'].current_job or '',
                'current_company': results['target_profile'].current_company or '',
                'created_at': results['target_profile'].created_at.isoformat() if results['target_profile'].created_at else None,
                'verified': results['target_profile'].verified,
                'confidence_score': results['target_profile'].confidence_score
            },
            'search_results': [
                {
                    'source': result.source,
                    'url': result.url,
                    'title': result.title,
                    'snippet': result.snippet,
                    'relevance_score': result.relevance_score,
                    'reliability_score': result.reliability_score,
                    'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                    'extracted_data': result.extracted_data or {},
                    'person_mentions': result.person_mentions or []
                }
                for result in results['search_results']
            ],
            'start_time': results['start_time'].isoformat() if results['start_time'] else None,
            'end_time': results['end_time'].isoformat() if results['end_time'] else None
        }
        
        # 添加验证结果
        if results.get('verification_result'):
            verification = results['verification_result']
            serializable_results['verification_result'] = {
                'is_same_person': bool(verification.is_same_person),
                'confidence_score': float(verification.confidence_score),
                'evidence': verification.evidence,
                'reasoning': verification.reasoning,
                'verification_methods': verification.verification_methods
            }
        
        # 添加网络分析结果
        if results.get('network_analysis'):
            network = results['network_analysis']
            # 转换numpy类型为Python原生类型
            serializable_results['network_analysis'] = {
                'total_nodes': int(network.total_nodes),
                'total_edges': int(network.total_edges),
                'density': float(network.density),
                'key_persons': [
                    {
                        'id': str(person['id']),
                        'name': str(person['name']),
                        'importance_score': float(person['importance_score']),
                        'degree_centrality': float(person['degree_centrality']),
                        'betweenness_centrality': float(person['betweenness_centrality']),
                        'is_center': bool(person['is_center']),
                        'job': person.get('job', ''),
                        'company': person.get('company', '')
                    }
                    for person in network.key_persons
                ] if network.key_persons else [],
                'communities': [
                    [str(node) for node in community]
                    for community in network.communities
                ] if network.communities else [],
                'influence_scores': {
                    str(k): float(v) for k, v in network.influence_scores.items()
                } if network.influence_scores else {},
                'relationship_patterns': {
                    str(k): v if not hasattr(v, 'item') else float(v)
                    for k, v in network.relationship_patterns.items()
                } if network.relationship_patterns else {}
            }
        
        # 添加履历数据
        if results.get('resume_data'):
            resume = results['resume_data']
            serializable_results['resume_data'] = {
                'personal_info': resume.personal_info or {},
                'summary': resume.summary or '',
                'confidence_score': float(resume.confidence_score) if hasattr(resume.confidence_score, 'item') else float(resume.confidence_score),
                'work_experiences': [
                    {
                        'company': work.company or '',
                        'position': work.position or '',
                        'start_date': work.start_date.isoformat() if work.start_date else None,
                        'end_date': work.end_date.isoformat() if work.end_date else None,
                        'is_current': bool(work.is_current) if hasattr(work.is_current, 'item') else bool(work.is_current),
                        'description': work.description or '',
                        'location': work.location or '',
                        'source': work.source or '',
                        'confidence': float(work.confidence) if hasattr(work.confidence, 'item') else float(work.confidence)
                    }
                    for work in resume.work_experiences
                ],
                'education': [
                    {
                        'institution': edu.institution or '',
                        'degree': edu.degree or '',
                        'major': edu.major or '',
                        'start_date': edu.start_date.isoformat() if edu.start_date else None,
                        'end_date': edu.end_date.isoformat() if edu.end_date else None,
                        'description': edu.description or '',
                        'source': edu.source or '',
                        'confidence': float(edu.confidence) if hasattr(edu.confidence, 'item') else float(edu.confidence)
                    }
                    for edu in resume.education
                ],
                'skills': [
                    {
                        'name': skill.name or '',
                        'category': skill.category or '',
                        'proficiency': skill.proficiency or '',
                        'years_experience': int(skill.years_experience) if skill.years_experience else None,
                        'source': skill.source or '',
                        'confidence': float(skill.confidence) if hasattr(skill.confidence, 'item') else float(skill.confidence)
                    }
                    for skill in resume.skills
                ],
                'achievements': [
                    {
                        'title': achievement.title or '',
                        'organization': achievement.organization or '',
                        'date': achievement.date.isoformat() if achievement.date else None,
                        'description': achievement.description or '',
                        'category': achievement.category or '',
                        'source': achievement.source or '',
                        'confidence': float(achievement.confidence) if hasattr(achievement.confidence, 'item') else float(achievement.confidence)
                    }
                    for achievement in resume.achievements
                ],
                'timeline': [
                    {
                        'date': event['date'].isoformat() if event.get('date') else None,
                        'type': event.get('type', ''),
                        'title': event.get('title', ''),
                        'category': event.get('category', '')
                    }
                    for event in resume.timeline
                ]
            }
        
        return serializable_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取调查结果失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"获取调查结果失败: {str(e)}"
        )

@router.delete("/investigations/{investigation_id}")
async def delete_investigation(
    investigation_id: str = Path(..., description="调查ID"),
    user_id: str = Query("anonymous", description="用户ID"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    删除调查
    """
    try:
        success = osint_service.delete_investigation(investigation_id, user_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在"
            )
        
        return {"message": f"调查 {investigation_id} 已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除调查失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"删除调查失败: {str(e)}"
        )

@router.get("/investigations/{investigation_id}/export")
async def export_investigation(
    investigation_id: str = Path(..., description="调查ID"),
    format: str = Query("json", description="导出格式: json, markdown"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    导出调查结果
    """
    try:
        if format not in ["json", "markdown"]:
            raise HTTPException(
                status_code=400,
                detail="支持的格式: json, markdown"
            )
        
        export_data = await osint_service.export_investigation(investigation_id, format)
        
        if not export_data:
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在"
            )
        
        # 根据格式返回不同的响应
        if format == "json":
            return Response(
                content=export_data,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=investigation_{investigation_id}.json"}
            )
        elif format == "markdown":
            return PlainTextResponse(
                content=export_data,
                headers={"Content-Disposition": f"attachment; filename=investigation_{investigation_id}.md"}
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出调查结果失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"导出调查结果失败: {str(e)}"
        )

@router.get("/investigations/{investigation_id}/network")
async def get_network_visualization(
    investigation_id: str = Path(..., description="调查ID"),
    osint_service: Any = Depends(get_osint_service)
):
    """
    获取社会关系网络可视化数据
    """
    try:
        network_data = await osint_service.get_network_visualization(investigation_id)
        
        if not network_data:
            raise HTTPException(
                status_code=404,
                detail=f"调查 {investigation_id} 不存在或网络分析未完成"
            )
        
        return network_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取网络可视化数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"获取网络可视化数据失败: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """
    健康检查接口
    """
    return {
        "status": "healthy",
        "message": "OSINT情报收集系统运行正常",
        "timestamp": logger.info("健康检查访问")
    } 