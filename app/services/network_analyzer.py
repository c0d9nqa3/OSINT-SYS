"""
社会关系网络分析服务
构建和分析人物社会关系网络
"""

import asyncio
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
import numpy as np
from datetime import datetime
import json

from app.models.person import PersonProfile, Relationship, SearchResult
from app.core.logger import logger, log_audit
from app.core.config import settings

@dataclass
class NetworkNode:
    """网络节点"""
    id: str
    name: str
    node_type: str  # person, organization, location
    attributes: Dict[str, Any]
    centrality_scores: Dict[str, float]

@dataclass
class NetworkEdge:
    """网络边"""
    source: str
    target: str
    relationship_type: str
    weight: float
    attributes: Dict[str, Any]

@dataclass
class NetworkAnalysisResult:
    """网络分析结果"""
    total_nodes: int
    total_edges: int
    density: float
    key_persons: List[Dict[str, Any]]
    communities: List[List[str]]
    influence_scores: Dict[str, float]
    relationship_patterns: Dict[str, Any]

class SocialNetworkAnalyzer:
    """社会关系网络分析器"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.relationship_weights = {
            'family': 1.0,
            'colleague': 0.8,
            'friend': 0.7,
            'business_partner': 0.9,
            'classmate': 0.6,
            'mentor': 0.8,
            'subordinate': 0.7,
            'acquaintance': 0.4,
            'unknown': 0.3
        }
        
    async def build_network(self, 
                          target_person: PersonProfile, 
                          search_results: List[SearchResult]) -> NetworkAnalysisResult:
        """构建社会关系网络"""
        
        log_audit("NETWORK_ANALYSIS", target_person.name, details="开始构建社会关系网络")
        
        # 清空现有图
        self.graph.clear()
        self.directed_graph.clear()
        
        # 1. 添加目标人物作为中心节点
        self._add_person_node(target_person, is_center=True)
        
        # 2. 从搜索结果中提取关系
        relationships = await self._extract_relationships_from_results(target_person, search_results)
        
        # 3. 构建网络图
        for relationship in relationships:
            self._add_relationship_to_graph(relationship)
        
        # 4. 分析网络结构
        analysis_result = await self._analyze_network_structure()
        
        log_audit("NETWORK_ANALYSIS", target_person.name, 
                 details=f"网络构建完成，节点数: {analysis_result.total_nodes}, 边数: {analysis_result.total_edges}")
        
        return analysis_result
    
    def _add_person_node(self, person: PersonProfile, is_center: bool = False):
        """添加人物节点"""
        node_id = person.id
        
        attributes = {
            'name': person.name,
            'type': 'person',
            'is_center': is_center,
            'job': person.current_job,
            'company': person.current_company,
            'location': person.address,
            'verified': person.verified,
            'confidence': person.confidence_score
        }
        
        self.graph.add_node(node_id, **attributes)
        self.directed_graph.add_node(node_id, **attributes)
    
    async def _extract_relationships_from_results(self, 
                                                target_person: PersonProfile, 
                                                search_results: List[SearchResult]) -> List[Relationship]:
        """从搜索结果中提取关系"""
        relationships = []
        
        for result in search_results:
            try:
                # 提取文本中的人名
                mentioned_persons = self._extract_person_mentions(result.content)
                
                # 分析关系类型
                for person_name in mentioned_persons:
                    if person_name.lower() != target_person.name.lower():
                        relationship_type = self._infer_relationship_type(
                            target_person.name, 
                            person_name, 
                            result.content
                        )
                        
                        if relationship_type:
                            relationship = Relationship(
                                person_id_1=target_person.id,
                                person_id_2=self._generate_person_id(person_name),
                                relationship_type=relationship_type,
                                relationship_strength=self._calculate_relationship_strength(
                                    relationship_type, result.content
                                ),
                                description=f"通过 {result.source} 发现的关系",
                                context=result.snippet,
                                source=result.source,
                                evidence=[result.url]
                            )
                            
                            relationships.append(relationship)
                            
            except Exception as e:
                logger.warning(f"从搜索结果提取关系失败: {e}")
                continue
        
        return relationships
    
    def _extract_person_mentions(self, text: str) -> List[str]:
        """从文本中提取人名提及"""
        import re
        
        person_mentions = []
        
        # 中文姓名模式 (2-4个汉字)
        chinese_names = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        
        # 英文姓名模式
        english_names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
        
        # 过滤常见非人名词汇
        common_words = {'公司', '有限', '科技', '网络', '系统', '管理', '发展'}
        
        for name in chinese_names:
            if name not in common_words and len(name) >= 2:
                person_mentions.append(name)
        
        person_mentions.extend(english_names)
        
        # 去重并限制数量
        return list(set(person_mentions))[:20]
    
    def _infer_relationship_type(self, person1: str, person2: str, context: str) -> Optional[str]:
        """推断关系类型"""
        context_lower = context.lower()
        
        # 定义关系关键词
        relationship_keywords = {
            'colleague': ['同事', '同僚', '工作', '公司', '团队', 'colleague', 'coworker', 'team'],
            'business_partner': ['合作', '伙伴', '商业', '业务', 'partner', 'business', 'cooperation'],
            'friend': ['朋友', '好友', '挚友', 'friend', 'buddy', 'pal'],
            'family': ['父亲', '母亲', '兄弟', '姐妹', '儿子', '女儿', '夫妻', '妻子', '丈夫', 
                      'father', 'mother', 'brother', 'sister', 'son', 'daughter', 'wife', 'husband'],
            'classmate': ['同学', '校友', '同班', 'classmate', 'schoolmate', 'alumni'],
            'mentor': ['导师', '老师', '师傅', '指导', 'mentor', 'teacher', 'advisor', 'supervisor'],
            'subordinate': ['下属', '员工', '助手', 'subordinate', 'employee', 'assistant']
        }
        
        # 根据关键词匹配关系类型
        for rel_type, keywords in relationship_keywords.items():
            for keyword in keywords:
                if keyword in context_lower:
                    return rel_type
        
        # 如果没有明确关系，返回 acquaintance
        return 'acquaintance'
    
    def _calculate_relationship_strength(self, relationship_type: str, context: str) -> float:
        """计算关系强度"""
        base_strength = self.relationship_weights.get(relationship_type, 0.5)
        
        # 根据上下文调整强度
        strength_modifiers = {
            '密切': 0.2,
            '紧密': 0.2,
            '亲密': 0.3,
            '长期': 0.1,
            '多年': 0.1,
            'close': 0.2,
            'tight': 0.2,
            'intimate': 0.3,
            'long-term': 0.1
        }
        
        context_lower = context.lower()
        modifier = 0.0
        
        for keyword, value in strength_modifiers.items():
            if keyword in context_lower:
                modifier += value
        
        final_strength = min(1.0, base_strength + modifier)
        return final_strength
    
    def _generate_person_id(self, person_name: str) -> str:
        """为人名生成ID"""
        import hashlib
        return hashlib.md5(person_name.encode()).hexdigest()[:16]
    
    def _add_relationship_to_graph(self, relationship: Relationship):
        """将关系添加到图中"""
        # 如果目标节点不存在，创建它
        if not self.graph.has_node(relationship.person_id_2):
            # 根据ID推断人名 (这里简化处理)
            person_name = f"Person_{relationship.person_id_2[:8]}"
            
            attributes = {
                'name': person_name,
                'type': 'person',
                'is_center': False,
                'confidence': 0.5  # 未验证的节点置信度较低
            }
            
            self.graph.add_node(relationship.person_id_2, **attributes)
            self.directed_graph.add_node(relationship.person_id_2, **attributes)
        
        # 添加边
        edge_attributes = {
            'relationship_type': relationship.relationship_type,
            'weight': relationship.relationship_strength,
            'description': relationship.description,
            'source': relationship.source,
            'confidence': relationship.confidence_score
        }
        
        self.graph.add_edge(
            relationship.person_id_1, 
            relationship.person_id_2, 
            **edge_attributes
        )
        
        self.directed_graph.add_edge(
            relationship.person_id_1, 
            relationship.person_id_2, 
            **edge_attributes
        )
    
    async def _analyze_network_structure(self) -> NetworkAnalysisResult:
        """分析网络结构"""
        if self.graph.number_of_nodes() == 0:
            return NetworkAnalysisResult(
                total_nodes=0,
                total_edges=0,
                density=0.0,
                key_persons=[],
                communities=[],
                influence_scores={},
                relationship_patterns={}
            )
        
        # 基本统计
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # 计算中心性指标
        centrality_scores = self._calculate_centrality_measures()
        
        # 识别关键人物
        key_persons = self._identify_key_persons(centrality_scores)
        
        # 社区发现
        communities = self._detect_communities()
        
        # 影响力评分
        influence_scores = self._calculate_influence_scores(centrality_scores)
        
        # 关系模式分析
        relationship_patterns = self._analyze_relationship_patterns()
        
        return NetworkAnalysisResult(
            total_nodes=total_nodes,
            total_edges=total_edges,
            density=density,
            key_persons=key_persons,
            communities=communities,
            influence_scores=influence_scores,
            relationship_patterns=relationship_patterns
        )
    
    def _calculate_centrality_measures(self) -> Dict[str, Dict[str, float]]:
        """计算各种中心性指标"""
        centrality_scores = {}
        
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.graph)
            
            # 接近中心性
            closeness_centrality = nx.closeness_centrality(self.graph)
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # 特征向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000)
            except:
                eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
            
            # 组合所有中心性指标
            for node in self.graph.nodes():
                centrality_scores[node] = {
                    'degree': degree_centrality.get(node, 0.0),
                    'closeness': closeness_centrality.get(node, 0.0),
                    'betweenness': betweenness_centrality.get(node, 0.0),
                    'eigenvector': eigenvector_centrality.get(node, 0.0)
                }
                
        except Exception as e:
            logger.error(f"计算中心性指标失败: {e}")
            # 返回默认值
            centrality_scores = {node: {
                'degree': 0.0, 'closeness': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0
            } for node in self.graph.nodes()}
        
        return centrality_scores
    
    def _identify_key_persons(self, centrality_scores: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """识别网络中的关键人物"""
        key_persons = []
        
        for node_id, scores in centrality_scores.items():
            node_data = self.graph.nodes[node_id]
            
            # 计算综合重要性评分
            importance_score = (
                scores['degree'] * 0.3 +
                scores['closeness'] * 0.2 +
                scores['betweenness'] * 0.3 +
                scores['eigenvector'] * 0.2
            )
            
            key_persons.append({
                'id': node_id,
                'name': node_data.get('name', 'Unknown'),
                'importance_score': importance_score,
                'degree_centrality': scores['degree'],
                'betweenness_centrality': scores['betweenness'],
                'is_center': node_data.get('is_center', False),
                'job': node_data.get('job'),
                'company': node_data.get('company')
            })
        
        # 按重要性排序
        key_persons.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return key_persons[:10]  # 返回前10个关键人物
    
    def _detect_communities(self) -> List[List[str]]:
        """检测社区结构"""
        communities = []
        
        try:
            if self.graph.number_of_nodes() > 1:
                # 使用Louvain算法进行社区发现
                import networkx.algorithms.community as nx_comm
                
                communities_generator = nx_comm.greedy_modularity_communities(self.graph)
                communities = [list(community) for community in communities_generator]
                
        except Exception as e:
            logger.warning(f"社区检测失败: {e}")
            # 如果失败，将每个节点作为单独的社区
            communities = [[node] for node in self.graph.nodes()]
        
        return communities
    
    def _calculate_influence_scores(self, centrality_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算影响力评分"""
        influence_scores = {}
        
        for node_id, scores in centrality_scores.items():
            node_data = self.graph.nodes[node_id]
            
            # 基础影响力 = 中心性指标加权平均
            base_influence = (
                scores['degree'] * 0.4 +
                scores['betweenness'] * 0.4 +
                scores['eigenvector'] * 0.2
            )
            
            # 根据节点属性调整
            confidence_multiplier = node_data.get('confidence', 0.5)
            is_center_bonus = 0.2 if node_data.get('is_center', False) else 0.0
            
            final_influence = (base_influence * confidence_multiplier) + is_center_bonus
            influence_scores[node_id] = min(1.0, final_influence)
        
        return influence_scores
    
    def _analyze_relationship_patterns(self) -> Dict[str, Any]:
        """分析关系模式"""
        patterns = {
            'relationship_types': {},
            'strength_distribution': {},
            'network_metrics': {}
        }
        
        # 关系类型分布
        relationship_types = {}
        strength_values = []
        
        for edge in self.graph.edges(data=True):
            rel_type = edge[2].get('relationship_type', 'unknown')
            weight = edge[2].get('weight', 0.5)
            
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            strength_values.append(weight)
        
        patterns['relationship_types'] = relationship_types
        
        # 关系强度分布
        if strength_values:
            patterns['strength_distribution'] = {
                'mean': np.mean(strength_values),
                'std': np.std(strength_values),
                'min': np.min(strength_values),
                'max': np.max(strength_values)
            }
        
        # 网络指标
        try:
            patterns['network_metrics'] = {
                'clustering_coefficient': nx.average_clustering(self.graph),
                'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else 0,
                'average_path_length': nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else 0,
                'number_of_components': nx.number_connected_components(self.graph)
            }
        except Exception as e:
            logger.warning(f"计算网络指标失败: {e}")
            patterns['network_metrics'] = {}
        
        return patterns
    
    def export_network_data(self, format: str = 'json') -> str:
        """导出网络数据"""
        if format == 'json':
            return self._export_to_json()
        elif format == 'gexf':
            return self._export_to_gexf()
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _export_to_json(self) -> str:
        """导出为JSON格式"""
        data = {
            'nodes': [],
            'edges': []
        }
        
        # 导出节点
        for node_id, node_data in self.graph.nodes(data=True):
            data['nodes'].append({
                'id': node_id,
                **node_data
            })
        
        # 导出边
        for source, target, edge_data in self.graph.edges(data=True):
            data['edges'].append({
                'source': source,
                'target': target,
                **edge_data
            })
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _export_to_gexf(self) -> str:
        """导出为GEXF格式"""
        try:
            from io import StringIO
            
            output = StringIO()
            nx.write_gexf(self.graph, output)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"导出GEXF格式失败: {e}")
            return ""
    
    def get_network_visualization_data(self) -> Dict[str, Any]:
        """获取网络可视化数据"""
        # 计算布局
        try:
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        except:
            pos = {node: (0, 0) for node in self.graph.nodes()}
        
        # 准备可视化数据
        viz_data = {
            'nodes': [],
            'edges': [],
            'layout': pos
        }
        
        # 节点数据
        for node_id, node_data in self.graph.nodes(data=True):
            x, y = pos.get(node_id, (0, 0))
            
            viz_data['nodes'].append({
                'id': node_id,
                'label': node_data.get('name', node_id),
                'x': float(x),
                'y': float(y),
                'size': max(10, node_data.get('confidence', 0.5) * 30),
                'color': self._get_node_color(node_data),
                'attributes': node_data
            })
        
        # 边数据
        for source, target, edge_data in self.graph.edges(data=True):
            viz_data['edges'].append({
                'source': source,
                'target': target,
                'weight': edge_data.get('weight', 0.5),
                'label': edge_data.get('relationship_type', ''),
                'color': self._get_edge_color(edge_data),
                'attributes': edge_data
            })
        
        return viz_data
    
    def _get_node_color(self, node_data: Dict[str, Any]) -> str:
        """获取节点颜色"""
        if node_data.get('is_center', False):
            return '#ff4444'  # 红色 - 中心节点
        elif node_data.get('verified', False):
            return '#44ff44'  # 绿色 - 已验证
        else:
            return '#4444ff'  # 蓝色 - 普通节点
    
    def _get_edge_color(self, edge_data: Dict[str, Any]) -> str:
        """获取边颜色"""
        rel_type = edge_data.get('relationship_type', 'unknown')
        
        color_map = {
            'family': '#ff0000',      # 红色
            'colleague': '#0000ff',    # 蓝色
            'friend': '#00ff00',       # 绿色
            'business_partner': '#ff8800',  # 橙色
            'classmate': '#8800ff',    # 紫色
            'mentor': '#ffff00',       # 黄色
            'unknown': '#888888'       # 灰色
        }
        
        return color_map.get(rel_type, '#888888') 