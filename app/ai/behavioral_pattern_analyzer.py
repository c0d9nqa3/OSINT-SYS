import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, GPT2Model, GPT2Tokenizer
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from datetime import datetime, timedelta
import random
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class BehavioralPatternEncoder(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=512, sequence_dim=128, num_patterns=20):
        super().__init__()
        
        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.attention_mechanism = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.pattern_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, sequence_dim)
        )
        
        self.behavior_classifier = nn.Sequential(
            nn.Linear(sequence_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_patterns)
        )
        
        self.anomaly_detector = nn.Sequential(
            nn.Linear(sequence_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.prediction_head = nn.Sequential(
            nn.Linear(sequence_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
        self.clustering_layer = nn.Sequential(
            nn.Linear(sequence_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
    
    def forward(self, behavioral_sequences):
        lstm_output, (hidden, cell) = self.temporal_encoder(behavioral_sequences)
        
        attended_output, attention_weights = self.attention_mechanism(
            lstm_output, lstm_output, lstm_output
        )
        
        pattern_features = self.pattern_extractor(attended_output.mean(dim=1))
        
        behavior_patterns = F.softmax(self.behavior_classifier(pattern_features), dim=-1)
        anomaly_scores = self.anomaly_detector(pattern_features)
        future_predictions = self.prediction_head(pattern_features)
        cluster_embeddings = self.clustering_layer(pattern_features)
        
        return {
            'behavior_patterns': behavior_patterns,
            'anomaly_scores': anomaly_scores,
            'future_predictions': future_predictions,
            'cluster_embeddings': cluster_embeddings,
            'pattern_features': pattern_features,
            'attention_weights': attention_weights
        }

class BehavioralPatternAnalyzer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BehavioralPatternEncoder()
        
        self.behavior_patterns = [
            'regular_work_schedule', 'social_media_addiction', 'night_owl_behavior',
            'early_bird_behavior', 'procrastination_pattern', 'multitasking_behavior',
            'focused_work_sessions', 'distracted_behavior', 'routine_follower',
            'spontaneous_behavior', 'risk_taking_pattern', 'conservative_behavior',
            'leadership_behavior', 'follower_behavior', 'collaborative_pattern',
            'independent_behavior', 'stress_response', 'relaxation_seeking',
            'information_seeking', 'entertainment_seeking'
        ]
        
        self.anomaly_types = [
            'unusual_timing', 'abnormal_frequency', 'pattern_break',
            'extreme_behavior', 'inconsistent_activity', 'security_risk'
        ]
        
        if model_path:
            self.load_model(model_path)
    
    def extract_behavioral_sequences(self, behavioral_data):
        sequences = []
        
        for data in behavioral_data:
            sequence_length = min(len(data.get('timeline', [])), 100)
            sequence = np.zeros((sequence_length, 200))
            
            timeline = data.get('timeline', [])
            for i, event in enumerate(timeline[:sequence_length]):
                sequence[i] = self.encode_behavioral_event(event)
            
            sequences.append(torch.tensor(sequence, dtype=torch.float32))
        
        return sequences
    
    def encode_behavioral_event(self, event):
        feature_vector = np.zeros(200)
        
        feature_vector[0] = event.get('hour', 0) / 24
        feature_vector[1] = event.get('day_of_week', 0) / 7
        feature_vector[2] = event.get('activity_type', 0) / 10
        feature_vector[3] = event.get('duration', 0) / 3600
        feature_vector[4] = event.get('intensity', 0)
        feature_vector[5] = event.get('location_type', 0) / 5
        feature_vector[6] = event.get('device_type', 0) / 3
        feature_vector[7] = event.get('interaction_type', 0) / 5
        feature_vector[8] = event.get('emotional_state', 0) / 5
        feature_vector[9] = event.get('stress_level', 0) / 5
        
        if 'typing_patterns' in event:
            typing = event['typing_patterns']
            feature_vector[10] = typing.get('speed', 0) / 100
            feature_vector[11] = typing.get('accuracy', 0)
            feature_vector[12] = typing.get('rhythm_variation', 0)
            feature_vector[13] = typing.get('pause_frequency', 0)
            feature_vector[14] = typing.get('backspace_ratio', 0)
        
        if 'mouse_patterns' in event:
            mouse = event['mouse_patterns']
            feature_vector[15] = mouse.get('movement_speed', 0) / 10
            feature_vector[16] = mouse.get('click_frequency', 0) / 10
            feature_vector[17] = mouse.get('scroll_intensity', 0) / 10
            feature_vector[18] = mouse.get('hover_duration', 0) / 10
            feature_vector[19] = mouse.get('path_efficiency', 0)
        
        if 'navigation_patterns' in event:
            nav = event['navigation_patterns']
            feature_vector[20] = nav.get('page_visit_speed', 0) / 10
            feature_vector[21] = nav.get('back_button_usage', 0)
            feature_vector[22] = nav.get('bookmark_usage', 0)
            feature_vector[23] = nav.get('search_frequency', 0) / 10
            feature_vector[24] = nav.get('tab_switching', 0) / 10
        
        if 'communication_patterns' in event:
            comm = event['communication_patterns']
            feature_vector[25] = comm.get('message_length', 0) / 500
            feature_vector[26] = comm.get('response_time', 0) / 3600
            feature_vector[27] = comm.get('emoji_usage', 0)
            feature_vector[28] = comm.get('formality_level', 0)
            feature_vector[29] = comm.get('question_ratio', 0)
        
        return feature_vector
    
    def analyze_behavioral_patterns(self, behavioral_data):
        self.model.eval()
        
        sequences = self.extract_behavioral_sequences(behavioral_data)
        
        if not sequences:
            return {'error': 'No behavioral data provided'}
        
        with torch.no_grad():
            batch_sequences = torch.stack(sequences).to(self.device)
            outputs = self.model(batch_sequences)
            
            behavior_patterns = outputs['behavior_patterns'].cpu().numpy()
            anomaly_scores = outputs['anomaly_scores'].cpu().numpy()
            future_predictions = outputs['future_predictions'].cpu().numpy()
            cluster_embeddings = outputs['cluster_embeddings'].cpu().numpy()
            
            analysis_results = {
                'behavior_patterns': self.interpret_behavior_patterns(behavior_patterns),
                'anomaly_analysis': self.analyze_anomalies(anomaly_scores),
                'future_behavior_prediction': self.predict_future_behavior(future_predictions),
                'behavioral_clustering': self.perform_behavioral_clustering(cluster_embeddings),
                'pattern_evolution': self.analyze_pattern_evolution(behavioral_data),
                'risk_assessment': self.assess_behavioral_risks(behavioral_data, anomaly_scores)
            }
            
            return analysis_results
    
    def interpret_behavior_patterns(self, behavior_patterns):
        interpretations = []
        
        for i, pattern_scores in enumerate(behavior_patterns):
            person_patterns = []
            
            for j, score in enumerate(pattern_scores):
                if score > 0.7:
                    person_patterns.append({
                        'pattern': self.behavior_patterns[j],
                        'confidence': float(score),
                        'strength': 'strong'
                    })
                elif score > 0.4:
                    person_patterns.append({
                        'pattern': self.behavior_patterns[j],
                        'confidence': float(score),
                        'strength': 'moderate'
                    })
            
            interpretations.append({
                'person_id': i,
                'identified_patterns': person_patterns,
                'dominant_patterns': sorted(person_patterns, key=lambda x: x['confidence'], reverse=True)[:3]
            })
        
        return interpretations
    
    def analyze_anomalies(self, anomaly_scores):
        anomalies = []
        
        for i, score in enumerate(anomaly_scores):
            if score > 0.8:
                anomalies.append({
                    'person_id': i,
                    'anomaly_score': float(score[0]),
                    'severity': 'high',
                    'description': 'Significant behavioral anomaly detected'
                })
            elif score > 0.5:
                anomalies.append({
                    'person_id': i,
                    'anomaly_score': float(score[0]),
                    'severity': 'medium',
                    'description': 'Moderate behavioral anomaly detected'
                })
        
        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies),
            'anomaly_distribution': self.calculate_anomaly_distribution(anomaly_scores)
        }
    
    def calculate_anomaly_distribution(self, anomaly_scores):
        scores = [float(score[0]) for score in anomaly_scores]
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentile_95': np.percentile(scores, 95)
        }
    
    def predict_future_behavior(self, future_predictions):
        predictions = []
        
        for i, prediction in enumerate(future_predictions):
            predicted_behaviors = {
                'person_id': i,
                'predicted_activity_level': float(prediction[0]),
                'predicted_response_time': float(prediction[1]),
                'predicted_engagement': float(prediction[2]),
                'confidence_interval': self.calculate_confidence_interval(prediction)
            }
            predictions.append(predicted_behaviors)
        
        return predictions
    
    def calculate_confidence_interval(self, prediction, confidence=0.95):
        mean = np.mean(prediction)
        std = np.std(prediction)
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_score * std
        
        return {
            'lower_bound': mean - margin_error,
            'upper_bound': mean + margin_error,
            'mean': mean
        }
    
    def perform_behavioral_clustering(self, cluster_embeddings):
        if len(cluster_embeddings) < 2:
            return {'clusters': [], 'silhouette_score': 0.0}
        
        optimal_clusters = min(5, len(cluster_embeddings) // 2)
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_embeddings)
        
        silhouette_avg = silhouette_score(cluster_embeddings, cluster_labels)
        
        cluster_analysis = []
        for cluster_id in range(optimal_clusters):
            cluster_members = np.where(cluster_labels == cluster_id)[0]
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'size': len(cluster_members),
                'members': cluster_members.tolist(),
                'centroid': kmeans.cluster_centers_[cluster_id].tolist()
            })
        
        return {
            'clusters': cluster_analysis,
            'silhouette_score': silhouette_avg,
            'optimal_clusters': optimal_clusters
        }
    
    def analyze_pattern_evolution(self, behavioral_data):
        evolution_analysis = []
        
        for i, data in enumerate(behavioral_data):
            timeline = data.get('timeline', [])
            if len(timeline) < 10:
                continue
            
            patterns_over_time = self.calculate_patterns_over_time(timeline)
            
            evolution_analysis.append({
                'person_id': i,
                'pattern_stability': self.calculate_pattern_stability(patterns_over_time),
                'trend_analysis': self.analyze_trends(patterns_over_time),
                'change_points': self.detect_change_points(patterns_over_time)
            })
        
        return evolution_analysis
    
    def calculate_patterns_over_time(self, timeline):
        window_size = 10
        patterns = []
        
        for i in range(0, len(timeline) - window_size, window_size):
            window = timeline[i:i + window_size]
            pattern_features = np.mean([self.encode_behavioral_event(event) for event in window], axis=0)
            patterns.append(pattern_features)
        
        return patterns
    
    def calculate_pattern_stability(self, patterns_over_time):
        if len(patterns_over_time) < 2:
            return 1.0
        
        distances = []
        for i in range(len(patterns_over_time) - 1):
            distance = np.linalg.norm(patterns_over_time[i] - patterns_over_time[i + 1])
            distances.append(distance)
        
        stability = 1.0 - (np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0)
        return max(0.0, min(1.0, stability))
    
    def analyze_trends(self, patterns_over_time):
        if len(patterns_over_time) < 3:
            return {'trend': 'insufficient_data'}
        
        trends = []
        for feature_idx in range(min(50, len(patterns_over_time[0]))):
            feature_values = [pattern[feature_idx] for pattern in patterns_over_time]
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(feature_values)), feature_values)
            
            if abs(r_value) > 0.7:
                trends.append({
                    'feature': feature_idx,
                    'trend': 'increasing' if slope > 0 else 'decreasing',
                    'strength': abs(r_value),
                    'significance': p_value
                })
        
        return {
            'significant_trends': trends,
            'overall_trend': 'stable' if len(trends) < 3 else 'changing'
        }
    
    def detect_change_points(self, patterns_over_time):
        if len(patterns_over_time) < 5:
            return []
        
        change_points = []
        threshold = np.std([np.linalg.norm(patterns_over_time[i] - patterns_over_time[i-1]) 
                           for i in range(1, len(patterns_over_time))]) * 2
        
        for i in range(1, len(patterns_over_time)):
            distance = np.linalg.norm(patterns_over_time[i] - patterns_over_time[i-1])
            if distance > threshold:
                change_points.append({
                    'time_point': i,
                    'change_magnitude': distance,
                    'description': 'Significant behavioral pattern change detected'
                })
        
        return change_points
    
    def assess_behavioral_risks(self, behavioral_data, anomaly_scores):
        risks = []
        
        for i, data in enumerate(behavioral_data):
            risk_factors = []
            risk_score = 0.0
            
            if i < len(anomaly_scores) and anomaly_scores[i][0] > 0.7:
                risk_factors.append('High anomaly score')
                risk_score += 0.3
            
            timeline = data.get('timeline', [])
            
            if self.detect_suspicious_patterns(timeline):
                risk_factors.append('Suspicious behavioral patterns')
                risk_score += 0.2
            
            if self.detect_unusual_timing(timeline):
                risk_factors.append('Unusual activity timing')
                risk_score += 0.15
            
            if self.detect_rapid_changes(timeline):
                risk_factors.append('Rapid behavioral changes')
                risk_score += 0.15
            
            risks.append({
                'person_id': i,
                'risk_score': min(risk_score, 1.0),
                'risk_factors': risk_factors,
                'risk_level': 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'
            })
        
        return risks
    
    def detect_suspicious_patterns(self, timeline):
        suspicious_indicators = 0
        
        for event in timeline:
            if event.get('hour', 0) < 3 or event.get('hour', 0) > 23:
                suspicious_indicators += 1
            
            if event.get('intensity', 0) > 0.9:
                suspicious_indicators += 1
            
            if event.get('stress_level', 0) > 0.8:
                suspicious_indicators += 1
        
        return suspicious_indicators > len(timeline) * 0.3
    
    def detect_unusual_timing(self, timeline):
        if len(timeline) < 5:
            return False
        
        hours = [event.get('hour', 12) for event in timeline]
        hour_variance = np.var(hours)
        
        return hour_variance > 50
    
    def detect_rapid_changes(self, timeline):
        if len(timeline) < 3:
            return False
        
        changes = 0
        for i in range(1, len(timeline)):
            prev_activity = timeline[i-1].get('activity_type', 0)
            curr_activity = timeline[i].get('activity_type', 0)
            
            if abs(prev_activity - curr_activity) > 5:
                changes += 1
        
        return changes > len(timeline) * 0.4
    
    def train_model(self, training_data, validation_data, epochs=100, batch_size=16, learning_rate=0.001):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        criterion_pattern = nn.CrossEntropyLoss()
        criterion_anomaly = nn.BCELoss()
        criterion_prediction = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in training_data:
                optimizer.zero_grad()
                
                behavioral_sequences = batch['sequences'].to(self.device)
                pattern_labels = batch['pattern_labels'].to(self.device)
                anomaly_labels = batch['anomaly_labels'].to(self.device)
                future_labels = batch['future_labels'].to(self.device)
                
                outputs = self.model(behavioral_sequences)
                
                loss_pattern = criterion_pattern(outputs['behavior_patterns'], pattern_labels)
                loss_anomaly = criterion_anomaly(outputs['anomaly_scores'].squeeze(), anomaly_labels.float())
                loss_prediction = criterion_prediction(outputs['future_predictions'], future_labels)
                
                total_loss = loss_pattern + loss_anomaly + loss_prediction
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(training_data)
            scheduler.step(avg_train_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}')
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'behavior_patterns': self.behavior_patterns,
            'anomaly_types': self.anomaly_types
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.behavior_patterns = checkpoint['behavior_patterns']
        self.anomaly_types = checkpoint['anomaly_types']
        self.model.to(self.device)

class BehavioralDataGenerator:
    def __init__(self):
        self.activity_types = [
            'work', 'social_media', 'email', 'browsing', 'gaming',
            'streaming', 'shopping', 'learning', 'exercise', 'socializing'
        ]
        
        self.emotional_states = [
            'calm', 'stressed', 'excited', 'focused', 'tired',
            'anxious', 'happy', 'frustrated', 'motivated', 'bored'
        ]
        
        self.device_types = ['desktop', 'mobile', 'tablet']
        self.location_types = ['home', 'office', 'public', 'commute', 'other']
    
    def generate_behavioral_timeline(self, person_profile, days=30):
        timeline = []
        
        for day in range(days):
            daily_activities = self.generate_daily_activities(person_profile, day)
            timeline.extend(daily_activities)
        
        return timeline
    
    def generate_daily_activities(self, person_profile, day):
        activities = []
        
        person_type = person_profile.get('personality_type', 'balanced')
        
        if person_type == 'early_bird':
            start_hour = 6
            peak_hours = [8, 9, 10, 14, 15]
        elif person_type == 'night_owl':
            start_hour = 10
            peak_hours = [20, 21, 22, 23, 0]
        else:
            start_hour = 8
            peak_hours = [9, 10, 11, 14, 15, 16]
        
        current_hour = start_hour
        
        while current_hour < 24:
            activity_probability = self.calculate_activity_probability(current_hour, peak_hours, person_type)
            
            if random.random() < activity_probability:
                activity = self.generate_single_activity(person_profile, current_hour, day)
                activities.append(activity)
                
                duration = activity.get('duration', 1)
                current_hour += duration
            else:
                current_hour += 1
        
        return activities
    
    def calculate_activity_probability(self, hour, peak_hours, person_type):
        base_probability = 0.3
        
        if hour in peak_hours:
            base_probability = 0.8
        elif 22 <= hour <= 5:
            base_probability = 0.1
        elif 6 <= hour <= 8:
            base_probability = 0.4
        
        if person_type == 'workaholic':
            if 9 <= hour <= 17:
                base_probability = 0.9
        
        return base_probability
    
    def generate_single_activity(self, person_profile, hour, day):
        activity_type = random.choice(self.activity_types)
        
        personality_traits = person_profile.get('personality_traits', {})
        
        if personality_traits.get('social', 0) > 0.7:
            activity_type = random.choice(['social_media', 'socializing', 'email'])
        elif personality_traits.get('work_focused', 0) > 0.7:
            activity_type = random.choice(['work', 'email', 'learning'])
        elif personality_traits.get('entertainment_seeking', 0) > 0.7:
            activity_type = random.choice(['gaming', 'streaming', 'browsing'])
        
        emotional_state = random.choice(self.emotional_states)
        stress_level = random.uniform(0.1, 0.9)
        
        if emotional_state in ['stressed', 'anxious', 'frustrated']:
            stress_level = random.uniform(0.6, 1.0)
        elif emotional_state in ['calm', 'happy', 'focused']:
            stress_level = random.uniform(0.1, 0.4)
        
        activity = {
            'hour': hour,
            'day_of_week': day % 7,
            'activity_type': self.activity_types.index(activity_type),
            'duration': random.uniform(0.5, 4.0),
            'intensity': random.uniform(0.3, 1.0),
            'location_type': random.randint(0, 4),
            'device_type': random.randint(0, 2),
            'interaction_type': random.randint(0, 4),
            'emotional_state': self.emotional_states.index(emotional_state),
            'stress_level': stress_level,
            'typing_patterns': self.generate_typing_patterns(person_profile),
            'mouse_patterns': self.generate_mouse_patterns(person_profile),
            'navigation_patterns': self.generate_navigation_patterns(person_profile),
            'communication_patterns': self.generate_communication_patterns(person_profile)
        }
        
        return activity
    
    def generate_typing_patterns(self, person_profile):
        traits = person_profile.get('personality_traits', {})
        
        base_speed = 50
        if traits.get('perfectionist', 0) > 0.7:
            base_speed = 35
            accuracy = random.uniform(0.95, 1.0)
        else:
            accuracy = random.uniform(0.85, 0.98)
        
        return {
            'speed': base_speed + random.uniform(-10, 20),
            'accuracy': accuracy,
            'rhythm_variation': random.uniform(0.1, 0.5),
            'pause_frequency': random.uniform(0.1, 0.4),
            'backspace_ratio': random.uniform(0.02, 0.1)
        }
    
    def generate_mouse_patterns(self, person_profile):
        traits = person_profile.get('personality_traits', {})
        
        if traits.get('impatient', 0) > 0.7:
            movement_speed = random.uniform(1.5, 3.0)
            click_frequency = random.uniform(2.0, 4.0)
        else:
            movement_speed = random.uniform(0.8, 2.0)
            click_frequency = random.uniform(1.0, 2.5)
        
        return {
            'movement_speed': movement_speed,
            'click_frequency': click_frequency,
            'scroll_intensity': random.uniform(0.5, 2.0),
            'hover_duration': random.uniform(0.5, 3.0),
            'path_efficiency': random.uniform(0.6, 0.95)
        }
    
    def generate_navigation_patterns(self, person_profile):
        traits = person_profile.get('personality_traits', {})
        
        if traits.get('explorer', 0) > 0.7:
            page_visit_speed = random.uniform(2.0, 5.0)
            search_frequency = random.uniform(3.0, 6.0)
        else:
            page_visit_speed = random.uniform(0.5, 2.0)
            search_frequency = random.uniform(0.5, 2.0)
        
        return {
            'page_visit_speed': page_visit_speed,
            'back_button_usage': random.uniform(0.1, 0.8),
            'bookmark_usage': random.uniform(0.2, 0.9),
            'search_frequency': search_frequency,
            'tab_switching': random.uniform(1.0, 5.0)
        }
    
    def generate_communication_patterns(self, person_profile):
        traits = person_profile.get('personality_traits', {})
        
        if traits.get('verbose', 0) > 0.7:
            message_length = random.uniform(100, 300)
            emoji_usage = random.uniform(0.3, 0.8)
        else:
            message_length = random.uniform(20, 100)
            emoji_usage = random.uniform(0.1, 0.4)
        
        return {
            'message_length': message_length,
            'response_time': random.uniform(0.1, 2.0),
            'emoji_usage': emoji_usage,
            'formality_level': random.uniform(0.3, 0.9),
            'question_ratio': random.uniform(0.1, 0.5)
        }
    
    def create_training_dataset(self, num_persons=1000, days_per_person=30):
        persons = []
        
        personality_types = ['early_bird', 'night_owl', 'workaholic', 'balanced', 'social', 'introvert']
        
        for i in range(num_persons):
            person_type = random.choice(personality_types)
            
            person_profile = {
                'person_id': f"person_{i}",
                'personality_type': person_type,
                'personality_traits': {
                    'social': random.uniform(0, 1),
                    'work_focused': random.uniform(0, 1),
                    'entertainment_seeking': random.uniform(0, 1),
                    'perfectionist': random.uniform(0, 1),
                    'impatient': random.uniform(0, 1),
                    'explorer': random.uniform(0, 1),
                    'verbose': random.uniform(0, 1)
                }
            }
            
            timeline = self.generate_behavioral_timeline(person_profile, days_per_person)
            
            persons.append({
                'person_profile': person_profile,
                'timeline': timeline
            })
        
        return persons
