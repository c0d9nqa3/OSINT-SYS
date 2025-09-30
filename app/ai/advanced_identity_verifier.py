import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
import numpy as np
import cv2
from PIL import Image
import face_recognition
import dlib
from typing import Dict, List, Tuple, Optional, Union
import json
import pickle
from datetime import datetime
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class MultiModalIdentityEncoder(nn.Module):
    def __init__(self, text_dim=768, image_dim=512, audio_dim=256, fusion_dim=1024):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        self.image_encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512)
        )
        
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.LSTM(50, 128, batch_first=True, bidirectional=True),
            nn.Linear(256, 256)
        )
        
        self.fusion_network = nn.Sequential(
            nn.Linear(text_dim + image_dim + audio_dim + 512 + 256 + 256, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        
        self.identity_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.feature_importance = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
    
    def forward(self, text_input, image_input, audio_input, face_input, behavioral_input, temporal_input):
        text_features = self.text_encoder(**text_input).pooler_output
        
        image_features = self.image_encoder.get_image_features(**image_input)
        
        audio_features = self.audio_encoder(audio_input)
        
        face_features = self.face_encoder(face_input)
        
        behavioral_features = self.behavioral_encoder(behavioral_input)
        
        temporal_features, _ = self.temporal_encoder(temporal_input)
        temporal_features = temporal_features[:, -1, :]
        
        combined_features = torch.cat([
            text_features, image_features, audio_features, 
            face_features, behavioral_features, temporal_features
        ], dim=-1)
        
        fused_features = self.fusion_network(combined_features)
        
        identity_probability = self.identity_classifier(fused_features)
        confidence_score = self.confidence_estimator(fused_features)
        feature_importance = F.softmax(self.feature_importance(fused_features), dim=-1)
        
        return {
            'identity_probability': identity_probability,
            'confidence_score': confidence_score,
            'feature_importance': feature_importance,
            'fused_features': fused_features
        }

class AdvancedIdentityVerifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        self.model = MultiModalIdentityEncoder()
        
        self.verification_methods = [
            'text_similarity', 'facial_recognition', 'behavioral_analysis',
            'temporal_consistency', 'social_network_analysis', 'biometric_fusion'
        ]
        
        if model_path:
            self.load_model(model_path)
    
    def extract_text_features(self, text_data):
        if isinstance(text_data, str):
            text_data = [text_data]
        
        features = []
        for text in text_data:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            features.append(encoding)
        
        return features
    
    def extract_image_features(self, image_paths):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        features = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert('RGB')
                encoding = self.clip_processor(images=image, return_tensors='pt')
                features.append(encoding)
            except Exception as e:
                dummy_encoding = self.clip_processor(images=Image.new('RGB', (224, 224)), return_tensors='pt')
                features.append(dummy_encoding)
        
        return features
    
    def extract_audio_features(self, audio_paths):
        features = []
        for audio_path in audio_paths:
            try:
                audio_data = self.load_audio_file(audio_path)
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                features.append(audio_tensor)
            except Exception as e:
                dummy_audio = torch.randn(1, 1, 16000)
                features.append(dummy_audio)
        
        return features
    
    def load_audio_file(self, audio_path):
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, duration=10)
        return audio
    
    def extract_face_features(self, image_paths):
        features = []
        for image_path in image_paths:
            try:
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(image_rgb)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0]
                    face_tensor = torch.tensor(face_encoding, dtype=torch.float32)
                else:
                    face_tensor = torch.zeros(128, dtype=torch.float32)
                
                image_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
                
                features.append({
                    'face_encoding': face_tensor,
                    'face_image': image_tensor
                })
            except Exception as e:
                dummy_features = {
                    'face_encoding': torch.zeros(128, dtype=torch.float32),
                    'face_image': torch.randn(1, 3, 224, 224)
                }
                features.append(dummy_features)
        
        return features
    
    def extract_behavioral_features(self, behavioral_data):
        features = []
        for data in behavioral_data:
            feature_vector = np.zeros(100)
            
            if 'typing_patterns' in data:
                typing = data['typing_patterns']
                feature_vector[0] = typing.get('avg_keystroke_interval', 0)
                feature_vector[1] = typing.get('typing_speed', 0)
                feature_vector[2] = typing.get('error_rate', 0)
                feature_vector[3] = typing.get('rhythm_consistency', 0)
            
            if 'mouse_patterns' in data:
                mouse = data['mouse_patterns']
                feature_vector[4] = mouse.get('movement_speed', 0)
                feature_vector[5] = mouse.get('click_frequency', 0)
                feature_vector[6] = mouse.get('scroll_patterns', 0)
            
            if 'navigation_patterns' in data:
                nav = data['navigation_patterns']
                feature_vector[7] = nav.get('page_visit_frequency', 0)
                feature_vector[8] = nav.get('session_duration', 0)
                feature_vector[9] = nav.get('back_button_usage', 0)
            
            if 'communication_patterns' in data:
                comm = data['communication_patterns']
                feature_vector[10] = comm.get('message_length_avg', 0)
                feature_vector[11] = comm.get('response_time', 0)
                feature_vector[12] = comm.get('emojis_usage', 0)
                feature_vector[13] = comm.get('formality_level', 0)
            
            features.append(torch.tensor(feature_vector, dtype=torch.float32))
        
        return features
    
    def extract_temporal_features(self, temporal_data):
        features = []
        for data in temporal_data:
            feature_sequence = np.zeros((50, 50))
            
            if 'activity_timeline' in data:
                timeline = data['activity_timeline']
                for i, activity in enumerate(timeline[:50]):
                    feature_sequence[i, 0] = activity.get('hour', 0) / 24
                    feature_sequence[i, 1] = activity.get('day_of_week', 0) / 7
                    feature_sequence[i, 2] = activity.get('activity_type', 0)
                    feature_sequence[i, 3] = activity.get('duration', 0)
                    feature_sequence[i, 4] = activity.get('intensity', 0)
            
            features.append(torch.tensor(feature_sequence, dtype=torch.float32))
        
        return features
    
    def verify_identity(self, target_profile, candidate_data):
        self.model.eval()
        
        with torch.no_grad():
            text_features = self.extract_text_features(target_profile.get('text_data', ['']))
            image_features = self.extract_image_features(target_profile.get('image_paths', []))
            audio_features = self.extract_audio_features(target_profile.get('audio_paths', []))
            face_features = self.extract_face_features(target_profile.get('image_paths', []))
            behavioral_features = self.extract_behavioral_features(target_profile.get('behavioral_data', [{}]))
            temporal_features = self.extract_temporal_features(target_profile.get('temporal_data', [{}]))
            
            if not text_features:
                text_features = [self.tokenizer("", return_tensors='pt', padding='max_length', max_length=512)]
            if not image_features:
                image_features = [self.clip_processor(images=Image.new('RGB', (224, 224)), return_tensors='pt')]
            if not audio_features:
                audio_features = [torch.randn(1, 1, 16000)]
            if not face_features:
                face_features = [{'face_encoding': torch.zeros(128), 'face_image': torch.randn(1, 3, 224, 224)}]
            if not behavioral_features:
                behavioral_features = [torch.zeros(100, dtype=torch.float32)]
            if not temporal_features:
                temporal_features = [torch.zeros(50, 50, dtype=torch.float32)]
            
            text_input = {
                'input_ids': torch.cat([f['input_ids'] for f in text_features[:1]], dim=0).to(self.device),
                'attention_mask': torch.cat([f['attention_mask'] for f in text_features[:1]], dim=0).to(self.device)
            }
            
            image_input = {
                'pixel_values': torch.cat([f['pixel_values'] for f in image_features[:1]], dim=0).to(self.device)
            }
            
            audio_input = audio_features[0].to(self.device)
            face_input = face_features[0]['face_image'].to(self.device)
            behavioral_input = behavioral_features[0].unsqueeze(0).to(self.device)
            temporal_input = temporal_features[0].unsqueeze(0).to(self.device)
            
            outputs = self.model(text_input, image_input, audio_input, face_input, behavioral_input, temporal_input)
            
            identity_probability = outputs['identity_probability'].item()
            confidence_score = outputs['confidence_score'].item()
            feature_importance = outputs['feature_importance'][0].cpu().numpy()
            
            verification_result = {
                'is_same_person': identity_probability > 0.7,
                'confidence_score': confidence_score,
                'identity_probability': identity_probability,
                'feature_importance': {
                    'text_similarity': feature_importance[0],
                    'image_similarity': feature_importance[1],
                    'audio_similarity': feature_importance[2],
                    'face_recognition': feature_importance[3],
                    'behavioral_analysis': feature_importance[4],
                    'temporal_consistency': feature_importance[5]
                },
                'verification_methods_used': self.verification_methods,
                'detailed_analysis': self.perform_detailed_analysis(target_profile, candidate_data, outputs)
            }
            
            return verification_result
    
    def perform_detailed_analysis(self, target_profile, candidate_data, model_outputs):
        analysis = {
            'text_analysis': self.analyze_text_similarity(target_profile, candidate_data),
            'facial_analysis': self.analyze_facial_similarity(target_profile, candidate_data),
            'behavioral_analysis': self.analyze_behavioral_patterns(target_profile, candidate_data),
            'temporal_analysis': self.analyze_temporal_consistency(target_profile, candidate_data),
            'social_network_analysis': self.analyze_social_connections(target_profile, candidate_data),
            'risk_assessment': self.assess_verification_risks(target_profile, candidate_data)
        }
        
        return analysis
    
    def analyze_text_similarity(self, target_profile, candidate_data):
        if not target_profile.get('text_data') or not candidate_data.get('text_data'):
            return {'similarity_score': 0.0, 'analysis': 'Insufficient text data'}
        
        target_text = ' '.join(target_profile['text_data'])
        candidate_text = ' '.join(candidate_data['text_data'])
        
        target_encoding = self.tokenizer(target_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        candidate_encoding = self.tokenizer(candidate_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            target_features = self.model.text_encoder(**target_encoding).pooler_output
            candidate_features = self.model.text_encoder(**candidate_encoding).pooler_output
            
            similarity = F.cosine_similarity(target_features, candidate_features, dim=-1).item()
        
        return {
            'similarity_score': similarity,
            'analysis': 'High similarity' if similarity > 0.8 else 'Medium similarity' if similarity > 0.5 else 'Low similarity'
        }
    
    def analyze_facial_similarity(self, target_profile, candidate_data):
        if not target_profile.get('image_paths') or not candidate_data.get('image_paths'):
            return {'similarity_score': 0.0, 'analysis': 'No facial images available'}
        
        target_faces = self.extract_face_features(target_profile['image_paths'])
        candidate_faces = self.extract_face_features(candidate_data['image_paths'])
        
        similarities = []
        for target_face in target_faces:
            for candidate_face in candidate_faces:
                similarity = F.cosine_similarity(
                    target_face['face_encoding'].unsqueeze(0),
                    candidate_face['face_encoding'].unsqueeze(0)
                ).item()
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            'similarity_score': avg_similarity,
            'analysis': 'High facial similarity' if avg_similarity > 0.8 else 'Medium similarity' if avg_similarity > 0.5 else 'Low similarity'
        }
    
    def analyze_behavioral_patterns(self, target_profile, candidate_data):
        target_behavior = target_profile.get('behavioral_data', [{}])
        candidate_behavior = candidate_data.get('behavioral_data', [{}])
        
        if not target_behavior or not candidate_behavior:
            return {'similarity_score': 0.0, 'analysis': 'Insufficient behavioral data'}
        
        target_features = self.extract_behavioral_features(target_behavior)
        candidate_features = self.extract_behavioral_features(candidate_behavior)
        
        similarities = []
        for target_feat in target_features:
            for candidate_feat in candidate_features:
                similarity = F.cosine_similarity(target_feat.unsqueeze(0), candidate_feat.unsqueeze(0)).item()
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            'similarity_score': avg_similarity,
            'analysis': 'Behavioral patterns match' if avg_similarity > 0.7 else 'Some behavioral similarities' if avg_similarity > 0.4 else 'Different behavioral patterns'
        }
    
    def analyze_temporal_consistency(self, target_profile, candidate_data):
        target_temporal = target_profile.get('temporal_data', [{}])
        candidate_temporal = candidate_data.get('temporal_data', [{}])
        
        if not target_temporal or not candidate_temporal:
            return {'consistency_score': 0.0, 'analysis': 'Insufficient temporal data'}
        
        target_activities = target_temporal[0].get('activity_timeline', [])
        candidate_activities = candidate_temporal[0].get('activity_timeline', [])
        
        if not target_activities or not candidate_activities:
            return {'consistency_score': 0.0, 'analysis': 'No activity timeline data'}
        
        time_overlaps = 0
        total_comparisons = 0
        
        for target_activity in target_activities:
            for candidate_activity in candidate_activities:
                time_diff = abs(target_activity.get('hour', 0) - candidate_activity.get('hour', 0))
                if time_diff <= 2:
                    time_overlaps += 1
                total_comparisons += 1
        
        consistency_score = time_overlaps / total_comparisons if total_comparisons > 0 else 0.0
        
        return {
            'consistency_score': consistency_score,
            'analysis': 'High temporal consistency' if consistency_score > 0.7 else 'Medium consistency' if consistency_score > 0.4 else 'Low temporal consistency'
        }
    
    def analyze_social_connections(self, target_profile, candidate_data):
        target_network = target_profile.get('social_network', {})
        candidate_network = candidate_data.get('social_network', {})
        
        if not target_network or not candidate_network:
            return {'connection_score': 0.0, 'analysis': 'No social network data'}
        
        target_contacts = set(target_network.get('contacts', []))
        candidate_contacts = set(candidate_network.get('contacts', []))
        
        if not target_contacts or not candidate_contacts:
            return {'connection_score': 0.0, 'analysis': 'No contact information'}
        
        common_contacts = target_contacts.intersection(candidate_contacts)
        total_contacts = target_contacts.union(candidate_contacts)
        
        connection_score = len(common_contacts) / len(total_contacts) if total_contacts else 0.0
        
        return {
            'connection_score': connection_score,
            'common_contacts': list(common_contacts),
            'analysis': 'Strong social connections' if connection_score > 0.3 else 'Some shared contacts' if connection_score > 0.1 else 'Weak social connections'
        }
    
    def assess_verification_risks(self, target_profile, candidate_data):
        risks = []
        risk_score = 0.0
        
        if not target_profile.get('image_paths'):
            risks.append('No facial images for verification')
            risk_score += 0.3
        
        if not target_profile.get('behavioral_data'):
            risks.append('Limited behavioral data')
            risk_score += 0.2
        
        if not target_profile.get('temporal_data'):
            risks.append('No temporal activity data')
            risk_score += 0.2
        
        text_similarity = self.analyze_text_similarity(target_profile, candidate_data)['similarity_score']
        if text_similarity < 0.3:
            risks.append('Low text similarity')
            risk_score += 0.3
        
        return {
            'risk_score': min(risk_score, 1.0),
            'identified_risks': risks,
            'recommendation': 'High confidence' if risk_score < 0.3 else 'Medium confidence' if risk_score < 0.6 else 'Low confidence - additional verification needed'
        }
    
    def train_model(self, training_data, validation_data, epochs=100, batch_size=16, learning_rate=0.001):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in training_data:
                optimizer.zero_grad()
                
                text_input = {
                    'input_ids': batch['text_input_ids'].to(self.device),
                    'attention_mask': batch['text_attention_mask'].to(self.device)
                }
                
                image_input = {
                    'pixel_values': batch['image_pixel_values'].to(self.device)
                }
                
                audio_input = batch['audio_input'].to(self.device)
                face_input = batch['face_input'].to(self.device)
                behavioral_input = batch['behavioral_input'].to(self.device)
                temporal_input = batch['temporal_input'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(text_input, image_input, audio_input, face_input, behavioral_input, temporal_input)
                
                loss = criterion(outputs['identity_probability'].squeeze(), labels.float())
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(training_data)
            scheduler.step(avg_train_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}')
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'verification_methods': self.verification_methods
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.verification_methods = checkpoint['verification_methods']
        self.model.to(self.device)

class IdentityVerificationDataGenerator:
    def __init__(self):
        self.person_templates = self.generate_person_templates()
    
    def generate_person_templates(self):
        return [
            {
                'name': 'John Smith',
                'age_range': (25, 35),
                'profession': 'Software Engineer',
                'personality_traits': ['analytical', 'introverted', 'detail-oriented'],
                'behavioral_patterns': ['early_riser', 'consistent_routine', 'tech_savvy']
            },
            {
                'name': 'Sarah Johnson',
                'age_range': (30, 40),
                'profession': 'Marketing Manager',
                'personality_traits': ['extroverted', 'creative', 'social'],
                'behavioral_patterns': ['social_media_active', 'networker', 'trend_follower']
            },
            {
                'name': 'Michael Chen',
                'age_range': (35, 45),
                'profession': 'Data Scientist',
                'personality_traits': ['logical', 'curious', 'methodical'],
                'behavioral_patterns': ['data_driven', 'research_oriented', 'systematic']
            }
        ]
    
    def generate_synthetic_identity_data(self, num_samples=5000):
        identities = []
        
        for _ in range(num_samples):
            template = random.choice(self.person_templates)
            
            identity = {
                'id': f"identity_{random.randint(10000, 99999)}",
                'name': template['name'],
                'age': random.randint(template['age_range'][0], template['age_range'][1]),
                'profession': template['profession'],
                'text_data': self.generate_text_data(template),
                'behavioral_data': self.generate_behavioral_data(template),
                'temporal_data': self.generate_temporal_data(template),
                'social_network': self.generate_social_network_data(template)
            }
            
            identities.append(identity)
        
        return identities
    
    def generate_text_data(self, template):
        text_samples = [
            f"I'm {template['name']}, a {template['profession']} with experience in technology.",
            f"Working as a {template['profession']}, I focus on {random.choice(['innovation', 'efficiency', 'quality'])}.",
            f"My name is {template['name']} and I'm passionate about my work in {template['profession'].lower()}.",
            f"As a {template['profession']}, I believe in {random.choice(['continuous learning', 'team collaboration', 'excellence'])}."
        ]
        return random.sample(text_samples, random.randint(2, 4))
    
    def generate_behavioral_data(self, template):
        behavioral_patterns = {
            'typing_patterns': {
                'avg_keystroke_interval': random.uniform(0.1, 0.3),
                'typing_speed': random.uniform(40, 80),
                'error_rate': random.uniform(0.01, 0.05),
                'rhythm_consistency': random.uniform(0.7, 0.95)
            },
            'mouse_patterns': {
                'movement_speed': random.uniform(0.5, 2.0),
                'click_frequency': random.uniform(0.5, 3.0),
                'scroll_patterns': random.uniform(0.2, 1.0)
            },
            'navigation_patterns': {
                'page_visit_frequency': random.uniform(5, 50),
                'session_duration': random.uniform(10, 120),
                'back_button_usage': random.uniform(0.1, 0.8)
            },
            'communication_patterns': {
                'message_length_avg': random.randint(20, 200),
                'response_time': random.uniform(0.5, 24),
                'emojis_usage': random.uniform(0, 0.5),
                'formality_level': random.uniform(0.3, 0.9)
            }
        }
        return [behavioral_patterns]
    
    def generate_temporal_data(self, template):
        activity_timeline = []
        for hour in range(24):
            if 6 <= hour <= 22:
                activity_type = random.choice(['work', 'social', 'leisure', 'exercise'])
                activity_timeline.append({
                    'hour': hour,
                    'day_of_week': random.randint(0, 6),
                    'activity_type': activity_type,
                    'duration': random.uniform(0.5, 4.0),
                    'intensity': random.uniform(0.3, 1.0)
                })
        
        return [{'activity_timeline': activity_timeline}]
    
    def generate_social_network_data(self, template):
        contact_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
        contacts = random.sample(contact_names, random.randint(3, 6))
        
        return {
            'contacts': contacts,
            'network_size': len(contacts),
            'influence_score': random.uniform(0.2, 0.9)
        }
    
    def create_identity_pairs(self, identities):
        pairs = []
        labels = []
        
        for i in range(len(identities) - 1):
            for j in range(i + 1, len(identities)):
                identity1 = identities[i]
                identity2 = identities[j]
                
                is_same_person = (identity1['name'] == identity2['name'] and 
                                abs(identity1['age'] - identity2['age']) <= 2)
                
                pairs.append((identity1, identity2))
                labels.append(1 if is_same_person else 0)
        
        return pairs, labels
    
    def create_training_dataset(self, num_identities=1000):
        identities = self.generate_synthetic_identity_data(num_identities)
        pairs, labels = self.create_identity_pairs(identities)
        
        return {
            'identity_pairs': pairs,
            'labels': labels,
            'identities': identities
        }
