import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
import numpy as np
import json
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
from collections import defaultdict
import pickle

class SocialEngineeringDataset:
    def __init__(self):
        self.attack_templates = {
            'phishing_email': [
                "Urgent: Your account security has been compromised. Click here to verify your identity immediately.",
                "Congratulations! You've won a prize. Click the link to claim your reward before it expires.",
                "Important notice from IT department: Please update your password immediately.",
                "Your subscription is about to expire. Renew now to avoid service interruption.",
                "Security alert: Unusual activity detected on your account. Verify now."
            ],
            'voice_phishing': [
                "This is an urgent call from your bank. We need to verify your account information.",
                "Your computer has been infected with a virus. We need remote access to fix it.",
                "This is the IRS calling about an overdue tax payment. Immediate action required.",
                "Your social security number has been compromised. We need to verify your identity.",
                "Emergency: Your family member is in trouble and needs money immediately."
            ],
            'social_media_impersonation': [
                "Hi! I'm having trouble accessing my account. Can you help me reset my password?",
                "I'm organizing a surprise party for our mutual friend. Can you send me their contact info?",
                "I'm stuck in a foreign country and need money. Can you wire me some cash?",
                "I've been hacked! Please don't trust any messages from my account until I fix this.",
                "I'm selling my old laptop cheap. Interested? I can ship it to you."
            ],
            'pretexting': [
                "I'm calling from the IT department. We need to update your software remotely.",
                "This is a survey for a research project. Can you answer a few questions?",
                "I'm a journalist writing about your company. Can you confirm some details?",
                "I'm from HR and need to verify your employment information.",
                "I'm a student doing a thesis on cybersecurity. Can you share your experiences?"
            ]
        }
        
        self.personality_traits = {
            'agreeableness': ['helpful', 'trusting', 'cooperative', 'forgiving', 'generous'],
            'conscientiousness': ['organized', 'disciplined', 'dutiful', 'achievement-oriented', 'careful'],
            'extraversion': ['outgoing', 'sociable', 'talkative', 'assertive', 'energetic'],
            'neuroticism': ['anxious', 'moody', 'stressed', 'worried', 'emotional'],
            'openness': ['creative', 'curious', 'imaginative', 'artistic', 'adventurous']
        }
        
        self.psychological_triggers = {
            'fear': ['urgent', 'immediate', 'threat', 'danger', 'emergency', 'critical', 'expires', 'deadline'],
            'greed': ['free', 'win', 'prize', 'bonus', 'discount', 'opportunity', 'exclusive', 'limited'],
            'curiosity': ['secret', 'exclusive', 'behind the scenes', 'revealed', 'discovered', 'hidden'],
            'authority': ['official', 'manager', 'director', 'supervisor', 'department', 'policy', 'required'],
            'social_proof': ['everyone', 'popular', 'trending', 'recommended', 'approved', 'endorsed'],
            'reciprocity': ['favor', 'help', 'gift', 'service', 'compliment', 'kindness']
        }
    
    def generate_attack_scenarios(self, num_scenarios=5000):
        scenarios = []
        
        for _ in range(num_scenarios):
            attack_type = random.choice(list(self.attack_templates.keys()))
            personality = random.choice(list(self.personality_traits.keys()))
            trigger = random.choice(list(self.psychological_triggers.keys()))
            
            template = random.choice(self.attack_templates[attack_type])
            trait_words = random.sample(self.personality_traits[personality], 2)
            trigger_words = random.sample(self.psychological_triggers[trigger], 2)
            
            scenario = {
                'attack_type': attack_type,
                'target_personality': personality,
                'psychological_trigger': trigger,
                'template': template,
                'trait_keywords': trait_words,
                'trigger_keywords': trigger_words,
                'success_probability': random.uniform(0.1, 0.9),
                'complexity_level': random.randint(1, 5),
                'urgency_level': random.uniform(0.1, 1.0)
            }
            
            scenarios.append(scenario)
        
        return scenarios

class AdvancedSocialEngineeringModel(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=512, hidden_dim=1024, num_attack_types=10):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1024, embedding_dim))
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ) for _ in range(6)
        ])
        
        self.personality_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        self.context_encoder = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.attack_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_attack_types)
        )
        
        self.success_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.message_generator = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, vocab_size)
        )
        
        self.timing_optimizer = nn.Sequential(
            nn.Linear(embedding_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )
    
    def forward(self, text_tokens, personality_traits, context_features, temporal_features):
        batch_size, seq_len = text_tokens.shape
        
        text_embeddings = self.embedding(text_tokens)
        text_embeddings = text_embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        
        for layer in self.encoder_layers:
            text_embeddings = layer(text_embeddings)
        
        text_features = text_embeddings.mean(dim=1)
        
        personality_encoded = self.personality_encoder(personality_traits)
        context_encoded = self.context_encoder(context_features)
        temporal_encoded = self.temporal_encoder(temporal_features)
        
        combined_features = torch.stack([text_features, personality_encoded, context_encoded, temporal_encoded], dim=1)
        
        attended_features, attention_weights = self.fusion_attention(
            combined_features, combined_features, combined_features
        )
        
        final_features = attended_features.view(batch_size, -1)
        
        attack_probabilities = F.softmax(self.attack_classifier(final_features), dim=-1)
        success_probability = self.success_predictor(final_features)
        message_logits = self.message_generator(final_features)
        optimal_timing = F.softmax(self.timing_optimizer(final_features), dim=-1)
        
        return {
            'attack_probabilities': attack_probabilities,
            'success_probability': success_probability,
            'message_logits': message_logits,
            'optimal_timing': optimal_timing,
            'attention_weights': attention_weights
        }

class SocialEngineeringAttackPlanner:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = AdvancedSocialEngineeringModel()
        
        self.attack_types = [
            'phishing_email', 'voice_phishing', 'social_media_impersonation',
            'pretexting', 'baiting', 'quid_pro_quo', 'tailgating',
            'shoulder_surfing', 'dumpster_diving', 'spear_phishing'
        ]
        
        self.timing_options = [
            'early_morning', 'morning', 'lunch_time', 'afternoon',
            'evening', 'late_evening', 'weekend'
        ]
        
        if model_path:
            self.load_model(model_path)
    
    def analyze_target_profile(self, target_data):
        personality_scores = self.extract_personality_traits(target_data)
        context_features = self.extract_context_features(target_data)
        temporal_patterns = self.extract_temporal_patterns(target_data)
        
        return {
            'personality_traits': personality_scores,
            'context_features': context_features,
            'temporal_patterns': temporal_patterns
        }
    
    def extract_personality_traits(self, target_data):
        traits = np.zeros(5)
        
        if 'social_behavior' in target_data:
            behavior = target_data['social_behavior']
            traits[0] = behavior.get('helpfulness_score', 0.5)
            traits[1] = behavior.get('organization_level', 0.5)
            traits[2] = behavior.get('social_activity', 0.5)
            traits[3] = behavior.get('stress_level', 0.5)
            traits[4] = behavior.get('creativity_score', 0.5)
        
        return torch.tensor(traits, dtype=torch.float32)
    
    def extract_context_features(self, target_data):
        features = np.zeros(20)
        
        if 'professional_context' in target_data:
            prof = target_data['professional_context']
            features[0] = prof.get('hierarchy_level', 0.5)
            features[1] = prof.get('decision_making_authority', 0.5)
            features[2] = prof.get('access_to_sensitive_info', 0.5)
            features[3] = prof.get('security_training_level', 0.5)
            features[4] = prof.get('compliance_orientation', 0.5)
        
        if 'social_context' in target_data:
            social = target_data['social_context']
            features[5] = social.get('network_influence', 0.5)
            features[6] = social.get('trust_level', 0.5)
            features[7] = social.get('information_sharing_tendency', 0.5)
            features[8] = social.get('conflict_avoidance', 0.5)
            features[9] = social.get('peer_pressure_susceptibility', 0.5)
        
        if 'technological_context' in target_data:
            tech = target_data['technological_context']
            features[10] = tech.get('technical_sophistication', 0.5)
            features[11] = tech.get('security_awareness', 0.5)
            features[12] = tech.get('device_usage_patterns', 0.5)
            features[13] = tech.get('software_update_habits', 0.5)
            features[14] = tech.get('password_management_practices', 0.5)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_temporal_patterns(self, target_data):
        patterns = np.zeros(10)
        
        if 'activity_patterns' in target_data:
            activity = target_data['activity_patterns']
            patterns[0] = activity.get('response_time_consistency', 0.5)
            patterns[1] = activity.get('peak_activity_hours', 0.5)
            patterns[2] = activity.get('weekend_activity_level', 0.5)
            patterns[3] = activity.get('stress_periods', 0.5)
            patterns[4] = activity.get('decision_making_speed', 0.5)
        
        return torch.tensor(patterns, dtype=torch.float32)
    
    def generate_attack_plan(self, target_data, attack_objective):
        self.model.eval()
        
        with torch.no_grad():
            analysis = self.analyze_target_profile(target_data)
            
            personality_traits = analysis['personality_traits'].unsqueeze(0).to(self.device)
            context_features = analysis['context_features'].unsqueeze(0).to(self.device)
            temporal_patterns = analysis['temporal_patterns'].unsqueeze(0).to(self.device)
            
            dummy_text = self.tokenizer.encode(attack_objective, return_tensors='pt', max_length=512, truncation=True)
            dummy_text = dummy_text.to(self.device)
            
            outputs = self.model(dummy_text, personality_traits, context_features, temporal_patterns)
            
            attack_probabilities = outputs['attack_probabilities'][0]
            success_probability = outputs['success_probability'][0].item()
            optimal_timing = outputs['optimal_timing'][0]
            
            recommended_attacks = []
            for i, attack_type in enumerate(self.attack_types):
                recommended_attacks.append({
                    'attack_type': attack_type,
                    'probability': float(attack_probabilities[i]),
                    'success_rate': success_probability
                })
            
            recommended_attacks.sort(key=lambda x: x['probability'], reverse=True)
            
            optimal_timing_idx = torch.argmax(optimal_timing).item()
            optimal_timing_str = self.timing_options[optimal_timing_idx]
            
            return {
                'recommended_attacks': recommended_attacks[:5],
                'overall_success_probability': success_probability,
                'optimal_timing': optimal_timing_str,
                'attack_plan': self.create_detailed_attack_plan(recommended_attacks[0], target_data, optimal_timing_str)
            }
    
    def create_detailed_attack_plan(self, primary_attack, target_data, optimal_timing):
        attack_type = primary_attack['attack_type']
        
        if attack_type == 'phishing_email':
            return {
                'method': 'phishing_email',
                'subject_line': self.generate_phishing_subject(target_data),
                'email_content': self.generate_phishing_content(target_data),
                'sender_identity': self.choose_sender_identity(target_data),
                'timing': optimal_timing,
                'follow_up_strategy': 'escalate_urgency_if_no_response',
                'success_indicators': ['email_opened', 'link_clicked', 'information_provided']
            }
        
        elif attack_type == 'voice_phishing':
            return {
                'method': 'voice_phishing',
                'caller_identity': self.generate_caller_identity(target_data),
                'script': self.generate_voice_script(target_data),
                'timing': optimal_timing,
                'escalation_tactics': ['authority_appeal', 'urgency_creation', 'fear_appeal'],
                'success_indicators': ['call_answered', 'information_shared', 'action_taken']
            }
        
        elif attack_type == 'social_media_impersonation':
            return {
                'method': 'social_media_impersonation',
                'impersonated_identity': self.choose_impersonation_target(target_data),
                'approach_message': self.generate_approach_message(target_data),
                'relationship_building': 'gradual_trust_establishment',
                'timing': optimal_timing,
                'success_indicators': ['message_responded', 'relationship_established', 'trust_built']
            }
        
        else:
            return {
                'method': attack_type,
                'approach': 'standard_social_engineering',
                'timing': optimal_timing,
                'success_probability': primary_attack['success_rate']
            }
    
    def generate_phishing_subject(self, target_data):
        subjects = [
            "Urgent: Security Alert - Action Required",
            "Your Account Needs Immediate Verification",
            "Important: Suspicious Activity Detected",
            "Action Required: Account Security Update",
            "Urgent: Your Access Will Be Suspended"
        ]
        return random.choice(subjects)
    
    def generate_phishing_content(self, target_data):
        templates = [
            "We have detected unusual activity on your account. Please verify your identity by clicking the link below.",
            "Your account security has been compromised. Immediate action is required to prevent unauthorized access.",
            "We need to update your account information for security purposes. Please confirm your details.",
            "Urgent: Your account will be suspended unless you verify your information within 24 hours."
        ]
        return random.choice(templates)
    
    def generate_caller_identity(self, target_data):
        identities = [
            "IT Security Department",
            "Bank Fraud Prevention",
            "Government Agency Representative",
            "Company HR Department",
            "Technical Support Team"
        ]
        return random.choice(identities)
    
    def generate_voice_script(self, target_data):
        scripts = [
            "Hello, this is an urgent call regarding your account security. We need to verify some information immediately.",
            "Good morning, I'm calling from the fraud prevention department. We've detected suspicious activity on your account.",
            "This is an important security call. Your account has been flagged for unusual activity and needs immediate attention."
        ]
        return random.choice(scripts)
    
    def choose_impersonation_target(self, target_data):
        if 'social_network' in target_data:
            network = target_data['social_network']
            close_contacts = network.get('close_contacts', [])
            if close_contacts:
                return random.choice(close_contacts)
        return "mutual_friend"
    
    def generate_approach_message(self, target_data):
        messages = [
            "Hi! I'm having trouble accessing my account. Can you help me?",
            "Hey! I need to ask you something important. Are you free to chat?",
            "Hi there! I'm in a bit of a situation and could use your help."
        ]
        return random.choice(messages)
    
    def train_model(self, training_data, validation_data, epochs=100, batch_size=16, learning_rate=0.001):
        self.model.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        criterion_attack = nn.CrossEntropyLoss()
        criterion_success = nn.BCELoss()
        criterion_message = nn.CrossEntropyLoss()
        criterion_timing = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in training_data:
                optimizer.zero_grad()
                
                text_tokens = batch['text_tokens'].to(self.device)
                personality_traits = batch['personality_traits'].to(self.device)
                context_features = batch['context_features'].to(self.device)
                temporal_features = batch['temporal_features'].to(self.device)
                
                attack_labels = batch['attack_labels'].to(self.device)
                success_labels = batch['success_labels'].to(self.device)
                message_labels = batch['message_labels'].to(self.device)
                timing_labels = batch['timing_labels'].to(self.device)
                
                outputs = self.model(text_tokens, personality_traits, context_features, temporal_features)
                
                loss_attack = criterion_attack(outputs['attack_probabilities'], attack_labels)
                loss_success = criterion_success(outputs['success_probability'].squeeze(), success_labels)
                loss_message = criterion_message(outputs['message_logits'], message_labels)
                loss_timing = criterion_timing(outputs['optimal_timing'], timing_labels)
                
                total_loss = loss_attack + loss_success + loss_message + loss_timing
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
            'attack_types': self.attack_types,
            'timing_options': self.timing_options,
            'tokenizer_vocab': self.tokenizer.get_vocab()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.attack_types = checkpoint['attack_types']
        self.timing_options = checkpoint['timing_options']
        self.model.to(self.device)

class SocialEngineeringDataGenerator:
    def __init__(self):
        self.dataset = SocialEngineeringDataset()
    
    def generate_training_data(self, num_samples=10000):
        scenarios = self.dataset.generate_attack_scenarios(num_samples)
        
        training_data = []
        for scenario in scenarios:
            sample = {
                'text_tokens': self.tokenize_scenario(scenario),
                'personality_traits': self.generate_personality_vector(scenario),
                'context_features': self.generate_context_vector(scenario),
                'temporal_features': self.generate_temporal_vector(scenario),
                'attack_labels': self.encode_attack_type(scenario['attack_type']),
                'success_labels': scenario['success_probability'],
                'message_labels': self.generate_message_labels(scenario),
                'timing_labels': self.generate_timing_labels(scenario)
            }
            training_data.append(sample)
        
        return training_data
    
    def tokenize_scenario(self, scenario):
        text = f"{scenario['template']} {' '.join(scenario['trait_keywords'])} {' '.join(scenario['trigger_keywords'])}"
        tokens = [hash(word) % 50000 for word in text.split()]
        return torch.tensor(tokens[:512], dtype=torch.long)
    
    def generate_personality_vector(self, scenario):
        personality_map = {
            'agreeableness': [1, 0, 0, 0, 0],
            'conscientiousness': [0, 1, 0, 0, 0],
            'extraversion': [0, 0, 1, 0, 0],
            'neuroticism': [0, 0, 0, 1, 0],
            'openness': [0, 0, 0, 0, 1]
        }
        return torch.tensor(personality_map.get(scenario['target_personality'], [0.2, 0.2, 0.2, 0.2, 0.2]), dtype=torch.float32)
    
    def generate_context_vector(self, scenario):
        context = np.random.rand(20)
        context[0] = scenario['complexity_level'] / 5.0
        context[1] = scenario['urgency_level']
        return torch.tensor(context, dtype=torch.float32)
    
    def generate_temporal_vector(self, scenario):
        temporal = np.random.rand(10)
        temporal[0] = scenario['urgency_level']
        return torch.tensor(temporal, dtype=torch.float32)
    
    def encode_attack_type(self, attack_type):
        attack_map = {
            'phishing_email': 0, 'voice_phishing': 1, 'social_media_impersonation': 2,
            'pretexting': 3, 'baiting': 4, 'quid_pro_quo': 5, 'tailgating': 6,
            'shoulder_surfing': 7, 'dumpster_diving': 8, 'spear_phishing': 9
        }
        return torch.tensor(attack_map.get(attack_type, 0), dtype=torch.long)
    
    def generate_message_labels(self, scenario):
        return torch.randint(0, 50000, (1,), dtype=torch.long)
    
    def generate_timing_labels(self, scenario):
        timing_map = {
            'early_morning': 0, 'morning': 1, 'lunch_time': 2, 'afternoon': 3,
            'evening': 4, 'late_evening': 5, 'weekend': 6
        }
        return torch.tensor(random.randint(0, 6), dtype=torch.long)
