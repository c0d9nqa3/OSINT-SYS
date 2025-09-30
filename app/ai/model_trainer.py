import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional

from .psychological_vulnerability_model import PsychologicalVulnerabilityModel, VulnerabilityDataGenerator
from .social_engineering_model import SocialEngineeringAttackPlanner, SocialEngineeringDataGenerator
from .advanced_identity_verifier import AdvancedIdentityVerifier, IdentityVerificationDataGenerator
from .behavioral_pattern_analyzer import BehavioralPatternAnalyzer, BehavioralDataGenerator
from .training_datasets import ComprehensiveDatasetGenerator, DatasetLoader

class ModelTrainer:
    def __init__(self, models_dir="models", data_dir="data/training_datasets"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def train_vulnerability_model(self, epochs=100, batch_size=16, learning_rate=0.001):
        print("Training Psychological Vulnerability Model...")
        
        model = PsychologicalVulnerabilityModel()
        data_generator = VulnerabilityDataGenerator()
        
        training_data = data_generator.create_training_dataset(8000)
        validation_data = data_generator.create_training_dataset(2000)
        
        model.train_model(
            training_data=training_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        model_path = self.models_dir / "psychological_vulnerability_model.pth"
        model.save_model(model_path)
        
        print(f"Vulnerability model saved to {model_path}")
        return model_path
    
    def train_social_engineering_model(self, epochs=100, batch_size=16, learning_rate=0.001):
        print("Training Social Engineering Attack Model...")
        
        model = SocialEngineeringAttackPlanner()
        data_generator = SocialEngineeringDataGenerator()
        
        training_data = data_generator.generate_training_data(4000)
        validation_data = data_generator.generate_training_data(1000)
        
        model.train_model(
            training_data=training_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        model_path = self.models_dir / "social_engineering_model.pth"
        model.save_model(model_path)
        
        print(f"Social engineering model saved to {model_path}")
        return model_path
    
    def train_identity_verification_model(self, epochs=100, batch_size=16, learning_rate=0.001):
        print("Training Advanced Identity Verification Model...")
        
        model = AdvancedIdentityVerifier()
        data_generator = IdentityVerificationDataGenerator()
        
        dataset = data_generator.create_training_dataset(1600)
        
        pairs = dataset['identity_pairs']
        labels = dataset['labels']
        
        training_pairs = pairs[:int(len(pairs) * 0.8)]
        training_labels = labels[:int(len(labels) * 0.8)]
        
        validation_pairs = pairs[int(len(pairs) * 0.8):]
        validation_labels = labels[int(len(labels) * 0.8):]
        
        training_data = self.prepare_identity_training_data(training_pairs, training_labels)
        validation_data = self.prepare_identity_training_data(validation_pairs, validation_labels)
        
        model.train_model(
            training_data=training_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        model_path = self.models_dir / "identity_verification_model.pth"
        model.save_model(model_path)
        
        print(f"Identity verification model saved to {model_path}")
        return model_path
    
    def train_behavioral_pattern_model(self, epochs=100, batch_size=16, learning_rate=0.001):
        print("Training Behavioral Pattern Analysis Model...")
        
        model = BehavioralPatternAnalyzer()
        data_generator = BehavioralDataGenerator()
        
        dataset = data_generator.create_training_dataset(800)
        
        training_data = dataset[:int(len(dataset) * 0.8)]
        validation_data = dataset[int(len(dataset) * 0.8):]
        
        training_data = self.prepare_behavioral_training_data(training_data)
        validation_data = self.prepare_behavioral_training_data(validation_data)
        
        model.train_model(
            training_data=training_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        model_path = self.models_dir / "behavioral_pattern_model.pth"
        model.save_model(model_path)
        
        print(f"Behavioral pattern model saved to {model_path}")
        return model_path
    
    def prepare_identity_training_data(self, pairs, labels):
        training_data = []
        
        for (identity1, identity2), label in zip(pairs, labels):
            text1 = " ".join(identity1.get('text_data', ['']))
            text2 = " ".join(identity2.get('text_data', ['']))
            
            text_input = self.prepare_text_input(text1 + " " + text2)
            
            behavioral1 = self.extract_behavioral_features(identity1.get('behavioral_data', [{}]))
            behavioral2 = self.extract_behavioral_features(identity2.get('behavioral_data', [{}]))
            
            temporal1 = self.extract_temporal_features(identity1.get('temporal_data', [{}]))
            temporal2 = self.extract_temporal_features(identity2.get('temporal_data', [{}]))
            
            sample = {
                'text_input_ids': text_input['input_ids'],
                'text_attention_mask': text_input['attention_mask'],
                'image_pixel_values': torch.randn(1, 3, 224, 224),
                'audio_input': torch.randn(1, 1, 16000),
                'face_input': torch.randn(1, 3, 224, 224),
                'behavioral_input': torch.cat([behavioral1, behavioral2], dim=0),
                'temporal_input': torch.cat([temporal1, temporal2], dim=0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
            
            training_data.append(sample)
        
        return training_data
    
    def prepare_behavioral_training_data(self, dataset):
        training_data = []
        
        for data in dataset:
            person_data = data['person_profile']
            timeline = data['timeline']
            
            sequences = self.extract_behavioral_sequences(timeline)
            pattern_labels = self.generate_pattern_labels(person_data)
            anomaly_labels = self.generate_anomaly_labels(timeline)
            future_labels = self.generate_future_labels(timeline)
            
            sample = {
                'sequences': sequences,
                'pattern_labels': pattern_labels,
                'anomaly_labels': anomaly_labels,
                'future_labels': future_labels
            }
            
            training_data.append(sample)
        
        return training_data
    
    def extract_behavioral_sequences(self, timeline):
        sequence_length = min(len(timeline), 100)
        sequence = np.zeros((sequence_length, 200))
        
        for i, event in enumerate(timeline[:sequence_length]):
            sequence[i] = self.encode_behavioral_event(event)
        
        return torch.tensor(sequence, dtype=torch.float32)
    
    def encode_behavioral_event(self, event):
        feature_vector = np.zeros(200)
        
        feature_vector[0] = event.get('hour', 0) / 24
        feature_vector[1] = event.get('day_of_week', 0) / 7
        feature_vector[2] = event.get('activity_type', 0) / 10
        feature_vector[3] = event.get('duration', 0) / 3600
        feature_vector[4] = event.get('intensity', 0)
        
        return feature_vector
    
    def generate_pattern_labels(self, person_profile):
        personality_type = person_profile.get('personality_type', 'balanced')
        
        pattern_map = {
            'early_bird': 0,
            'night_owl': 1,
            'workaholic': 2,
            'balanced': 3,
            'social': 4,
            'introvert': 5
        }
        
        return torch.tensor(pattern_map.get(personality_type, 3), dtype=torch.long)
    
    def generate_anomaly_labels(self, timeline):
        anomaly_score = 0.0
        
        for event in timeline:
            if event.get('hour', 12) < 3 or event.get('hour', 12) > 23:
                anomaly_score += 0.1
            if event.get('intensity', 0.5) > 0.9:
                anomaly_score += 0.1
        
        return torch.tensor(1.0 if anomaly_score > 0.3 else 0.0, dtype=torch.float32)
    
    def generate_future_labels(self, timeline):
        if len(timeline) < 10:
            return torch.zeros(200, dtype=torch.float32)
        
        recent_events = timeline[-10:]
        avg_features = np.mean([self.encode_behavioral_event(event) for event in recent_events], axis=0)
        
        return torch.tensor(avg_features, dtype=torch.float32)
    
    def prepare_text_input(self, text):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return encoding
    
    def extract_behavioral_features(self, behavioral_data):
        if not behavioral_data:
            return torch.zeros(100, dtype=torch.float32)
        
        features = np.zeros(100)
        data = behavioral_data[0]
        
        if 'typing_patterns' in data:
            typing = data['typing_patterns']
            features[0] = typing.get('speed', 50) / 100
            features[1] = typing.get('accuracy', 0.9)
        
        if 'mouse_patterns' in data:
            mouse = data['mouse_patterns']
            features[2] = mouse.get('movement_speed', 1.0) / 10
            features[3] = mouse.get('click_frequency', 2.0) / 10
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_temporal_features(self, temporal_data):
        if not temporal_data:
            return torch.zeros(50, 50, dtype=torch.float32)
        
        timeline = temporal_data[0].get('activity_timeline', [])
        sequence_length = min(len(timeline), 50)
        sequence = np.zeros((sequence_length, 50))
        
        for i, activity in enumerate(timeline[:sequence_length]):
            sequence[i, 0] = activity.get('hour', 0) / 24
            sequence[i, 1] = activity.get('day_of_week', 0) / 7
            sequence[i, 2] = activity.get('activity_type', 0) / 10
            sequence[i, 3] = activity.get('duration', 0) / 3600
            sequence[i, 4] = activity.get('intensity', 0)
        
        return torch.tensor(sequence, dtype=torch.float32)
    
    def train_all_models(self, epochs=100, batch_size=16, learning_rate=0.001):
        print("Starting training of all AI models...")
        
        trained_models = {}
        
        try:
            model_path = self.train_vulnerability_model(epochs, batch_size, learning_rate)
            trained_models['vulnerability'] = model_path
        except Exception as e:
            print(f"Error training vulnerability model: {e}")
        
        try:
            model_path = self.train_social_engineering_model(epochs, batch_size, learning_rate)
            trained_models['social_engineering'] = model_path
        except Exception as e:
            print(f"Error training social engineering model: {e}")
        
        try:
            model_path = self.train_identity_verification_model(epochs, batch_size, learning_rate)
            trained_models['identity_verification'] = model_path
        except Exception as e:
            print(f"Error training identity verification model: {e}")
        
        try:
            model_path = self.train_behavioral_pattern_model(epochs, batch_size, learning_rate)
            trained_models['behavioral_pattern'] = model_path
        except Exception as e:
            print(f"Error training behavioral pattern model: {e}")
        
        self.save_training_summary(trained_models)
        
        print("All model training completed!")
        return trained_models
    
    def save_training_summary(self, trained_models):
        summary = {
            'training_date': datetime.now().isoformat(),
            'device_used': str(self.device),
            'trained_models': trained_models,
            'model_count': len(trained_models)
        }
        
        summary_path = self.models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")

class ModelEvaluator:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_vulnerability_model(self, model_path, test_data):
        print("Evaluating Psychological Vulnerability Model...")
        
        model = PsychologicalVulnerabilityModel(model_path)
        
        correct_predictions = 0
        total_predictions = 0
        
        for person_data in test_data:
            try:
                analysis = model.analyze_vulnerability(person_data)
                
                if analysis['overall_risk_score'] > 0.5:
                    predicted_high_risk = True
                else:
                    predicted_high_risk = False
                
                actual_high_risk = person_data.get('actual_high_risk', False)
                
                if predicted_high_risk == actual_high_risk:
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Vulnerability Model Accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_social_engineering_model(self, model_path, test_data):
        print("Evaluating Social Engineering Model...")
        
        model = SocialEngineeringAttackPlanner(model_path)
        
        correct_predictions = 0
        total_predictions = 0
        
        for target_data in test_data:
            try:
                attack_plan = model.generate_attack_plan(target_data, "test_objective")
                
                predicted_success = attack_plan['overall_success_probability'] > 0.5
                actual_success = target_data.get('actual_success', False)
                
                if predicted_success == actual_success:
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Social Engineering Model Accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_identity_verification_model(self, model_path, test_data):
        print("Evaluating Identity Verification Model...")
        
        model = AdvancedIdentityVerifier(model_path)
        
        correct_predictions = 0
        total_predictions = 0
        
        for identity_pair in test_data:
            try:
                identity1, identity2, actual_match = identity_pair
                
                result = model.verify_identity(identity1, identity2)
                predicted_match = result['is_same_person']
                
                if predicted_match == actual_match:
                    correct_predictions += 1
                
                total_predictions += 1
                
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Identity Verification Model Accuracy: {accuracy:.4f}")
        return accuracy
    
    def evaluate_behavioral_pattern_model(self, model_path, test_data):
        print("Evaluating Behavioral Pattern Model...")
        
        model = BehavioralPatternAnalyzer(model_path)
        
        correct_predictions = 0
        total_predictions = 0
        
        for behavioral_data in test_data:
            try:
                analysis = model.analyze_behavioral_patterns(behavioral_data)
                
                if 'behavior_patterns' in analysis:
                    predicted_patterns = analysis['behavior_patterns']
                    actual_patterns = behavioral_data.get('actual_patterns', [])
                    
                    if len(predicted_patterns) > 0 and len(actual_patterns) > 0:
                        predicted_dominant = predicted_patterns[0]['dominant_patterns'][0]['pattern'] if predicted_patterns[0]['dominant_patterns'] else 'unknown'
                        actual_dominant = actual_patterns[0] if actual_patterns else 'unknown'
                        
                        if predicted_dominant == actual_dominant:
                            correct_predictions += 1
                        
                        total_predictions += 1
                
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        print(f"Behavioral Pattern Model Accuracy: {accuracy:.4f}")
        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train AI models for OSINT system')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, choices=['all', 'vulnerability', 'social_engineering', 'identity_verification', 'behavioral_pattern'], 
                       default='all', help='Which model to train')
    parser.add_argument('--generate-data', action='store_true', help='Generate training data first')
    
    args = parser.parse_args()
    
    if args.generate_data:
        print("Generating training datasets...")
        generator = ComprehensiveDatasetGenerator()
        generator.generate_all_datasets()
    
    trainer = ModelTrainer()
    
    if args.model == 'all':
        trainer.train_all_models(args.epochs, args.batch_size, args.learning_rate)
    elif args.model == 'vulnerability':
        trainer.train_vulnerability_model(args.epochs, args.batch_size, args.learning_rate)
    elif args.model == 'social_engineering':
        trainer.train_social_engineering_model(args.epochs, args.batch_size, args.learning_rate)
    elif args.model == 'identity_verification':
        trainer.train_identity_verification_model(args.epochs, args.batch_size, args.learning_rate)
    elif args.model == 'behavioral_pattern':
        trainer.train_behavioral_pattern_model(args.epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
