import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

from .psychological_vulnerability_model import VulnerabilityDataGenerator
from .social_engineering_model import SocialEngineeringDataGenerator
from .advanced_identity_verifier import IdentityVerificationDataGenerator
from .behavioral_pattern_analyzer import BehavioralDataGenerator

class ComprehensiveDatasetGenerator:
    def __init__(self):
        self.vulnerability_generator = VulnerabilityDataGenerator()
        self.social_engineering_generator = SocialEngineeringDataGenerator()
        self.identity_generator = IdentityVerificationDataGenerator()
        self.behavioral_generator = BehavioralDataGenerator()
        
        self.output_dir = Path("data/training_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_vulnerability_dataset(self, num_samples=10000):
        print("Generating psychological vulnerability dataset...")
        
        dataset = self.vulnerability_generator.create_training_dataset(num_samples)
        
        training_data = {
            'texts': dataset['texts'][:int(num_samples * 0.8)],
            'labels': dataset['labels'][:int(num_samples * 0.8)]
        }
        
        validation_data = {
            'texts': dataset['texts'][int(num_samples * 0.8):],
            'labels': dataset['labels'][int(num_samples * 0.8):]
        }
        
        self.save_dataset(training_data, 'vulnerability_training.json')
        self.save_dataset(validation_data, 'vulnerability_validation.json')
        
        print(f"Generated vulnerability dataset: {len(training_data['texts'])} training samples, {len(validation_data['texts'])} validation samples")
        
        return training_data, validation_data
    
    def generate_social_engineering_dataset(self, num_scenarios=5000):
        print("Generating social engineering dataset...")
        
        scenarios = self.social_engineering_generator.dataset.generate_attack_scenarios(num_scenarios)
        
        training_scenarios = scenarios[:int(num_scenarios * 0.8)]
        validation_scenarios = scenarios[int(num_scenarios * 0.8):]
        
        training_data = self.social_engineering_generator.generate_training_data(len(training_scenarios))
        validation_data = self.social_engineering_generator.generate_training_data(len(validation_scenarios))
        
        self.save_dataset(training_data, 'social_engineering_training.json')
        self.save_dataset(validation_data, 'social_engineering_validation.json')
        
        print(f"Generated social engineering dataset: {len(training_data)} training samples, {len(validation_data)} validation samples")
        
        return training_data, validation_data
    
    def generate_identity_verification_dataset(self, num_identities=2000):
        print("Generating identity verification dataset...")
        
        dataset = self.identity_generator.create_training_dataset(num_identities)
        
        training_pairs = dataset['identity_pairs'][:int(len(dataset['identity_pairs']) * 0.8)]
        training_labels = dataset['labels'][:int(len(dataset['labels']) * 0.8)]
        
        validation_pairs = dataset['identity_pairs'][int(len(dataset['identity_pairs']) * 0.8):]
        validation_labels = dataset['labels'][int(len(dataset['labels']) * 0.8):]
        
        training_data = {
            'identity_pairs': training_pairs,
            'labels': training_labels
        }
        
        validation_data = {
            'identity_pairs': validation_pairs,
            'labels': validation_labels
        }
        
        self.save_dataset(training_data, 'identity_verification_training.json')
        self.save_dataset(validation_data, 'identity_verification_validation.json')
        
        print(f"Generated identity verification dataset: {len(training_pairs)} training pairs, {len(validation_pairs)} validation pairs")
        
        return training_data, validation_data
    
    def generate_behavioral_pattern_dataset(self, num_persons=1000):
        print("Generating behavioral pattern dataset...")
        
        dataset = self.behavioral_generator.create_training_dataset(num_persons)
        
        training_data = dataset[:int(len(dataset) * 0.8)]
        validation_data = dataset[int(len(dataset) * 0.8):]
        
        self.save_dataset(training_data, 'behavioral_patterns_training.json')
        self.save_dataset(validation_data, 'behavioral_patterns_validation.json')
        
        print(f"Generated behavioral pattern dataset: {len(training_data)} training samples, {len(validation_data)} validation samples")
        
        return training_data, validation_data
    
    def generate_osint_specific_dataset(self, num_targets=5000):
        print("Generating OSINT-specific dataset...")
        
        osint_dataset = []
        
        for i in range(num_targets):
            target_profile = self.generate_osint_target_profile(i)
            osint_dataset.append(target_profile)
        
        training_data = osint_dataset[:int(num_targets * 0.8)]
        validation_data = osint_dataset[int(num_targets * 0.8):]
        
        self.save_dataset(training_data, 'osint_targets_training.json')
        self.save_dataset(validation_data, 'osint_targets_validation.json')
        
        print(f"Generated OSINT-specific dataset: {len(training_data)} training targets, {len(validation_data)} validation targets")
        
        return training_data, validation_data
    
    def generate_osint_target_profile(self, target_id):
        profile = {
            'target_id': f"target_{target_id}",
            'basic_info': self.generate_basic_info(),
            'digital_footprint': self.generate_digital_footprint(),
            'social_media_profiles': self.generate_social_media_profiles(),
            'professional_info': self.generate_professional_info(),
            'personal_preferences': self.generate_personal_preferences(),
            'security_practices': self.generate_security_practices(),
            'vulnerability_indicators': self.generate_vulnerability_indicators(),
            'behavioral_patterns': self.generate_behavioral_indicators(),
            'relationship_network': self.generate_relationship_network(),
            'temporal_patterns': self.generate_temporal_patterns(),
            'geographic_data': self.generate_geographic_data(),
            'financial_indicators': self.generate_financial_indicators(),
            'communication_style': self.generate_communication_style(),
            'psychological_profile': self.generate_psychological_profile()
        }
        
        return profile
    
    def generate_basic_info(self):
        first_names = ['John', 'Sarah', 'Michael', 'Emily', 'David', 'Jessica', 'Robert', 'Amanda', 'James', 'Jennifer']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        return {
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'age': random.randint(18, 65),
            'gender': random.choice(['male', 'female', 'other']),
            'location': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']),
            'education_level': random.choice(['high_school', 'bachelor', 'master', 'phd', 'other'])
        }
    
    def generate_digital_footprint(self):
        return {
            'email_addresses': [f"user{random.randint(1000, 9999)}@{random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"],
            'phone_numbers': [f"+1-{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}"],
            'usernames': [f"user{random.randint(1000, 9999)}", f"person{random.randint(100, 999)}"],
            'website_mentions': random.randint(0, 50),
            'news_articles': random.randint(0, 20),
            'professional_listings': random.randint(0, 10),
            'social_media_activity_score': random.uniform(0.1, 1.0),
            'online_presence_strength': random.uniform(0.2, 0.9)
        }
    
    def generate_social_media_profiles(self):
        platforms = ['facebook', 'twitter', 'linkedin', 'instagram', 'youtube', 'tiktok', 'snapchat']
        profiles = {}
        
        for platform in platforms:
            if random.random() < 0.7:
                profiles[platform] = {
                    'username': f"user{random.randint(1000, 9999)}",
                    'followers': random.randint(10, 10000),
                    'following': random.randint(10, 5000),
                    'posts_count': random.randint(0, 1000),
                    'activity_level': random.uniform(0.1, 1.0),
                    'privacy_settings': random.choice(['public', 'private', 'friends_only']),
                    'last_active': self.generate_random_date()
                }
        
        return profiles
    
    def generate_professional_info(self):
        companies = ['Microsoft', 'Google', 'Apple', 'Amazon', 'Facebook', 'Netflix', 'Tesla', 'Uber', 'Airbnb', 'Spotify']
        positions = ['Software Engineer', 'Data Scientist', 'Product Manager', 'Marketing Manager', 'Sales Representative', 'HR Manager', 'Financial Analyst', 'Designer']
        
        return {
            'current_company': random.choice(companies),
            'position': random.choice(positions),
            'industry': random.choice(['technology', 'finance', 'healthcare', 'education', 'retail', 'manufacturing']),
            'experience_years': random.randint(1, 20),
            'skills': random.sample(['Python', 'JavaScript', 'Java', 'SQL', 'Machine Learning', 'Project Management', 'Marketing', 'Sales'], random.randint(3, 8)),
            'certifications': random.sample(['AWS', 'Google Cloud', 'Microsoft Azure', 'PMP', 'CPA'], random.randint(0, 3)),
            'education': {
                'degree': random.choice(['Bachelor', 'Master', 'PhD']),
                'field': random.choice(['Computer Science', 'Business', 'Engineering', 'Marketing', 'Finance']),
                'university': random.choice(['Stanford', 'MIT', 'Harvard', 'UC Berkeley', 'Carnegie Mellon'])
            }
        }
    
    def generate_personal_preferences(self):
        return {
            'interests': random.sample(['technology', 'sports', 'music', 'travel', 'photography', 'cooking', 'reading', 'gaming'], random.randint(3, 6)),
            'hobbies': random.sample(['hiking', 'swimming', 'running', 'painting', 'writing', 'gardening', 'chess', 'video_games'], random.randint(2, 5)),
            'lifestyle': random.choice(['urban', 'suburban', 'rural']),
            'political_leaning': random.choice(['liberal', 'conservative', 'moderate', 'apolitical']),
            'religious_affiliation': random.choice(['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'atheist', 'agnostic', 'other'])
        }
    
    def generate_security_practices(self):
        return {
            'password_strength': random.uniform(0.2, 1.0),
            'two_factor_auth': random.choice([True, False]),
            'privacy_settings': random.uniform(0.3, 1.0),
            'data_sharing_tendency': random.uniform(0.1, 0.9),
            'security_awareness': random.uniform(0.2, 1.0),
            'software_updates': random.uniform(0.3, 1.0),
            'phishing_susceptibility': random.uniform(0.1, 0.8)
        }
    
    def generate_vulnerability_indicators(self):
        return {
            'social_engineering_risk': random.uniform(0.1, 0.9),
            'authority_compliance': random.uniform(0.3, 1.0),
            'urgency_susceptibility': random.uniform(0.2, 0.8),
            'curiosity_level': random.uniform(0.3, 1.0),
            'fear_response': random.uniform(0.2, 0.9),
            'greed_susceptibility': random.uniform(0.1, 0.7),
            'trust_level': random.uniform(0.2, 0.9),
            'information_sharing': random.uniform(0.1, 0.8)
        }
    
    def generate_behavioral_indicators(self):
        return {
            'online_activity_hours': random.randint(2, 16),
            'social_media_usage': random.uniform(0.1, 1.0),
            'email_checking_frequency': random.randint(1, 20),
            'response_time': random.uniform(0.1, 24.0),
            'communication_style': random.choice(['formal', 'casual', 'mixed']),
            'decision_making_speed': random.uniform(0.1, 1.0),
            'risk_taking_tendency': random.uniform(0.1, 0.9),
            'routine_adherence': random.uniform(0.2, 1.0)
        }
    
    def generate_relationship_network(self):
        return {
            'family_size': random.randint(1, 8),
            'close_friends': random.randint(2, 15),
            'professional_contacts': random.randint(10, 500),
            'social_network_density': random.uniform(0.2, 0.9),
            'influence_score': random.uniform(0.1, 1.0),
            'network_diversity': random.uniform(0.3, 1.0)
        }
    
    def generate_temporal_patterns(self):
        return {
            'morning_person': random.choice([True, False]),
            'night_owl': random.choice([True, False]),
            'weekend_activity': random.uniform(0.1, 1.0),
            'travel_frequency': random.randint(0, 12),
            'schedule_consistency': random.uniform(0.2, 1.0),
            'peak_productivity_hours': random.sample(range(24), random.randint(2, 6))
        }
    
    def generate_geographic_data(self):
        return {
            'home_location': random.choice(['urban', 'suburban', 'rural']),
            'work_location': random.choice(['urban', 'suburban', 'rural']),
            'commute_time': random.randint(5, 120),
            'travel_history': random.randint(0, 20),
            'location_sharing': random.choice([True, False]),
            'geographic_stability': random.uniform(0.3, 1.0)
        }
    
    def generate_financial_indicators(self):
        return {
            'income_level': random.choice(['low', 'medium', 'high']),
            'spending_patterns': random.choice(['conservative', 'moderate', 'liberal']),
            'investment_activity': random.uniform(0.1, 1.0),
            'debt_level': random.uniform(0.0, 0.8),
            'financial_transparency': random.uniform(0.2, 1.0)
        }
    
    def generate_communication_style(self):
        return {
            'formality_level': random.uniform(0.2, 1.0),
            'response_speed': random.uniform(0.1, 1.0),
            'message_length': random.uniform(0.2, 1.0),
            'emoji_usage': random.uniform(0.0, 0.8),
            'question_frequency': random.uniform(0.1, 0.9),
            'conflict_avoidance': random.uniform(0.2, 1.0)
        }
    
    def generate_psychological_profile(self):
        return {
            'openness': random.uniform(0.2, 1.0),
            'conscientiousness': random.uniform(0.2, 1.0),
            'extraversion': random.uniform(0.2, 1.0),
            'agreeableness': random.uniform(0.2, 1.0),
            'neuroticism': random.uniform(0.2, 1.0),
            'stress_tolerance': random.uniform(0.1, 1.0),
            'impulsiveness': random.uniform(0.1, 0.9),
            'empathy_level': random.uniform(0.2, 1.0)
        }
    
    def generate_random_date(self):
        start_date = datetime.now() - timedelta(days=365)
        random_days = random.randint(0, 365)
        return (start_date + timedelta(days=random_days)).isoformat()
    
    def save_dataset(self, data, filename):
        filepath = self.output_dir / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        print(f"Saved dataset to {filepath}")
    
    def generate_all_datasets(self):
        print("Starting comprehensive dataset generation...")
        
        datasets = {}
        
        try:
            vuln_train, vuln_val = self.generate_vulnerability_dataset(10000)
            datasets['vulnerability'] = {'training': vuln_train, 'validation': vuln_val}
        except Exception as e:
            print(f"Error generating vulnerability dataset: {e}")
        
        try:
            se_train, se_val = self.generate_social_engineering_dataset(5000)
            datasets['social_engineering'] = {'training': se_train, 'validation': se_val}
        except Exception as e:
            print(f"Error generating social engineering dataset: {e}")
        
        try:
            id_train, id_val = self.generate_identity_verification_dataset(2000)
            datasets['identity_verification'] = {'training': id_train, 'validation': id_val}
        except Exception as e:
            print(f"Error generating identity verification dataset: {e}")
        
        try:
            beh_train, beh_val = self.generate_behavioral_pattern_dataset(1000)
            datasets['behavioral_patterns'] = {'training': beh_train, 'validation': beh_val}
        except Exception as e:
            print(f"Error generating behavioral pattern dataset: {e}")
        
        try:
            osint_train, osint_val = self.generate_osint_specific_dataset(5000)
            datasets['osint_targets'] = {'training': osint_train, 'validation': osint_val}
        except Exception as e:
            print(f"Error generating OSINT dataset: {e}")
        
        self.generate_dataset_summary(datasets)
        
        print("Dataset generation completed!")
        return datasets
    
    def generate_dataset_summary(self, datasets):
        summary = {
            'generation_date': datetime.now().isoformat(),
            'datasets': {},
            'statistics': {}
        }
        
        for dataset_name, dataset_data in datasets.items():
            if 'training' in dataset_data and 'validation' in dataset_data:
                train_size = len(dataset_data['training'])
                val_size = len(dataset_data['validation'])
                
                summary['datasets'][dataset_name] = {
                    'training_samples': train_size,
                    'validation_samples': val_size,
                    'total_samples': train_size + val_size
                }
        
        summary['statistics'] = {
            'total_datasets': len(datasets),
            'total_training_samples': sum(d['training_samples'] for d in summary['datasets'].values()),
            'total_validation_samples': sum(d['validation_samples'] for d in summary['datasets'].values()),
            'total_samples': sum(d['total_samples'] for d in summary['datasets'].values())
        }
        
        self.save_dataset(summary, 'dataset_summary.json')
        
        print(f"\nDataset Summary:")
        print(f"Total datasets: {summary['statistics']['total_datasets']}")
        print(f"Total training samples: {summary['statistics']['total_training_samples']}")
        print(f"Total validation samples: {summary['statistics']['total_validation_samples']}")
        print(f"Total samples: {summary['statistics']['total_samples']}")

class DatasetLoader:
    def __init__(self, data_dir="data/training_datasets"):
        self.data_dir = Path(data_dir)
    
    def load_vulnerability_dataset(self):
        train_path = self.data_dir / 'vulnerability_training.json'
        val_path = self.data_dir / 'vulnerability_validation.json'
        
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        return training_data, validation_data
    
    def load_social_engineering_dataset(self):
        train_path = self.data_dir / 'social_engineering_training.json'
        val_path = self.data_dir / 'social_engineering_validation.json'
        
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        return training_data, validation_data
    
    def load_identity_verification_dataset(self):
        train_path = self.data_dir / 'identity_verification_training.json'
        val_path = self.data_dir / 'identity_verification_validation.json'
        
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        return training_data, validation_data
    
    def load_behavioral_pattern_dataset(self):
        train_path = self.data_dir / 'behavioral_patterns_training.json'
        val_path = self.data_dir / 'behavioral_patterns_validation.json'
        
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        return training_data, validation_data
    
    def load_osint_targets_dataset(self):
        train_path = self.data_dir / 'osint_targets_training.json'
        val_path = self.data_dir / 'osint_targets_validation.json'
        
        with open(train_path, 'r') as f:
            training_data = json.load(f)
        
        with open(val_path, 'r') as f:
            validation_data = json.load(f)
        
        return training_data, validation_data

if __name__ == "__main__":
    generator = ComprehensiveDatasetGenerator()
    datasets = generator.generate_all_datasets()
