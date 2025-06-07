import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    ViTFeatureExtractor,
    ViTForImageClassification
)
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import imagehash
import warnings
from tqdm.auto import tqdm
import re
import hashlib
from collections import Counter, defaultdict
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class EnhancedAIFraudDetector:
    def __init__(self):
        print("Initializing models...")
        
        # Text analysis models
        try:
            self.fake_review_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1,
                batch_size=16
            )
        except:
            print("Using basic sentiment analysis...")
            self.fake_review_classifier = None
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Image analysis models
        try:
            self.image_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            self.image_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        except:
            print("Image models not available, using hash-based analysis...")
            self.image_feature_extractor = None
            self.image_model = None
        
        # ML models for fraud detection
        self.user_behavior_model = IsolationForest(contamination=0.1, random_state=42)
        self.duplicate_classifier = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Global storage
        self.global_image_registry = {}
        self.product_features = {}
        self.image_hashes = {}
        self.review_image_hashes = {}
        
        print("All models initialized successfully!")

    def extract_advanced_product_features(self, df):
        product_features = {}
        
        for product_id in df['product_id'].unique():
            product_data = df[df['product_id'] == product_id]
            
            # Temporal features
            review_count = len(product_data)
            unique_reviewers = product_data['user_id'].nunique()
            
            # Review quality features
            avg_review_length = product_data['review_content'].fillna('').astype(str).apply(len).mean()
            review_diversity = product_data['review_content'].fillna('').nunique() / max(len(product_data), 1)
            
            # Rating features
            rating_mean = product_data['rating'].mean()
            rating_std = product_data['rating'].std() if len(product_data) > 1 else 0
            
            # Username diversity
            username_patterns = product_data['user_name'].fillna('').apply(self._analyze_username_pattern)
            username_suspicion_avg = np.mean(username_patterns)
            
            # Category consistency
            category_consistency = 1.0 if len(product_data['category'].unique()) == 1 else 0.0
            
            product_features[product_id] = {
                'review_count': review_count,
                'unique_reviewers': unique_reviewers,
                'reviewer_ratio': unique_reviewers / max(review_count, 1),
                'avg_review_length': avg_review_length,
                'review_diversity': review_diversity,
                'rating_mean': rating_mean,
                'rating_std': rating_std,
                'username_suspicion_avg': username_suspicion_avg,
                'category_consistency': category_consistency,
            }
        
        return product_features

    def _analyze_username_pattern(self, username):
        ## analyzing usernames
        if pd.isna(username) or not isinstance(username, str):
            return 1.0
            
        ## basic pattern analysis
        digit_ratio = sum(c.isdigit() for c in username) / max(len(username), 1)
        length_score = 1.0 if len(username) < 4 else 0.0
        random_score = 1.0 if digit_ratio > 0.7 else 0.0
        
        return (digit_ratio * 0.4 + length_score * 0.3 + random_score * 0.3)

    def train_duplicate_authenticity_model(self, product_features):
        
        # creating training data from product features
        feature_matrix = []
        labels = []
        
        for product_id, features in product_features.items():
            feature_vector = [
                features['review_count'],
                features['reviewer_ratio'],
                features['avg_review_length'],
                features['review_diversity'],
                features['rating_mean'],
                features['rating_std'],
                features['username_suspicion_avg'],
                features['category_consistency']
            ]
            
            authenticity_score = (
                features['reviewer_ratio'] * 0.5 +
                features['review_diversity'] * 0.2 +
                (1 - features['username_suspicion_avg']) * 0.3
            )
            
            feature_matrix.append(feature_vector)
            labels.append(authenticity_score)
        
        if len(feature_matrix) > 0:
            self.duplicate_classifier.fit(feature_matrix, labels)
            print("Duplicate authenticity model trained successfully!")

    def analyze_cross_product_image_duplicates_ml(self, df):
        print("Analyzing cross-product image duplicates...")
        
        # Extracting product features
        self.product_features = self.extract_advanced_product_features(df)
        
        # Training the authenticity model
        self.train_duplicate_authenticity_model(self.product_features)
        
        # Getting unique products with their images
        unique_products = df.groupby('product_id')['img_link'].first().reset_index()
        
        # Building global image registry
        for _, row in tqdm(unique_products.iterrows(), desc="Building image registry"):
            product_id = row['product_id']
            img_url = row['img_link']
            
            if pd.isna(img_url) or not isinstance(img_url, str):
                continue
                
            try:
                response = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img_hash = str(imagehash.phash(img))
                
                if img_hash not in self.global_image_registry:
                    self.global_image_registry[img_hash] = []
                self.global_image_registry[img_hash].append(product_id)
                
            except Exception as e:
                logging.warning(f"Error processing image for product {product_id}: {str(e)[:50]}")
        
        # Calculating ML-based duplicate scores
        product_duplicate_scores = {}
        
        for img_hash, product_ids in self.global_image_registry.items():
            if len(product_ids) > 1:  ### Duplicate found
                authenticity_predictions = []
                
                for product_id in product_ids:
                    if product_id in self.product_features:
                        features = self.product_features[product_id]
                        feature_vector = [[
                            features['review_count'],
                            features['reviewer_ratio'],
                            features['avg_review_length'],
                            features['review_diversity'],
                            features['rating_mean'],
                            features['rating_std'],
                            features['username_suspicion_avg'],
                            features['category_consistency']
                        ]]
                        
                        try:
                            authenticity_score = self.duplicate_classifier.predict(feature_vector)[0]
                            authenticity_predictions.append((product_id, authenticity_score))
                        except:
                            authenticity_predictions.append((product_id, 0.5))
                    else:
                        authenticity_predictions.append((product_id, 0.5))
                
                # sort by desc ML-predicted authenticity
                authenticity_predictions.sort(key=lambda x: x[1], reverse=True)
                
                # assigning fraud scores based on ML predictions
                max_auth = authenticity_predictions[0][1] if authenticity_predictions else 0.5
                min_auth = authenticity_predictions[-1][1] if authenticity_predictions else 0.5
                auth_range = max(max_auth - min_auth, 0.1)
                
                for product_id, auth_score in authenticity_predictions:
                    # Normalize authenticity to fraud score (inverse relationship)
                    relative_auth = (auth_score - min_auth) / auth_range
                    fraud_score = 1.0 - relative_auth  # Higher authenticity = lower fraud score
                    
                    # Scale to reasonable range
                    fraud_score = 0.2 + (fraud_score * 0.6)  # Scale to 0.2-0.8 range
                    product_duplicate_scores[product_id] = min(fraud_score, 1.0)
            else:
                # NO duplicates found
                product_duplicate_scores[product_ids[0]] = 0.0
        
        return product_duplicate_scores

    def analyze_review_images(self, df):
        review_image_scores = []
        
        for idx, row in tqdm(df.iterrows(), desc="Processing review images"):
            review_content = str(row['review_content']) if not pd.isna(row['review_content']) else ''
            image_urls = self.extract_image_urls_from_text(review_content)
            
            if not image_urls:
                review_image_scores.append(0.0)
                continue
            
            # Analyze each image URL found
            total_suspicion = 0.0
            valid_images = 0
            
            for img_url in image_urls:
                try:
                    response = requests.get(img_url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    img_hash = str(imagehash.phash(img))
                    
                    if img_hash in self.review_image_hashes:
                        total_suspicion += 0.8
                    else:
                        self.review_image_hashes[img_hash] = row['review_id']
                        total_suspicion += 0.1
                    
                    valid_images += 1
                    
                except Exception:
                    total_suspicion += 0.3
            
            avg_suspicion = total_suspicion / max(valid_images, 1) if valid_images > 0 else 0.5
            review_image_scores.append(min(avg_suspicion, 1.0))
        
        return review_image_scores

    def extract_image_urls_from_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        
        image_patterns = [
            r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|bmp|webp)',
            r'https?://[^\s]*(?:image|img|photo)[^\s]*',
            r'https?://m\.media-amazon\.com/[^\s]+',
        ]
        
        urls = []
        for pattern in image_patterns:
            urls.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return list(set(urls))

    def analyze_username_patterns_ml(self, username):
        if pd.isna(username) or not isinstance(username, str):
            return 1.0
            
        try:
            # Extract features
            length = len(username)
            digit_ratio = sum(c.isdigit() for c in username) / length
            special_char_ratio = sum(not c.isalnum() for c in username) / length
            vowel_ratio = sum(c.lower() in 'aeiou' for c in username) / length
            consecutive_digits = self._count_consecutive_chars(username, str.isdigit)
            consecutive_chars = self._count_consecutive_chars(username, str.isalpha)
            
            # Calculate entropy
            entropy = self._calculate_entropy(username)
            
            # Pattern-based scoring
            pattern_score = 0
            if digit_ratio > 0.5: pattern_score += 0.3
            if length < 4: pattern_score += 0.2
            if consecutive_digits > 3: pattern_score += 0.3
            if vowel_ratio < 0.1: pattern_score += 0.2
            if entropy < 0.5: pattern_score += 0.3
            
            return min(pattern_score, 1.0)
        except:
            return 0.5

    def _count_consecutive_chars(self, text, condition):
        ## counting max consecutive chars meeting condition
        max_count = current_count = 0
        for char in text:
            current_count = current_count + 1 if condition(char) else 0
            max_count = max(max_count, current_count)
        return max_count

    def _calculate_entropy(self, text):
        ## calculating Shannon entropy
        counts = Counter(text)
        length = len(text)
        if length <= 1:
            return 0
        entropy = -sum((c/length) * np.log2(c/length) for c in counts.values())
        max_entropy = np.log2(len(counts))
        return entropy / max_entropy if max_entropy > 0 else 0

    def analyze_review_content_ml(self, text):
        ## ML-enhanced review content analysis
        if pd.isna(text) or len(str(text).strip()) < 10:
            return 1.0
            
        try:
            text_str = str(text)[:512]
            
            # Image URL penalty
            image_urls = self.extract_image_urls_from_text(text_str)
            image_penalty = min(0.1 * len(image_urls), 0.3)
            
            # Text analysis
            if self.fake_review_classifier:
                try:
                    result = self.fake_review_classifier(text_str)[0]
                    toxicity_score = result['score'] if result['label'] == 'TOXIC' else 0
                except:
                    toxicity_score = 0
            else:
                toxicity_score = 0
            
            # Linguistic analysis
            linguistic_score = self._analyze_linguistic_patterns(text_str)
            sentiment_score = self._analyze_sentiment_coherence(text_str)
            complexity_score = self._analyze_text_complexity(text_str)
            
            final_score = (toxicity_score * 0.25 + linguistic_score * 0.25 +
                          sentiment_score * 0.2 + complexity_score * 0.2 + 
                          image_penalty * 0.1)
            return min(final_score, 1.0)
        except:
            return 0.5

    def _analyze_linguistic_patterns(self, text):
        words = text.split()
        if not words:
            return 1.0
            
        avg_word_length = np.mean([len(w) for w in words])
        unique_ratio = len(set(words)) / len(words)
        exclamation_ratio = text.count('!') / max(len(text), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        score = 0
        if avg_word_length < 3 or avg_word_length > 8: score += 0.2
        if unique_ratio < 0.5: score += 0.3
        if exclamation_ratio > 0.05: score += 0.2
        if caps_ratio > 0.3: score += 0.3
        
        return min(score, 1.0)

    def _analyze_sentiment_coherence(self, text):
        pos_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best']
        neg_words = ['bad', 'terrible', 'worst', 'hate', 'awful', 'horrible']
        
        text_lower = text.lower()
        pos = sum(w in text_lower for w in pos_words)
        neg = sum(w in text_lower for w in neg_words)
        
        if pos and neg:
            return min(pos, neg) / max(pos, neg)
        return 0

    def _analyze_text_complexity(self, text):
        sentences = [s for s in text.split('.') if s.strip()]
        if not sentences:
            return 1.0
            
        avg_len = np.mean([len(s.split()) for s in sentences])
        word_counts = Counter(text.split())
        repetition = sum(1 for c in word_counts.values() if c > 2) / max(len(word_counts), 1)
        
        score = 0
        if avg_len < 3 or avg_len > 25: score += 0.3
        if repetition > 0.3: score += 0.4
        
        return min(score, 1.0)

    def detect_copy_paste_reviews_ml(self, reviews):
        if len(reviews) < 2:
            return pd.Series([0.0] * len(reviews))
            
        try:
            clean_reviews = reviews.fillna('').astype(str).tolist()
            embeddings = self.sentence_model.encode(clean_reviews, batch_size=32)
            similarity = cosine_similarity(embeddings)
            np.fill_diagonal(similarity, 0)
            max_sim = np.max(similarity, axis=1)
            
            scores = []
            for s in max_sim:
                if s > 0.95:
                    scores.append(1.0)  # near-exact match
                elif s > 0.85:
                    scores.append(0.8)  # very high match
                elif s > 0.7:
                    scores.append(0.6)  # high match
                elif s > 0.5:
                    scores.append(0.3)  # moderate match
                else:
                    scores.append(0.0)  # low match
            
            return pd.Series(scores)
        except:
            return pd.Series([0.0] * len(reviews))

    def calculate_product_confidence(self, df):
        
        # grouping by product and calculate metrics
        product_confidence = df.groupby('product_id').apply(
            lambda x: pd.Series({
                'review_confidence': (1 - x['review_suspicion'].mean()),
                'image_confidence': (1 - x['product_image_duplicate_score'].iloc[0])
            })
        ).reset_index()
        
        product_confidence['product_confidence'] = (
            product_confidence['review_confidence'] * 0.7 + 
            product_confidence['image_confidence'] * 0.3
        ) * 100
        
        # rounding and format
        product_confidence['product_confidence'] = product_confidence['product_confidence'].round(2)
        
        # saving res
        product_confidence[['product_id', 'product_confidence']].to_csv(
            'product_confidence.csv', 
            index=False
        )
        
        print("Product confidence scores saved to product_confidence.csv")

    def calculate_final_authenticity_score(self, df):
        
        # analyses
        product_duplicate_scores = self.analyze_cross_product_image_duplicates_ml(df)
        review_image_scores = self.analyze_review_images(df)
        
        # apply scores to dataframe
        df['product_image_duplicate_score'] = df['product_id'].map(product_duplicate_scores).fillna(0.0)
        df['review_image_score'] = review_image_scores
        df['username_suspicion'] = df['user_name'].apply(self.analyze_username_patterns_ml)
        df['review_suspicion'] = df['review_content'].apply(self.analyze_review_content_ml)
        df['copy_paste_score'] = self.detect_copy_paste_reviews_ml(df['review_content'])
        
        # calculating weighted fraud score
        df['fraud_score'] = (
            df['product_image_duplicate_score'] * 0.25 +
            df['review_image_score'] * 0.15 +
            df['username_suspicion'] * 0.20 +
            df['review_suspicion'] * 0.25 +
            df['copy_paste_score'] * 0.15
        )
        
        # authenticity score 
        df['authenticity_score'] = ((1 - df['fraud_score']) * 100).round(2)
        
        return df

    def full_analysis_pipeline(self, df):
        # Ensuring required columns exist
        required_columns = [
            'product_id', 'product_name', 'category', 'discounted_price', 
            'actual_price', 'discount_percentage', 'rating', 'rating_count',
            'about_product', 'user_id', 'user_name', 'review_id', 
            'review_title', 'review_content', 'img_link', 'product_link'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # clean data
        df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce').fillna(0)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0)
        df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce').fillna(0)
        df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce').fillna(0)
        
        # complete analysis
        results_df = self.calculate_final_authenticity_score(df)
        
        output_columns = [
            'product_id', 'product_name', 'category', 'discounted_price', 
            'actual_price', 'discount_percentage', 'rating', 'rating_count',
            'about_product', 'user_id', 'user_name', 'review_id', 
            'review_title', 'review_content', 'img_link', 'product_link',
            'product_image_duplicate_score', 'review_image_score', 
            'username_suspicion', 'review_suspicion', 'copy_paste_score', 
            'fraud_score', 'authenticity_score'
        ]
        
        for col in output_columns:
            if col not in results_df.columns:
                if col in ['product_image_duplicate_score', 'review_image_score', 
                          'username_suspicion', 'review_suspicion', 'copy_paste_score', 
                          'fraud_score']:
                    results_df[col] = 0.0
                elif col == 'authenticity_score':
                    results_df[col] = 100.0
                else:
                    results_df[col] = ''
        
        numerical_cols = ['product_image_duplicate_score', 'review_image_score', 
                         'username_suspicion', 'review_suspicion', 'copy_paste_score', 
                         'fraud_score', 'authenticity_score']
        
        for col in numerical_cols:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0).round(4)
        
        results_df['authenticity_score'] = results_df['authenticity_score'].astype(float)
        
        final_df = results_df[output_columns].copy()
        
        # Calculate and save product confidence scores
        self.calculate_product_confidence(final_df)
        
        print(f"Analysis complete. Processed {len(final_df)} reviews.")
        print(f"Average authenticity score: {final_df['authenticity_score'].mean():.2f}")
        
        return final_df

def main():
    """Main execution function"""
    try:
        # initialize
        analyzer = EnhancedAIFraudDetector()
        
        # load data
        print("Loading data...")
        df = pd.read_csv("cleaned_data.csv")
        print(f"Loaded {len(df)} records")
        
        # complete analysis
        results_df = analyzer.full_analysis_pipeline(df)
        
        # save res
        output_filename = "analysis_results.csv"
        results_df.to_csv(output_filename, index=False)
        
        print(f"\n<< Analysis Complete >>")
        print(f"Total reviews analyzed: {len(results_df)}")
        print(f"\nResults saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()