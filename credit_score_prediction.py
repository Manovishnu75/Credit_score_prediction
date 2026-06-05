"""
Credit Score Prediction - Machine Learning Project
Cross-platform version (Windows/Linux/Mac compatible)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class CreditScorePredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        
        # Create output directory for saving files (cross-platform)
        self.output_dir = 'output_files'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
    def load_data(self):
        """Load the dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nDataset Info:")
        print(self.df.info())
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Missing values
        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found!")
        
        # Target variable distribution
        if 'Credit_Score' in self.df.columns:
            print("\nCredit Score Distribution:")
            print(self.df['Credit_Score'].value_counts())
            
            # Plot credit score distribution
            plt.figure(figsize=(10, 6))
            self.df['Credit_Score'].value_counts().plot(kind='bar', color='skyblue')
            plt.title('Credit Score Distribution', fontsize=16, fontweight='bold')
            plt.xlabel('Credit Score Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save with cross-platform path
            filepath = os.path.join(self.output_dir, 'credit_score_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
            plt.close()
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Make a copy
        df_processed = self.df.copy()
        
        # Handle missing values
        print("\nHandling missing values...")
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Remove unwanted columns (like ID, Name, etc.)
        cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Unnamed: 0']
        cols_to_drop = [col for col in cols_to_drop if col in df_processed.columns]
        if cols_to_drop:
            df_processed.drop(cols_to_drop, axis=1, inplace=True)
            print(f"Dropped columns: {cols_to_drop}")
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from categorical encoding if present
        if 'Credit_Score' in categorical_cols:
            categorical_cols.remove('Credit_Score')
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
            print(f"  Encoded: {col}")
        
        # Encode target variable if it's categorical
        if 'Credit_Score' in df_processed.columns:
            if df_processed['Credit_Score'].dtype == 'object':
                le_target = LabelEncoder()
                df_processed['Credit_Score'] = le_target.fit_transform(df_processed['Credit_Score'])
                self.label_encoders['Credit_Score'] = le_target
                print(f"  Target variable encoded: Credit_Score")
                print(f"  Classes: {le_target.classes_}")
            else:
                # Target is already numeric, create mapping
                unique_values = sorted(df_processed['Credit_Score'].unique())
                print(f"  Target variable is numeric with values: {unique_values}")
                # Create a dummy label encoder for consistency
                class DummyEncoder:
                    def __init__(self, classes):
                        self.classes_ = np.array([f"Class_{i}" for i in classes])
                    def inverse_transform(self, y):
                        return np.array([self.classes_[int(i)] for i in y])
                self.label_encoders['Credit_Score'] = DummyEncoder(unique_values)
        
        self.df_processed = df_processed
        print("\nPreprocessing completed!")
        print(f"Final dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def prepare_features(self):
        """Prepare features and target for modeling"""
        print("\n" + "="*50)
        print("FEATURE PREPARATION")
        print("="*50)
        
        # Separate features and target
        X = self.df_processed.drop('Credit_Score', axis=1)
        y = self.df_processed['Credit_Score']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeature columns: {list(X.columns)}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("\nFeatures scaled successfully!")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        # Train each model
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = accuracy
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
        
        return results
    
    def evaluate_models(self):
        """Evaluate all models and display results"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"{name} Evaluation")
            print(f"{'='*50}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            
            # Classification Report
            print("\nClassification Report:")
            target_names = self.label_encoders['Credit_Score'].classes_
            print(classification_report(self.y_test, y_pred, target_names=target_names))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=target_names, yticklabels=target_names)
            plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            
            # Save with cross-platform path
            filename = f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
            plt.close()
    
    def feature_importance(self):
        """Display feature importance for tree-based models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        # Get feature names
        feature_names = [col for col in self.df_processed.columns if col != 'Credit_Score']
        
        # Random Forest feature importance
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Most Important Features (Random Forest):")
        for i in range(min(10, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_n = min(15, len(feature_names))
        plt.barh(range(top_n), importances[indices[:top_n]], color='teal')
        plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
        plt.xlabel('Importance Score')
        plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save with cross-platform path
        filepath = os.path.join(self.output_dir, 'feature_importance.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {filepath}")
        plt.close()
    
    def predict_new_sample(self, sample_data):
        """Predict credit score for new data"""
        # Use best model (Random Forest typically performs well)
        best_model = self.models['Random Forest']
        
        # Scale the sample
        sample_scaled = self.scaler.transform([sample_data])
        
        # Predict
        prediction = best_model.predict(sample_scaled)
        prediction_proba = best_model.predict_proba(sample_scaled)
        
        # Decode prediction
        credit_score = self.label_encoders['Credit_Score'].inverse_transform(prediction)[0]
        
        return credit_score, prediction_proba
    
    def run_complete_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("\n" + "="*60)
        print("CREDIT SCORE PREDICTION - MACHINE LEARNING PROJECT")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Prepare features
        self.prepare_features()
        
        # Train models
        results = self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Feature importance
        self.feature_importance()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nModel Performance Summary:")
        for model_name, accuracy in results.items():
            print(f"  {model_name}: {accuracy:.4f}")
        
        print(f"\nAll visualizations have been saved to: {self.output_dir}/")
        print("\nFiles generated:")
        print(f"  - {os.path.join(self.output_dir, 'credit_score_distribution.png')}")
        print(f"  - {os.path.join(self.output_dir, 'confusion_matrix_*.png')} (for each model)")
        print(f"  - {os.path.join(self.output_dir, 'feature_importance.png')}")


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    # You'll need to update this path to your actual dataset
    predictor = CreditScorePredictor('train.csv')
    
    # Run the complete pipeline
    predictor.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("Project completed! Check the 'output_files' folder for visualizations.")
    print("="*60)