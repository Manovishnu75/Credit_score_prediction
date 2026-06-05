"""
Sample Prediction Script
Use this after training your model to make predictions on new data
"""

import pandas as pd
import numpy as np
import pickle
from credit_score_prediction import CreditScorePredictor

def save_trained_model(predictor, filename='credit_score_model.pkl'):
    """
    Save the trained model and preprocessing objects
    """
    model_data = {
        'model': predictor.models['Random Forest'],
        'scaler': predictor.scaler,
        'label_encoders': predictor.label_encoders,
        'feature_names': [col for col in predictor.df_processed.columns if col != 'Credit_Score']
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {filename}")


def load_trained_model(filename='credit_score_model.pkl'):
    """
    Load the trained model and preprocessing objects
    """
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model loaded from {filename}")
    return model_data


def predict_single_customer(model_data, customer_features):
    """
    Predict credit score for a single customer
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model, scaler, and encoders
    customer_features : dict
        Dictionary with feature names as keys and values
    
    Returns:
    --------
    prediction : str
        Predicted credit score category
    probabilities : dict
        Probability for each category
    """
    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoders']['Credit_Score']
    feature_names = model_data['feature_names']
    
    # Convert to array in correct order
    feature_values = [customer_features.get(name, 0) for name in feature_names]
    
    # Scale features
    features_scaled = scaler.transform([feature_values])
    
    # Predict
    prediction_encoded = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Decode prediction
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Create probability dictionary
    prob_dict = {
        label: prob 
        for label, prob in zip(label_encoder.classes_, probabilities)
    }
    
    return prediction, prob_dict


def predict_from_csv(model_data, csv_file, output_file='predictions.csv'):
    """
    Make predictions for multiple customers from a CSV file
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing model, scaler, and encoders
    csv_file : str
        Path to CSV file with customer data
    output_file : str
        Path to save predictions
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Extract components
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoders']['Credit_Score']
    feature_names = model_data['feature_names']
    
    # Prepare features
    X = df[feature_names]
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    predictions_encoded = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Decode predictions
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    # Add to dataframe
    df['Predicted_Credit_Score'] = predictions
    
    # Add probabilities for each class
    for i, class_name in enumerate(label_encoder.classes_):
        df[f'Probability_{class_name}'] = probabilities[:, i]
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return df


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("CREDIT SCORE PREDICTION - SAMPLE USAGE")
    print("="*60)
    
    # Step 1: Train the model (do this once)
    print("\n1. Training model...")
    predictor = CreditScorePredictor('train.csv')
    predictor.run_complete_pipeline()
    
    # Step 2: Save the trained model
    print("\n2. Saving model...")
    save_trained_model(predictor)
    
    # Step 3: Load the model for predictions
    print("\n3. Loading model...")
    model_data = load_trained_model()
    
    # Step 4: Example - Predict for a single customer
    print("\n4. Example prediction for single customer:")
    print("-" * 60)
    
    # This is just an example - replace with actual feature values
    sample_customer = {
        'Age': 35,
        'Annual_Income': 50000,
        'Monthly_Inhand_Salary': 4000,
        'Num_Bank_Accounts': 3,
        'Num_Credit_Card': 2,
        'Interest_Rate': 15,
        'Num_of_Loan': 2,
        'Delay_from_due_date': 10,
        'Num_of_Delayed_Payment': 3,
        'Outstanding_Debt': 5000,
        'Credit_Utilization_Ratio': 30.5,
        'Total_EMI_per_month': 1500,
        'Amount_invested_monthly': 500,
        'Monthly_Balance': 1000
        # Add all other features from your dataset
    }
    
    prediction, probabilities = predict_single_customer(model_data, sample_customer)
    
    print(f"\nPredicted Credit Score: {prediction}")
    print(f"\nProbabilities:")
    for category, prob in probabilities.items():
        print(f"  {category}: {prob:.2%}")
    
    # Step 5: Example - Predict from CSV file
    # Uncomment the following if you have a CSV file with new customers
    # print("\n5. Predicting from CSV file...")
    # predictions_df = predict_from_csv(model_data, 'new_customers.csv', 'predictions.csv')
    # print(predictions_df.head())
    
    print("\n" + "="*60)
    print("SAMPLE USAGE COMPLETED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Use load_trained_model() to load your saved model")
    print("2. Use predict_single_customer() for single predictions")
    print("3. Use predict_from_csv() for batch predictions")
    print("\nTip: You can integrate this into a web app or API!")
