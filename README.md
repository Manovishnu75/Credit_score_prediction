# Credit Score Prediction - Machine Learning Project

## 📊 Project Overview
This project predicts credit scores based on customer financial data using machine learning algorithms. The system classifies customers into different credit score categories (Good, Standard, Poor) based on various financial and demographic features.

## 🎯 Objective
To build a machine learning model that can accurately predict a customer's credit score category based on their financial behavior and demographic information.

## 📁 Kaggle Dataset

### Dataset Information
**Dataset Name:** Credit Score Classification Dataset

**Kaggle Link:** https://www.kaggle.com/datasets/parisrohan/credit-score-classification

### How to Download from Kaggle

#### Method 1: Manual Download
1. Go to: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
2. Click on "Download" button
3. Extract the zip file
4. You'll get `train.csv` and `test.csv` files
5. Place `train.csv` in your project directory

#### Method 2: Using Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle API (you need to download kaggle.json from your Kaggle account)
# Go to: Kaggle Account Settings → API → Create New API Token
# Place kaggle.json in ~/.kaggle/ directory

# Download the dataset
kaggle datasets download -d parisrohan/credit-score-classification

# Unzip the dataset
unzip credit-score-classification.zip
```

### Dataset Description
The dataset contains the following types of features:

**Demographic Information:**
- Age
- Occupation
- Annual_Income
- Monthly_Inhand_Salary
- Num_Bank_Accounts
- Num_Credit_Card

**Credit Behavior:**
- Interest_Rate
- Num_of_Loan
- Delay_from_due_date
- Num_of_Delayed_Payment
- Changed_Credit_Limit
- Num_Credit_Inquiries
- Credit_Mix
- Outstanding_Debt
- Credit_Utilization_Ratio
- Credit_History_Age
- Payment_of_Min_Amount
- Total_EMI_per_month
- Amount_invested_monthly
- Payment_Behaviour
- Monthly_Balance

**Target Variable:**
- Credit_Score: Categories (Good, Standard, Poor)

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
# Create a new directory for the project
mkdir credit_score_prediction
cd credit_score_prediction
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset
- Download from Kaggle using one of the methods mentioned above
- Place `train.csv` in the project directory

### Step 4: Run the Project
```bash
python credit_score_prediction.py
```

## 📊 Machine Learning Models Used

The project implements and compares three machine learning algorithms:

1. **Logistic Regression**
   - Linear model for classification
   - Good baseline model
   - Fast training and prediction

2. **Random Forest Classifier**
   - Ensemble of decision trees
   - Handles non-linear relationships well
   - Provides feature importance

3. **Gradient Boosting Classifier**
   - Sequential ensemble method
   - High accuracy
   - Good for complex patterns

## 🔍 Project Features

### 1. Data Exploration
- Statistical summary of the dataset
- Missing value analysis
- Target variable distribution visualization
- Data type analysis

### 2. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling using StandardScaler
- Train-test split (80-20)

### 3. Model Training
- Training multiple ML models
- Cross-validation for better generalization
- Hyperparameter optimization

### 4. Model Evaluation
- Accuracy scores for all models
- Confusion matrices
- Classification reports (Precision, Recall, F1-score)
- Comparative analysis

### 5. Feature Importance Analysis
- Identifies most influential features
- Visualization of top features
- Helps in feature selection

### 6. Predictions
- Predict credit scores for new customers
- Probability estimates for each category

## 📈 Expected Output

The script will generate:

1. **Console Output:**
   - Dataset statistics
   - Model training progress
   - Evaluation metrics
   - Feature importance rankings

2. **Visualizations:**
   - `credit_score_distribution.png` - Distribution of credit scores
   - `confusion_matrix_logistic_regression.png` - Logistic Regression results
   - `confusion_matrix_random_forest.png` - Random Forest results
   - `confusion_matrix_gradient_boosting.png` - Gradient Boosting results
   - `feature_importance.png` - Top 15 important features

3. **Model Performance:**
   - Accuracy scores typically range from 75-85%
   - Random Forest and Gradient Boosting usually perform best

## 💻 Usage Example

### Basic Usage
```python
from credit_score_prediction import CreditScorePredictor

# Initialize predictor
predictor = CreditScorePredictor('train.csv')

# Run complete pipeline
predictor.run_complete_pipeline()
```

### Custom Prediction
```python
# After running the pipeline, you can make predictions
# Example: Create a sample with same features as your dataset
sample_data = [45, 2, 50000, 4000, 3, 2, ...]  # Feature values

# Predict
credit_score, probabilities = predictor.predict_new_sample(sample_data)
print(f"Predicted Credit Score: {credit_score}")
print(f"Probabilities: {probabilities}")
```

## 📊 Model Performance Metrics

### Typical Results:
- **Logistic Regression:** 72-76% accuracy
- **Random Forest:** 78-83% accuracy
- **Gradient Boosting:** 79-84% accuracy

### Evaluation Metrics:
- **Accuracy:** Overall correctness of the model
- **Precision:** How many selected items are relevant
- **Recall:** How many relevant items are selected
- **F1-Score:** Harmonic mean of precision and recall

## 🔧 Customization

### Modify Model Parameters
You can adjust model hyperparameters in the `train_models()` method:

```python
self.models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,  # Increase trees
        max_depth=15,      # Set max depth
        random_state=42
    ),
    # ... other models
}
```

### Add More Models
You can easily add more models:

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

self.models['SVM'] = SVC(kernel='rbf', random_state=42)
self.models['Decision Tree'] = DecisionTreeClassifier(random_state=42)
```

## 📂 Project Structure
```
credit_score_prediction/
│
├── credit_score_prediction.py    # Main script
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── train.csv                      # Dataset (download from Kaggle)
│
└── outputs/                       # Generated visualizations
    ├── credit_score_distribution.png
    ├── confusion_matrix_*.png
    └── feature_importance.png
```

## 🚀 Future Enhancements

1. **Deep Learning Models**
   - Implement Neural Networks
   - Try LSTM for sequential patterns

2. **Feature Engineering**
   - Create interaction features
   - Polynomial features
   - Time-based features

3. **Hyperparameter Tuning**
   - Grid Search CV
   - Random Search CV
   - Bayesian Optimization

4. **Deployment**
   - Create Flask/FastAPI web app
   - Build REST API
   - Deploy on cloud (AWS, Azure, GCP)

5. **Advanced Techniques**
   - Ensemble methods (Stacking, Voting)
   - XGBoost, LightGBM, CatBoost
   - AutoML solutions

## 🐛 Troubleshooting

### Common Issues:

1. **Import Error:**
   ```bash
   # Reinstall packages
   pip install -r requirements.txt --upgrade
   ```

2. **File Not Found Error:**
   - Ensure `train.csv` is in the same directory as the script
   - Check file path in the code

3. **Memory Error:**
   - Reduce dataset size for testing
   - Use sampling for large datasets

4. **Low Accuracy:**
   - Check data quality
   - Try feature engineering
   - Tune hyperparameters

## 📚 Learning Resources

- **Scikit-learn Documentation:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/
- **Credit Score Fundamentals:** Research credit scoring models and FICO scores
- **Machine Learning Course:** Andrew Ng's ML course on Coursera

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Add new models

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Machine Learning Project - Credit Score Prediction

## 📧 Contact

For questions or suggestions, please create an issue in the repository.

---

**Happy Learning! 🎓**
