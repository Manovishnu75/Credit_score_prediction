# Credit Score Prediction - Quick Start Guide

## 🚀 Get Started in 5 Minutes!

### Step 1: Download the Dataset from Kaggle

**Option A: Manual Download**
1. Visit: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
2. Click the "Download" button (you may need to sign in)
3. Extract the ZIP file
4. Copy `train.csv` to your project folder

**Option B: Using Kaggle API (Faster)**
```bash
# Install Kaggle
pip install kaggle

# Download Kaggle API token
# 1. Go to kaggle.com → Your Profile → Account
# 2. Scroll to "API" section
# 3. Click "Create New API Token"
# 4. This downloads kaggle.json

# Place kaggle.json in the right location
# Windows: C:\Users\<YourUsername>\.kaggle\kaggle.json
# Mac/Linux: ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d parisrohan/credit-score-classification
unzip credit-score-classification.zip
```

### Step 2: Install Required Libraries
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Step 3: Run the Project

**Option A: Run Python Script**
```bash
python credit_score_prediction.py
```

**Option B: Use Jupyter Notebook**
```bash
jupyter notebook credit_score_prediction.ipynb
```

### Step 4: View Results
After running, check these generated files:
- `credit_score_distribution.png` - Distribution chart
- `confusion_matrix_*.png` - Model performance
- `feature_importance.png` - Important features

---

## 📊 What This Project Does

1. **Loads** credit score data from Kaggle
2. **Analyzes** customer financial patterns
3. **Trains** 3 ML models (Logistic Regression, Random Forest, Gradient Boosting)
4. **Predicts** credit scores (Good/Standard/Poor)
5. **Evaluates** model performance
6. **Identifies** most important financial factors

---

## 🎯 Expected Accuracy
- Logistic Regression: ~75%
- Random Forest: ~80%
- Gradient Boosting: ~81%

---

## 💡 Key Features in Dataset
- Age, Income, Occupation
- Number of bank accounts & credit cards
- Outstanding debt & payment history
- Credit utilization ratio
- Delayed payments
- And 15+ more features!

---

## 🆘 Need Help?

**Dataset not found?**
- Make sure `train.csv` is in the same folder as the script

**Import errors?**
- Run: `pip install -r requirements.txt`

**Low accuracy?**
- Try increasing n_estimators in Random Forest
- Check for data quality issues

**Want to customize?**
- Edit model parameters in the script
- Add more models
- Try different preprocessing techniques

---

## 📚 Learn More

Check the full README.md for:
- Detailed documentation
- Dataset description
- Customization options
- Advanced techniques
- Troubleshooting guide

---

**Happy Learning! 🎓**

For questions, refer to README.md or check the comments in the code.
