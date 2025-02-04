# Credit Card Fraud Detection: Handling Imbalanced Datasets  

This project is a credit card fraud detection model based on the Kaggle notebook by [Janio Bachmann](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets). The main focus of this project is to handle highly imbalanced datasets effectively and build a robust classification model for fraud detection.  

## 🔍 Overview  

Credit card fraud detection is a critical task in the financial sector, where fraudulent transactions are rare compared to legitimate ones. This imbalance makes it difficult for traditional machine learning models to perform well. The approach in this project includes:  

- **Data Exploration**: Understanding the dataset and its distribution.  
- **Feature Engineering**: Applying transformations and selecting relevant features.  
- **Handling Imbalanced Data**: Using techniques like oversampling (SMOTE), undersampling, and cost-sensitive learning.  
- **Model Training**: Implementing machine learning models such as Logistic Regression, Random Forest, and Gradient Boosting.  
- **Evaluation Metrics**: Assessing model performance using precision, recall, F1-score, and ROC-AUC.  

## 📂 Dataset  

The dataset used in this project is the **Credit Card Fraud Detection Dataset** from Kaggle, which contains anonymized transaction data with labeled fraud and non-fraud transactions.  

- **Features**: 28 PCA-transformed numerical features + Time & Amount.  
- **Target Variable**:  
  - `0` → Legitimate Transaction  
  - `1` → Fraudulent Transaction  

## 🛠️ Steps in the Project  

1. **Data Preprocessing**  
   - Load and explore the dataset.  
   - Check class distribution and identify imbalance.  
   - Scale numerical features.  

2. **Handling Imbalance**  
   - Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.  
   - Try **undersampling** to remove excess majority class samples.  
   - Use **class-weight adjustments** in models.  

3. **Model Training & Evaluation**  
   - Train models including:  
     - Logistic Regression  
     - Random Forest  
     - Gradient Boosting (XGBoost, LightGBM)  
   - Evaluate models using precision, recall, F1-score, and ROC-AUC.  

4. **Model Optimization**  
   - Hyperparameter tuning using GridSearchCV.  
   - Feature selection to reduce overfitting.  

## 🚀 Results & Insights  

- Handling imbalance improves fraud detection without compromising overall performance.  
- SMOTE and cost-sensitive learning significantly enhance recall for fraud cases.  
- The best-performing model balances precision and recall effectively.  

## 📌 Requirements  

To run the notebook, install the following dependencies:  

```bash  
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm  
```

## 📜 Usage  

1. Clone this repository:  

   ```bash  
   git clone https://github.com/your-username/credit-fraud-detection.git  
   cd credit-fraud-detection  
   ```

2. Open and run the Jupyter notebook:  

   ```bash  
   jupyter notebook main.ipynb  
   ```

## 📚 References  

- Kaggle Notebook: [Credit Fraud - Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)  
- Imbalanced-learn Documentation: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)  
