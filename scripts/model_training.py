import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

def tune_hyperparameters_xgb():
    # Load selected features data for training
    X_train = pd.read_csv("../data/processed/X_train_selected.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    
    # Define a pipeline with XGBoost classifier
    pipeline = Pipeline([
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # Define hyperparameter grid for XGBoost
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2]
    }
    
    # Run grid search with 5-fold cross-validation using F1 scoring
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Save the best estimator
    joblib.dump(grid.best_estimator_, 'models/best_model_xgb.pkl')
    print(f"✅ Best hyperparameters (XGBoost): {grid.best_params_}")

def train_model_xgb():
    # First, perform hyperparameter tuning using XGBoost
    tune_hyperparameters_xgb()
    
    # Load selected feature data and target variables
    X_train = pd.read_csv("../data/processed/X_train_selected.csv")
    X_test = pd.read_csv("../data/processed/X_test_selected.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()
    
    # Address class imbalance using SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Load the best XGBoost model from hyperparameter tuning
    model = joblib.load('models/best_model_xgb.pkl')
    model.fit(X_train_smote, y_train_smote)
    
    # Evaluate model performance on test data
    y_pred = model.predict(X_test)
    print("Classification Report (XGBoost):")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    
    # Save the final trained model
    joblib.dump(model, 'models/final_model_xgb.pkl')
    print("✅ Final XGBoost model saved as 'models/final_model_xgb.pkl'")

if __name__ == "__main__":
    train_model_xgb()
