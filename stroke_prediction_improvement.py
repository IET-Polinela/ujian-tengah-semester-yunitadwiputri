import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve, 
                            precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath):
    """Load and preprocess the stroke dataset"""
    # Load data
    data = pd.read_csv(filepath)
    
    # Handle missing values in bmi
    data['bmi'].fillna(data['bmi'].median(), inplace=True)
    
    # Drop id column as it's not useful for prediction
    data.drop('id', axis=1, inplace=True)
    
    # Handle rare categories in smoking_status
    data['smoking_status'] = data['smoking_status'].replace('Unknown', 'never smoked')
    
    return data

def create_preprocessor():
    """Create preprocessing pipeline for numeric and categorical features"""
    numeric_features = ['age', 'avg_glucose_level', 'bmi']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 
                           'work_type', 'Residence_type', 'smoking_status']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and plot metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Print metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Feature Importance for tree-based models
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        feature_names = (['age', 'avg_glucose_level', 'bmi'] + 
                        list(model.named_steps['preprocessor']
                            .named_transformers_['cat']
                            .named_steps['onehot']
                            .get_feature_names_out(['gender', 'hypertension', 'heart_disease', 
                                                   'ever_married', 'work_type', 
                                                   'Residence_type', 'smoking_status'])))

        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.named_steps['classifier'].feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances.head(10))
        plt.title('Top 10 Important Features')
        plt.tight_layout()
        plt.show()

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('healthcare-dataset-stroke-data.csv')
    
    # Split data into features and target
    X = data.drop('stroke', axis=1)
    y = data['stroke']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Get preprocessor
    preprocessor = create_preprocessor()
    
    # Define models to try
    models = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
    }
    
    # Train and evaluate each model with SMOTE
    print("=== Evaluating Base Models with SMOTE ===")
    for name, model in models.items():
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        
        print(f"\n{name} Performance:")
        evaluate_model(pipeline, X_test, y_test)
    
    # Optimize Random Forest with GridSearchCV
    print("\n=== Optimizing Random Forest ===")
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
    print("\nPerforming grid search...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print("\nOptimized Random Forest Performance:")
    evaluate_model(best_model, X_test, y_test)
    
    # Try Ensemble Model
    print("\n=== Trying Ensemble Model ===")
    estimators = [
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=42,
                                    n_estimators=200, max_depth=20,
                                    min_samples_split=2, min_samples_leaf=1)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                             random_state=42))
    ]
    
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    ensemble_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', ensemble)
    ])
    
    print("\nTraining ensemble model...")
    ensemble_pipeline.fit(X_train, y_train)
    
    print("\nEnsemble Model Performance:")
    evaluate_model(ensemble_pipeline, X_test, y_test)

if __name__ == "__main__":
    main()
