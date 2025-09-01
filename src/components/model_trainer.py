from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from src.components.data_ingestion import DataIngestionConfig
from src.exception import CustomException
import os
import sys
import pandas as pd


# loading data from artifact
data_config=DataIngestionConfig()
try:
    train_data_path=data_config.train_data_path
    test_data_path=data_config.test_data_path
    test_data=pd.read_csv(test_data_path)
    train_data=pd.read_csv(train_data_path)
    X_train=train_data.drop(['diabetes'],axis=1)
    X_test=test_data.drop(['diabetes'],axis=1)
    y_train=train_data['diabetes']
    y_test=test_data['diabetes']
    
except Exception as e:
    raise CustomException(e,sys)

# Dictionary to hold models and their param grids
models = {
    "MultinomialNB": {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.01, 0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
    },
    "LogisticRegression": {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
    },
    "RandomForest": {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
    },
    "SVM": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ]),
        'params': {
            'svc__C': [0.1, 1, 10],
            'svc__kernel': ['linear', 'rbf']
        }
    },
    "KNN": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ]),
        'params': {
            'knn__n_neighbors': [3, 5, 7, 9],
            'knn__weights': ['uniform', 'distance']
        }
    },
    "DecisionTree": {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    "GradientBoosting": {
        'model': GradientBoostingClassifier(),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5]
        }
    },
    "AdaBoost": {
        'model': AdaBoostClassifier(),
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0, 1.5]
        }
    },
    "SGDClassifier": {
        'model': SGDClassifier(max_iter=1000, tol=1e-3),
        'params': {
            'loss': ['hinge', 'log_loss'],
            'alpha': [0.0001, 0.001, 0.01]
        }
    },
    "MLPClassifier": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(max_iter=500))
        ]),
        'params': {
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu', 'tanh'],
            'mlp__alpha': [0.0001, 0.001]
        }
    }
}


# Run grid search for each model
best_models = {}

for name, m in models.items():
    print(f"\nTraining and tuning {name}...")
    grid = GridSearchCV(m['model'], m['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print(f"Best parameters for {name}: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    
    # Test accuracy
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_acc:.4f}")
    
    best_models[name] = {
        'model': best_model,
        'cv_accuracy': grid.best_score_,
        'test_accuracy': test_acc
    }