from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from src.preprocess.tabular_preprocess import build_column_transformer

def initialize_models():
    """
    Initializes models according to the EDA roadmap, encapsulated inside robust sklearn Pipelines.
    Includes Linear Regression as a baseline, and Tree-based models.
    """
    linear_preprocessor = build_column_transformer(is_tree=False)
    tree_preprocessor = build_column_transformer(is_tree=True)

    models = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', linear_preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', tree_preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'XGBoost': Pipeline(steps=[
            ('preprocessor', tree_preprocessor),
            ('regressor', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1))
        ]),
        'LightGBM': Pipeline(steps=[
            ('preprocessor', tree_preprocessor),
            ('regressor', LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1))
        ])
    }
    return models

def train_models(X_train, y_train, models):
    """
    Trains a dictionary of Pipeline models on the training dataset.
    Returns the fitted models.
    """
    fitted_models = {}
    for name, pipeline in models.items():
        print(f"Training Pipeline: {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        
    return fitted_models
