from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def initialize_models():
    """
    Initializes models according to the EDA roadmap.
    Includes Linear Regression as a baseline, and Tree-based models 
    to capture the Exponential Multiplier effects.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1)
    }
    return models

def train_models(X_train, y_train, models):
    """
    Trains a dictionary of models on the training dataset.
    Returns the fitted models.
    """
    fitted_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
    return fitted_models
