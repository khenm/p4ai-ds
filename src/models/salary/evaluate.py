import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(fitted_models, X_test, y_test, results_dir='ui/assets/data/jobsalary_ml'):
    """
    Evaluates trained Pipeline models, captures metrics, and outputs them as JSON for the Dashboard UI.
    Extracts Feature Importances dynamically from the regressors within the Pipelines.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    evaluation_results = []
    feature_importances = {}

    for name, pipeline in fitted_models.items():
        print(f"Evaluating {name}...")
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results.append({
            'Model': name,
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2_Score': float(r2)
        })

        # Extract Regressor and Preprocessor from the Pipeline
        regressor = pipeline.named_steps['regressor']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Extract Feature Importances if available
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            
            feat_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Save into dictionary format for JSON
            feature_importances[name] = feat_imp.to_dict(orient='records')

    # Save Metrics File
    metrics_path = os.path.join(results_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
        
    # Save Feature Importance File
    importance_path = os.path.join(results_dir, 'feature_importances.json')
    with open(importance_path, 'w') as f:
        json.dump(feature_importances, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {results_dir}")
    return evaluation_results, feature_importances
