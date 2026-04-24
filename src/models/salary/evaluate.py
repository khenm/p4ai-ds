import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_models(fitted_models, X_test, y_test, results_dir='ui/assets/data/jobsalary_ml'):
    """
    Evaluates trained models, captures metrics, and outputs them as JSON for the Dashboard UI.
    Also extracts Feature Importances for Tree-based models.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    evaluation_results = []
    feature_importances = {}

    for name, model in fitted_models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results.append({
            'Model': name,
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2_Score': float(r2)
        })

        # Extract Feature Importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Depending on TargetEncoder, the columns of X_test remain unchanged
            feat_imp = pd.DataFrame({
                'Feature': X_test.columns,
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
