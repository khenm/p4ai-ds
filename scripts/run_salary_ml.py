import os
import json
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

from src.preprocess.tabular_preprocess import load_data
from src.models.salary.train import initialize_models, train_models
from src.models.salary.evaluate import evaluate_models

# Ensure working directory is project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def export_pipeline_mappings(fitted_pipeline, export_path):
    """
    Extracts Ordinal and Target mappings dynamically from the fitted sklearn Pipeline to JSON.
    """
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Navigate: Pipeline -> 'preprocessor' ColumnTransformer -> 'nom' or 'ord' Pipeline -> TargetEncoder or OrdinalEncoder
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    
    # 1. Extract Target Encoder mappings
    # The actual transformer is nested inside the 'nom' branch Pipeline
    target_encoder = preprocessor.named_transformers_['nom'].named_steps['target_encoder']
    
    # TargetEncoder internally creates an OrdinalEncoder. We need its category_mapping to map integer keys back to String labels.
    target_str_maps = {}
    for ord_dict in target_encoder.ordinal_encoder.category_mapping:
        col_name = ord_dict['col']
        target_str_maps[col_name] = {v: k for k, v in ord_dict['mapping'].items() if not pd.isna(k)}
        
    target_mappings = {}
    for col_name, mapping_series in target_encoder.mapping.items():
        temp_dict = mapping_series.to_dict()
        if col_name in target_str_maps:
            clean_dict = {str(target_str_maps[col_name].get(k, k)): float(v) for k, v in temp_dict.items()}
        else:
            clean_dict = {str(k): float(v) for k, v in temp_dict.items()}
        target_mappings[col_name] = clean_dict

    # 2. Extract Ordinal Encoder mappings
    ordinal_encoder = preprocessor.named_transformers_['ord'].named_steps['ordinal_encoder']
    ord_maps = {}
    for ord_dict in ordinal_encoder.category_mapping:
        col_name = ord_dict['col']
        # Reverse mapping: integer -> string category for UI display
        ord_maps[col_name] = {int(v): str(k) for k, v in ord_dict['mapping'].items() if not pd.isna(k) and str(k).lower() != 'nan'}

    # Bundle and export
    final_mapping = {
        'Target': target_mappings,
        'Ordinal': ord_maps
    }
    
    with open(export_path, 'w') as f:
        json.dump(final_mapping, f, indent=4)
    print(f"Exported Preprocessing JSON maps to {export_path}")


def run_pipeline():
    print("--- Starting ML Pipeline with Scikit-Learn Pipelines ---")
    
    # 1. Load configuration and raw split data
    data_path = "data/jobsalary/job_salary_prediction_dataset.csv"
    results_dir = "ui/assets/data/jobsalary_ml"
    
    print("1. Loading raw dataset and performing split...")
    X_train, X_test, y_train, y_test = load_data(data_path)
    print(f"   Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # 2. Initialize Models wrapped in Sklearn Pipelines
    print("\n2. Initializing Models with embedded Preprocessing Pipelines...")
    models = initialize_models()
    
    # 3. Train all Pipelines
    print("\n3. Training Pipelines...")
    import pandas as pd # Needed above in export mapping
    fitted_models = train_models(X_train, y_train, models)
    
    # 4. Evaluate and generate UI JSON payloads
    print("\n4. Evaluating Models on Test Set...")
    evaluate_models(fitted_models, X_test, y_test, results_dir)
    
    # 5. Extract Feature Transformations JSON
    print("\n5. Extracting internal Scaler Mappings for Dashboard visualizations...")
    mapping_path = os.path.join(results_dir, "preprocessing_mappings.json")
    export_pipeline_mappings(fitted_models['XGBoost'], mapping_path)
    
    print("\n--- Pipeline Complete! Refresh the Salary ML Dashboard... ---")

if __name__ == "__main__":
    run_pipeline()
