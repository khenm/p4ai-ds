import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

def load_and_preprocess(data_path: str, test_size: float = 0.2, random_state: int = 42, export_mapping_path: str = None):
    """
    Loads salary data, splits into Train/Test, and applies EDA-driven encoding.
    """
    df = pd.read_csv(data_path)

    # 1. Prepare X and y
    X = df.drop(columns=['salary'])
    y = df['salary']

    # 2. Train-test split (CRITICAL: separate before Target Encoding to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Work on copies to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 3. Ordinal Encoding for Hierarchical Features (Based on EDA)
    education_order = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    company_order = {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Enterprise': 4}

    for dt in [X_train, X_test]:
        dt['education_level'] = dt['education_level'].map(education_order)
        dt['company_size'] = dt['company_size'].map(company_order)
        
        # Ensure 'experience_years' is int/float
        dt['experience_years'] = dt['experience_years'].astype(float)

    # 4. Target Encoding for Nominal Features
    # The EDA explicitly observed independent features with high cardinalities. 
    nominal_cols = ['job_title', 'industry', 'location', 'remote_work']
    
    # Initialize TargetEncoder
    te = TargetEncoder(cols=nominal_cols)
    
    # Fit strictly on Train subset, transform on both
    X_train = te.fit_transform(X_train, y_train)
    X_test = te.transform(X_test)
    
    # 5. Extract and Export Mappings (For the UI)
    if export_mapping_path:
        import json
        import os
        os.makedirs(os.path.dirname(export_mapping_path), exist_ok=True)
        
        target_mappings = {}
        
        # Get ordinal integer to string mapping from internal ordinal_encoder
        ord_maps = {}
        for ord_dict in te.ordinal_encoder.category_mapping:
            col_name = ord_dict['col']
            # Reverse mapping: integer -> string category
            ord_maps[col_name] = {v: k for k, v in ord_dict['mapping'].items()}
            
        for col_name, mapping_series in te.mapping.items():
            temp_dict = mapping_series.to_dict()
            col_target_map = {}
            for k, v in temp_dict.items():
                if pd.isna(k) or str(k) in ['nan', '-1', '-2']:
                    continue
                # k is the internal integer. Lookup real string name.
                real_name = ord_maps[col_name].get(k, str(k))
                col_target_map[str(real_name)] = float(v)
            target_mappings[col_name] = col_target_map

        mappings_export = {
            'Ordinal': {
                'Education Level': education_order,
                'Company Size': company_order
            },
            'Target': target_mappings
        }
        
        with open(export_mapping_path, 'w') as f:
            json.dump(mappings_export, f, indent=4)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Simple self-test
    X_train, X_test, y_train, y_test = load_and_preprocess('../../../data/jobsalary/job_salary_prediction_dataset.csv')
    print("Preprocessing completed.")
    print("X_train shape:", X_train.shape)
    print("X_train head:\n", X_train.head(3))
