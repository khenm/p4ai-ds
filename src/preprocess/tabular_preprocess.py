import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder, OrdinalEncoder

def load_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Loads salary data and splits into Train/Test natively without applying transformations.
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=['salary'])
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def build_column_transformer(is_tree: bool = True):
    """
    Returns a unified sklearn ColumnTransformer.
    - is_tree=True: skips StandardScaler for numerical variables.
    """
    numeric_cols = ['experience_years', 'skills_count']
    ordinal_cols = ['education_level', 'company_size']
    nominal_cols = ['job_title', 'industry', 'location', 'remote_work']

    # 1. Numeric Branch
    if is_tree:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('passthrough', 'passthrough')
        ])
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    # 2. Ordinal Branch
    education_order = [{'col': 'education_level', 'mapping': {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}}]
    company_order = [{'col': 'company_size', 'mapping': {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Enterprise': 4}}]
    
    ordinal_transformer = Pipeline(steps=[
        ('ordinal_encoder', OrdinalEncoder(mapping=education_order + company_order))
    ])

    # 3. Nominal Branch
    # smoothing is auto defaulted in TargetEncoder often, we explicitly enable it.
    nominal_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder(smoothing=10.0))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    return preprocessor

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data('../../../data/jobsalary/job_salary_prediction_dataset.csv')
    preprocessor = build_column_transformer(is_tree=True)
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    print("Preprocessing completed.")
    print("X_train shape:", X_train_trans.shape)
