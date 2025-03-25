from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# Define feature lists as module-level constants
CATEGORICAL_FEATURES = [
    'Gender', 'Location', 'Industry', 
    'Financial Needs'
]

NUMERICAL_FEATURES = [
    'Age', 'Income per year', 'amount_sum',
    'amount_mean', 'revenue_numeric', 'employee_mid'
]

def get_feature_lists():
    """Return dictionary of feature lists"""
    return {
        'categorical': CATEGORICAL_FEATURES.copy(),
        'numerical': NUMERICAL_FEATURES.copy()
    }

def create_feature_pipeline():
    """Create processing pipeline using the feature lists"""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ], verbose_feature_names_out=False)