import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download sentiment analysis model
nltk.download('vader_lexicon', quiet=True)

def convert_revenue(value):
    """Convert revenue strings like '150M-200M' to numeric average"""
    if isinstance(value, str):
        if '-' in value:
            low, high = value.split('-')
            return (parse_revenue(low) + parse_revenue(high)) / 2
        return parse_revenue(value)
    return float(value)

def parse_revenue(val):
    """Parse values like 150M to 150000000"""
    val = str(val).upper().replace(' ', '')
    multiplier = 1
    if 'M' in val:
        multiplier = 1_000_000
        val = val.replace('M', '')
    elif 'B' in val:
        multiplier = 1_000_000_000
        val = val.replace('B', '')
    try:
        return float(val) * multiplier
    except:
        return 0.0

def load_data():
    """Load and preprocess all datasets"""
    try:
        logger.info("Loading datasets...")
        data_path = Path("data")

        # Load Individual Customer Profiles
        individual_cols = {
            'Customer_Id': 'string',
            'Age': 'Int64',
            'Gender': 'string',  # Changed from 'category' to handle missing values first
            'Location': 'string',
            'Interests': 'string',
            'Preferences': 'string',
            'Income per year': 'float64',
            'Education': 'string',  # Changed from 'category'
            'Occupation': 'string'
        }
        individuals = pd.read_excel("data/Expanded_Customer_Profile_Individual.xlsx", dtype=individual_cols)

        # Load Organization Profiles
        org_cols = {
            'Customer_Id': 'string',
            'Industry': 'string',  # Changed from 'category'
            'Financial Needs': 'string',
            'Preferences': 'string',
            'Revenue (in dollars)': 'string',
            'No of employees': 'string'
        }
        orgs = pd.read_excel("data/Expanded_Customer_Profile_Organization.xlsx", dtype=org_cols)

        # Load Social Media Data
        social_cols = {
            'Customer_Id': 'string',
            'Post_Id': 'string',
            'Platform': 'string',  # Changed from 'category'
            'Content': 'string',
            'Timestamp': 'string'
        }
        social = pd.read_excel("data/Expanded_Social_Media_Sentiment.xlsx", dtype=social_cols)

        # Load Transaction History
        transaction_cols = {
            'Customer_Id': 'string',
            'Product_Id': 'string',
            'Transaction Type': 'string',  # Changed from 'category'
            'Category': 'string',  # Changed from 'category'
            'Amount (in Dollars)': 'string',
            'Purchase Date': 'string',
            'Payment Mode': 'string'  # Changed from 'category'
        }
        transactions = pd.read_excel("data/Expanded_Transaction_History.xlsx", dtype=transaction_cols)

        # Check if 'Category' exists in transactions
        if 'Category' not in transactions.columns:
            logger.warning("⚠️ 'Category' column missing in transactions! Assigning default category.")
            transactions['Category'] = 'general'
        else:
            # Fill NA values before converting to categorical
            transactions['Category'] = transactions['Category'].fillna('general')

        # Convert purchase dates
        transactions['Purchase_Date'] = pd.to_datetime(
            transactions['Purchase Date'], format='%m/%d/%Y', errors='coerce'
        ).fillna(pd.to_datetime('2025-01-01'))

        # Clean and convert amount
        transactions['Amount'] = (
            transactions['Amount (in Dollars)']
            .astype(str)
            .str.replace(r'[^\d.]', '', regex=True)
            .replace('', np.nan)
            .astype(float)
            .fillna(0)
        )

        # Function to determine preferred category
        def get_preferred_category(x):
            return x.mode()[0] if not x.empty and not x.mode().empty else 'general'

        # Aggregate transactions per customer
        transaction_stats = transactions.groupby('Customer_Id').agg({
            'Amount': ['sum', 'mean', 'count'],
            'Category': get_preferred_category
        })
        transaction_stats.columns = ['amount_sum', 'amount_mean', 'transaction_count', 'preferred_category']

        # Process organization data
        logger.info("Processing organization data...")
        orgs['revenue_numeric'] = orgs['Revenue (in dollars)'].apply(convert_revenue)

        def process_employees(x):
            try:
                if isinstance(x, str):
                    parts = [int(n.strip()) for n in x.split('-') if n.strip().isdigit()]
                    return np.mean(parts) if parts else 0
                return int(x) if pd.notna(x) else 0
            except:
                return 0

        orgs['employee_mid'] = orgs['No of employees'].apply(process_employees)

        # Process sentiment analysis
        logger.info("Processing sentiment analysis...")
        sia = SentimentIntensityAnalyzer()
        social['sentiment'] = social['Content'].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )

        # Merge all datasets
        logger.info("Merging datasets...")
        customers = pd.concat([individuals, orgs], axis=0, ignore_index=True)

        # Ensure consistent Customer_Id type
        customers['Customer_Id'] = customers['Customer_Id'].astype('string')
        social['Customer_Id'] = social['Customer_Id'].astype('string')
        transaction_stats.index = transaction_stats.index.astype('string')

        full_data = (
            customers
            .merge(social.groupby('Customer_Id')['sentiment'].mean().reset_index(), on='Customer_Id', how='left')
            .merge(transaction_stats.reset_index(), on='Customer_Id', how='left')
        )

        # Now convert to categorical after all data is merged and cleaned
        categorical_cols = {
            'Gender': 'Unknown',
            'Education': 'Unknown',
            'Industry': 'Unknown',
            'Platform': 'Unknown',
            'Transaction Type': 'Unknown',
            'Payment Mode': 'Unknown',
            'preferred_category': 'general'
        }

        for col, default_val in categorical_cols.items():
            if col in full_data.columns:
                # First ensure the column exists and has no missing values
                if col not in full_data.columns:
                    full_data[col] = default_val
                else:
                    full_data[col] = full_data[col].fillna(default_val)
                
                # Then convert to categorical
                categories = list(full_data[col].unique())
                full_data[col] = pd.Categorical(full_data[col], categories=categories)

        # Ensure all expected columns exist with proper types
        expected_columns = {
            'Age': ('Int64', 0),
            'Gender': ('category', 'Unknown'),
            'Location': ('string', 'Unknown'),
            'Interests': ('string', ''),
            'Preferences': ('string', ''),
            'Income per year': ('float64', 0),
            'Education': ('category', 'Unknown'),
            'Occupation': ('string', 'Unknown'),
            'Industry': ('category', 'Unknown'),
            'Financial Needs': ('string', ''),
            'revenue_numeric': ('float64', 0),
            'employee_mid': ('float64', 0),
            'sentiment': ('float64', 0),
            'amount_sum': ('float64', 0),
            'amount_mean': ('float64', 0),
            'transaction_count': ('Int64', 0),
            'preferred_category': ('category', 'general')
        }

        for col, (dtype, default) in expected_columns.items():
            if col not in full_data.columns:
                logger.warning(f"Adding missing column: {col} (default={default})")
                full_data[col] = default
            try:
                if dtype == 'category':
                    if not pd.api.types.is_categorical_dtype(full_data[col]):
                        full_data[col] = pd.Categorical(full_data[col].fillna(default))
                else:
                    full_data[col] = full_data[col].fillna(default).astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to {dtype}: {str(e)}")
                full_data[col] = full_data[col].fillna(default)

        if 'preferred_category' not in full_data.columns:
            full_data['preferred_category'] = 'general'
        
        # Convert to categorical with proper handling
        try:
            # Get unique categories while preserving order
            categories = full_data['preferred_category'].astype(str).unique().tolist()
            
            # Convert to categorical type
            full_data['preferred_category'] = pd.Categorical(
                full_data['preferred_category'],
                categories=categories,
                ordered=False
            )
            
            logger.info(f"'preferred_category' converted to category. Categories: {categories}")
        except Exception as e:
            logger.error(f"Failed to convert 'preferred_category': {str(e)}")
            full_data['preferred_category'] = full_data['preferred_category'].astype('category')

        
        full_data['preferred_category'] = pd.Categorical(
            full_data['preferred_category'].astype(str),
            categories=full_data['preferred_category'].unique()
        )
        
        logger.info(f"'preferred_category' dtype after conversion: {full_data['preferred_category'].dtype}")
        logger.info(f"Categories: {full_data['preferred_category'].cat.categories.tolist()}")

        logger.info(f"Data loaded successfully with {len(full_data)} records.")
        logger.info("Sample of loaded data:")
        logger.info(full_data.head(10))

        return full_data

    except Exception as e:
        logger.error(f"Error in load_data: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to load data: {str(e)}") from e