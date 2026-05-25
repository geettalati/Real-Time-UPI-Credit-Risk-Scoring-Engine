import pandas as pd
import numpy as np
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from scipy.stats import entropy

async def process_features():
    print("Connecting to MongoDB...")
    # MongoDB connection via motor
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client['credit_risk']
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    raw_csv = os.path.join(base_dir, 'raw', 'transactions.csv')
    processed_csv = os.path.join(base_dir, 'processed', 'features.csv')
    os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
    
    print("Reading raw transactions from CSV...")
    df = pd.read_csv(raw_csv)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Optionally insert raw data to MongoDB if we want to use MongoDB throughout
    # Doing a fast bulk insert might take a moment for ~1M rows. To keep the pipeline efficient,
    # we'll compute the features using Pandas, then store the rich feature representations in MongoDB.
    # If the user wants the raw transactions in MongoDB too, we can insert them. 
    # Let's drop existing and insert the raw transactions to satisfy the "MongoDB throughout" constraint.
    print("Loading raw data into MongoDB (this might take a moment)...")
    await db.raw_transactions.drop()
    # Insert in chunks
    records = df.to_dict('records')
    chunk_size = 50000
    for i in range(0, len(records), chunk_size):
        await db.raw_transactions.insert_many(records[i:i+chunk_size])
    print("Raw data loaded to MongoDB.")

    # Now let's calculate features in pandas since doing entropy/std in Mongo aggregates is very complex.
    # We could also read back from Mongo, but we already have the dataframe in memory.
    
    print("Engineering features...")
    # Helper to calculate entropy
    def calc_entropy(series):
        counts = series.value_counts()
        return entropy(counts) if len(counts) > 0 else 0

    features = []
    
    # Group by user
    grouped = df.groupby('user_id')
    
    # Identify the latest date in the dataset to calculate "days_since_last_income"
    max_global_date = df['timestamp'].max()
    
    for user_id, group in grouped:
        # Filter credits & debits
        credits = group[group['transaction_type'] == 'credit']
        debits = group[group['transaction_type'] == 'debit']
        salary_credits = credits[credits['merchant_category'] == 'salary']
        
        # 1. avg_monthly_income (assuming 6 months of data)
        total_salary = salary_credits['amount'].sum()
        avg_monthly_income = total_salary / 6.0
        
        # 2. income_regularity_score (std dev of salary credits)
        income_regularity_score = salary_credits['amount'].std() if len(salary_credits) > 1 else 0
        if pd.isna(income_regularity_score):
            income_regularity_score = 0
            
        # 3. avg_daily_spend
        total_spend = debits['amount'].sum()
        avg_daily_spend = total_spend / 180.0
        
        # 4. spend_to_income_ratio
        spend_to_income_ratio = (total_spend / total_salary) if total_salary > 0 else (total_spend / 1.0)
        
        # 5. p2p_outflow_ratio
        p2p_debits = debits[debits['merchant_category'] == 'p2p']['amount'].sum()
        p2p_outflow_ratio = (p2p_debits / total_spend) if total_spend > 0 else 0
        
        # 6. merchant_diversity_score (entropy of merchant categories)
        merchant_diversity_score = calc_entropy(debits['merchant_category'])
        
        # 7. min_balance_30d
        cutoff_date = max_global_date - pd.Timedelta(days=30)
        last_30d = group[group['timestamp'] >= cutoff_date]
        min_balance_30d = last_30d['balance_after'].min() if not last_30d.empty else group['balance_after'].iloc[-1]
        
        # 8. balance_volatility
        balance_volatility = group['balance_after'].std() if len(group) > 1 else 0
        if pd.isna(balance_volatility):
            balance_volatility = 0
            
        # 9. days_since_last_income
        if not salary_credits.empty:
            last_income_date = salary_credits['timestamp'].max()
            days_since_last_income = (max_global_date - last_income_date).days
        else:
            days_since_last_income = 180
            
        features.append({
            'user_id': int(user_id),
            'avg_monthly_income': float(avg_monthly_income),
            'income_regularity_score': float(income_regularity_score),
            'avg_daily_spend': float(avg_daily_spend),
            'spend_to_income_ratio': float(spend_to_income_ratio),
            'p2p_outflow_ratio': float(p2p_outflow_ratio),
            'merchant_diversity_score': float(merchant_diversity_score),
            'min_balance_30d': float(min_balance_30d),
            'balance_volatility': float(balance_volatility),
            'days_since_last_income': int(days_since_last_income)
        })
        
    features_df = pd.DataFrame(features)
    
    print("Writing features to MongoDB...")
    await db.user_features.drop()
    feature_records = features_df.to_dict('records')
    # Insert chunks
    for i in range(0, len(feature_records), chunk_size):
        await db.user_features.insert_many(feature_records[i:i+chunk_size])
        
    print(f"Writing features to {processed_csv}...")
    features_df.to_csv(processed_csv, index=False)
    
    print("Feature engineering pipeline completed successfully!")

if __name__ == "__main__":
    asyncio.run(process_features())
