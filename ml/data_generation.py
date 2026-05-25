import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_users=10000, days=180):
    np.random.seed(42)
    start_date = pd.Timestamp(datetime.today() - timedelta(days=days))
    
    # 60% low-risk (label 0), 40% high-risk (label 1)
    num_low_risk = int(num_users * 0.6)
    num_high_risk = num_users - num_low_risk
    
    user_ids = np.arange(1, num_users + 1)
    risk_labels = np.concatenate([np.zeros(num_low_risk, dtype=int), np.ones(num_high_risk, dtype=int)])
    
    # Shuffle users
    np.random.shuffle(user_ids)
    
    # Create Labels DataFrame
    labels_df = pd.DataFrame({'user_id': user_ids, 'label': risk_labels})
    
    # File Paths
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    os.makedirs(out_dir, exist_ok=True)
    tx_file = os.path.join(out_dir, 'transactions.csv')
    labels_file = os.path.join(out_dir, 'labels.csv')
    
    # Save Labels
    labels_df.to_csv(labels_file, index=False)
    
    # Initialize Transactions CSV with Headers
    with open(tx_file, 'w') as f:
        f.write('user_id,timestamp,amount,merchant_category,transaction_type,balance_after\n')
    
    print(f"Generating synthetic data for {num_users} users over {days} days...")
    
    batch_size = 1000
    for batch_start in range(0, num_users, batch_size):
        batch_users = user_ids[batch_start:batch_start+batch_size]
        batch_labels = risk_labels[batch_start:batch_start+batch_size]
        
        batch_transactions = []
        
        for user_id, is_risky in zip(batch_users, batch_labels):
            # Initial balance
            balance = np.random.uniform(5000, 50000) if not is_risky else np.random.uniform(100, 5000)
            
            # Generate timestamps once per user
            dates = pd.date_range(start_date, periods=days, freq='D')
            
            # Markov chain simple simulation over days
            if not is_risky:
                # LOW RISK (Salary Users)
                for d in dates:
                    # Salary Day Logic (1st or near end of month)
                    is_salary_day = d.day == 1 or d.day >= 28
                    if is_salary_day and np.random.rand() < 0.2:
                        amt = np.random.uniform(30000, 100000)
                        balance += amt
                        batch_transactions.append((user_id, d, amt, 'salary', 'credit', balance))
                    
                    # Regular spending
                    num_tx = np.random.randint(0, 3) # 0 to 2 transactions a day
                    if num_tx > 0:
                        cats = np.random.choice(['groceries', 'fuel', 'rent', 'entertainment', 'p2p'], size=num_tx, p=[0.4, 0.2, 0.1, 0.2, 0.1])
                        for cat in cats:
                            amt = np.random.uniform(10000, 25000) if cat == 'rent' else np.random.uniform(100, 2000)
                            if balance >= amt:
                                balance -= amt
                                batch_transactions.append((user_id, d, amt, cat, 'debit', balance))
            else:
                # HIGH RISK (Risky Users)
                for d in dates:
                    # Irregular Income
                    if np.random.rand() < 0.05:
                        amt = np.random.uniform(1000, 10000)
                        balance += amt
                        batch_transactions.append((user_id, d, amt, 'salary', 'credit', balance))
                    
                    # Frequent P2P outflows and high spending relative to balance
                    num_tx = np.random.randint(1, 4) # 1 to 3 transactions a day
                    cats = np.random.choice(['groceries', 'fuel', 'rent', 'entertainment', 'p2p'], size=num_tx, p=[0.1, 0.1, 0.05, 0.15, 0.6])
                    for cat in cats:
                        amt = np.random.uniform(500, 5000)
                        if balance > 0:
                            actual_amt = min(amt, balance) if np.random.rand() < 0.5 else amt
                            balance -= actual_amt
                            if balance < 0: balance = 0
                            batch_transactions.append((user_id, d, actual_amt, cat, 'debit', balance))
                            
        # Append batch to CSV
        df_batch = pd.DataFrame(batch_transactions, columns=['user_id', 'timestamp', 'amount', 'merchant_category', 'transaction_type', 'balance_after'])
        df_batch.to_csv(tx_file, mode='a', header=False, index=False)
        print(f"Processed {batch_start + len(batch_users)}/{num_users} users...")
        
    print(f"Data generation complete! Saved to {tx_file} and {labels_file}")

def generate_report():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
    tx_file = os.path.join(out_dir, 'transactions.csv')
    labels_file = os.path.join(out_dir, 'labels.csv')
    
    print("\n" + "="*50)
    print("DATASET STATISTICS REPORT")
    print("="*50)
    
    labels_df = pd.read_csv(labels_file)
    print("\n--- User Statistics ---")
    print(f"Total Users: {len(labels_df)}")
    print(labels_df['label'].value_counts().rename(index={0: 'Low-Risk (0)', 1: 'High-Risk (1)'}))
    
    print("\n--- Loading Transactions for Report ---")
    tx_df = pd.read_csv(tx_file)
    print(f"Total Transactions: {len(tx_df):,}")
    
    print("\n--- Transaction Type Distribution ---")
    print((tx_df['transaction_type'].value_counts(normalize=True) * 100).round(2).astype(str) + '%')
    
    print("\n--- Merchant Category Distribution ---")
    print(tx_df['merchant_category'].value_counts())
    
    print("\n--- Average Transaction Amount by Type ---")
    print(tx_df.groupby('transaction_type')['amount'].mean().round(2))
    
    print("\n--- Average Final Balance by User Risk Level ---")
    # Get last transaction for each user
    last_tx = tx_df.drop_duplicates(subset='user_id', keep='last')
    merged = last_tx.merge(labels_df, on='user_id')
    
    avg_balances = merged.groupby('label')['balance_after'].mean().rename(index={0: 'Low-Risk (0)', 1: 'High-Risk (1)'}).round(2)
    print(avg_balances)
    
    print("="*50 + "\n")

if __name__ == "__main__":
    generate_synthetic_data()
    generate_report()
