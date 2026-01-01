import sqlite3
import pandas as pd
import json

def check_db(db_path):
    conn = sqlite3.connect(db_path)
    
    # Check trial states
    states = pd.read_sql_query("SELECT state, count(*) as count FROM trials GROUP BY state", conn)
    print("Trial States:")
    print(states)
    print("\n")
    
    # Detailed check for Trial 1 (as an example)
    print("--- Detailed Attributes for Trial 1 ---")
    trial_1_attrs = pd.read_sql_query("""
        SELECT key, value_json 
        FROM trial_user_attributes 
        WHERE trial_id = (SELECT trial_id FROM trials WHERE number = 1)
    """, conn)
    for _, row in trial_1_attrs.iterrows():
        print(f"{row['key']}: {row['value_json']}")
    print("\n")

    # Check for any TrialPruned events in the journal if available
    # Since I don't know where the journal is, I'll just check attributes for ALL trials
    print("--- Metrics for All Trials ---")
    attrs = pd.read_sql_query("SELECT trial_id, key, value_json FROM trial_user_attributes", conn)
    if not attrs.empty:
        attrs['value'] = attrs['value_json'].apply(lambda x: json.loads(x) if x else None)
        pivot_attrs = attrs.pivot(index='trial_id', columns='key', values='value')
        
        cols = ['mean_auroc', 'mean_auprc', 'mean_safety', 'robust_utility_captured', 'noble_grace', 'prune_reason']
        available = [c for c in cols if c in pivot_attrs.columns]
        print(pivot_attrs[available])
    
    conn.close()

if __name__ == "__main__":
    check_db("apex_moar_v20.db")
