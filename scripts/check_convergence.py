import sqlite3
import json
import pandas as pd

def check_convergence(db_path):
    # Use uri=True for read-only mode
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    except:
        conn = sqlite3.connect(db_path)
    
    # Extract params for all COMPLETE trials
    query = """
    SELECT t.number, p.param_name, p.param_value 
    FROM trials t 
    JOIN trial_params p ON t.trial_id = p.trial_id 
    WHERE t.state = 'COMPLETE'
    """
    df_params = pd.read_sql_query(query, conn)
    
    # Extract AUC for all COMPLETE trials
    query = """
    SELECT t.number, a.value_json as auroc 
    FROM trials t 
    JOIN trial_user_attributes a ON t.trial_id = a.trial_id 
    WHERE a.key = 'mean_auroc'
    """
    df_auc = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if df_params.empty or df_auc.empty:
        print("No data found.")
        return

    # Parse JSON (AUC is still JSON-encoded in user_attrs)
    # df_params['param_value'] is already a float
    df_auc['auroc'] = df_auc['auroc'].apply(json.loads)
    
    # Pivot and join
    pivot = df_params.pivot(index='number', columns='param_name', values='param_value')
    final = pivot.join(df_auc.set_index('number'), how='inner')
    
    # Sort and print
    sorted_df = final.sort_values('auroc', ascending=False)
    print("Top 10 Trials:")
    print(sorted_df.head(10))
    
    print("\nParameter Ranges (Top 10):")
    top_10 = sorted_df.head(10)
    for col in pivot.columns:
        print(f"{col}: {top_10[col].min():.4f} to {top_10[col].max():.4f}")

if __name__ == "__main__":
    check_convergence("apex_moar_v20.db")
