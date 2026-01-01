import sqlite3
import json

def audit_all_auc_keys(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    print("--- AUC KEY AUDIT ---")
    query = """
    SELECT t.number, t.state, a.key, a.value_json
    FROM trials t
    JOIN trial_user_attributes a ON t.trial_id = a.trial_id
    WHERE a.key LIKE '%auc%' OR a.key LIKE '%AUC%'
    ORDER BY CAST(a.value_json AS FLOAT) DESC
    """
    
    try:
        cur.execute(query)
        rows = cur.fetchall()
        
        print(f"{'Trial #':<8} | {'State':<10} | {'Key':<20} | {'Value':<10}")
        print("-" * 60)
        
        for num, state, key, val_json in rows:
            try:
                val = json.loads(val_json)
                print(f"{num:<8} | {state:<10} | {key:<20} | {val:.4f}")
            except:
                continue
    except Exception as e:
        print(f"Error querying: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    audit_all_auc_keys("apex_moar_v20.db")
