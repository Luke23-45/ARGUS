import sqlite3
import json

def get_best_auc(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Study names for different versions if they exist
    potential_keys = ['auc', 'sepsis_auroc', 'mean_auroc', 'sepsis_AUC']
    
    print(f"{'Trial #':<10} | {'Metric':<15} | {'Value':<10}")
    print("-" * 40)
    
    query = """
    SELECT t.number, a.key, a.value_json
    FROM trials t
    JOIN trial_user_attributes a ON t.trial_id = a.trial_id
    WHERE a.key IN ({})
    """.format(','.join(['?'] * len(potential_keys)))
    
    cur.execute(query, potential_keys)
    rows = cur.fetchall()
    
    results = []
    for num, key, val_json in rows:
        try:
            val = json.loads(val_json)
            if isinstance(val, (int, float)):
                results.append((num, key, val))
        except:
            continue
            
    if not results:
        print("No AUC metrics found in database.")
        return

    # Sort and display top 10
    results.sort(key=lambda x: x[2], reverse=True)
    
    for num, key, val in results[:10]:
        print(f"{num:<10} | {key:<15} | {val:.4f}")
        
    print("\n" + "="*40)
    best = results[0]
    print(f"🥇 BEST OVERALL AUC: {best[2]:.4f} (Trial {best[0]}, Metric: {best[1]})")
    print("="*40)
    
    conn.close()

if __name__ == "__main__":
    get_best_auc("apex_moar_v20.db")
