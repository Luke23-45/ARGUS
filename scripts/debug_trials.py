import sqlite3
import pandas as pd
import json

conn = sqlite3.connect('apex_moar_v20.db')
query = """
SELECT t.number, t.state, a.key, a.value_json
FROM trials t
JOIN trial_user_attributes a ON t.trial_id = a.trial_id
WHERE t.number >= 15
"""
df = pd.read_sql_query(query, conn)
conn.close()

if not df.empty:
    df['value'] = df['value_json'].apply(lambda x: json.loads(x) if x else None)
    pivot = df.pivot(index=['number', 'state'], columns='key', values='value').reset_index()
    cols = ['number', 'state', 'mean_auroc', 'robust_utility_captured', 'prune_reason']
    available = [c for c in cols if c in pivot.columns]
    print(pivot[available].to_string())
else:
    print("No data found for trials >= 15")
