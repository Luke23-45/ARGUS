
import re
import pandas as pd


def parse_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    # Identify lines like "Epoch X: 100%"
    # z1: Epoch 0: 100% 120/120 ...
    # z3: Epoch 0: 100% 200/200 ...
    
    for line in lines:
        if "Epoch" in line and ("100%" in line or "120/120" in line or "200/200" in line):
            # Extract Epoch number
            epoch_match = re.search(r'Epoch (\d+):', line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            
            # Helper to extract a float metric
            def get_metric(name, text):
                # Search for "name=VALUE" or "name: VALUE"
                # Some logs uses "name=value", others might vary.
                # Assuming "name=value" based on file view.
                # Use word boundary to avoid partial matches if needed, but simple search is usually fine.
                m = re.search(rf'{name}=([\d\.\-eE]+)', text)
                if m:
                    return float(m.group(1))
                return None

            auc = get_metric("AUC", line)
            gmse = get_metric("GMSE", line)
            mse_hemo = get_metric("mse_hemo", line)
            mse_labs = get_metric("mse_labs", line)
            mse_electrolytes = get_metric("mse_electrolytes", line)
            ood = get_metric("ood_score", line)
            gn = get_metric("GN", line)
            
            # Only add if we found at least some metrics
            if auc is not None or gmse is not None:
                data.append({
                    "Epoch": epoch,
                    "AUC": auc,
                    "GMSE": gmse,
                    "Hemo MSE": mse_hemo,
                    "Labs MSE": mse_labs,
                    "Electrolytes MSE": mse_electrolytes,
                    "OOD Score": ood,
                    "GN": gn
                })
    return data


z1_path = r"C:\Users\Hellx\Documents\Programming\python\Project\iron\icu_research\logs\z1\rt\z1.logs"
z3_path = r"C:\Users\Hellx\Documents\Programming\python\Project\iron\icu_research\logs\z1\rt\z3.md"

z1_data = parse_log_file(z1_path)
z3_data = parse_log_file(z3_path)

print("Z1 Data:")
for d in z1_data:
    print(d)

print("\nZ3 Data:")
for d in z3_data:
    print(d)
