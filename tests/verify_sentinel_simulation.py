
import torch
import math
import numpy as np

# ==============================================================================
# COPY OF IMPLEMENTED LOGIC (stabilization.py)
# ==============================================================================
class TrendSentinel:
    @staticmethod
    def is_unstable(ema: float, std: float, max_pressure: float = 5.0, max_sigma: float = 2.0) -> bool:
        """
        [SOTA v2026] Hybrid Sentinel: Detects both Drift (Boiling Frog) and Shock.
        1. Pressure Check: Is the manifold under absolute physiological stress? (EMA > 5.0)
        2. Volatility Check: Is the manifold vibrating uncontrollably? (STD > 2.0)
        """
        # [Guard 1] Absolute Pressure (The "Boiling Frog" Detector)
        if ema > max_pressure:
            return True
            
        # [Guard 2] Volatility Shock (The "Earthquake" Detector)
        if std > max_sigma:
            return True
            
        return False

    @staticmethod
    def calculate_z_score(current_val: float, ema: torch.Tensor, std: torch.Tensor) -> float:
        """Computes the standard deviation distance from the moving baseline."""
        diff = abs(current_val - ema.item())
        # Use a floor for std to prevent division by zero in stable regimes
        safe_std = max(std.item(), 0.05) 
        return diff / safe_std
        
    @staticmethod
    def update_stats(current_val: float, ema: torch.Tensor, std: torch.Tensor, decay: float):
        """Standard EMA update for mean and variance (Welford-style approximation)."""
        # Check for NaN/Inf
        if not math.isfinite(current_val):
            return

        with torch.no_grad():
            delta = current_val - ema.item()
            # Update Mean
            ema.mul_(decay).add_(current_val, alpha=1.0 - decay)
            # Update Variance (EMA of squared differences)
            # Var_new = decay * Var_old + (1-decay) * (delta * new_delta)
            new_delta = current_val - ema.item()
            sq_diff = delta * new_delta
            
            # We store STD directly for easier Z-score calculation
            var = (std.item() ** 2)
            new_var = decay * var + (1.0 - decay) * sq_diff
            std.fill_(math.sqrt(max(new_var, 1e-6)))

# ==============================================================================
# SIMULATION ENGINE
# ==============================================================================

def run_simulation(scenario_name, input_data, decay=0.99):
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # Initialize State
    ema = torch.tensor(input_data[0]) 
    std = torch.tensor(0.5) 
    
    triggered_at = -1
    trigger_reason = None
    
    history_ema = []
    history_std = []
    
    for t, val in enumerate(input_data):
        # 1. Update Stats
        TrendSentinel.update_stats(val, ema, std, decay)
        
        # 2. Check Sentinel
        is_shock = TrendSentinel.is_unstable(ema.item(), std.item(), max_pressure=5.0, max_sigma=2.0)
        
        # 3. Log
        history_ema.append(ema.item())
        history_std.append(std.item())
        
        if is_shock and triggered_at == -1:
            triggered_at = t
            reason = []
            if ema.item() > 5.0: reason.append(f"Pressure({ema.item():.2f}>5.0)")
            if std.item() > 2.0: reason.append(f"Volatility({std.item():.2f}>2.0)")
            trigger_reason = " + ".join(reason)
            # We continue simulation to see if it recovers, but mark first trigger
            
    print(f"Results for {scenario_name}:")
    if triggered_at != -1:
        print(f"  [ALARM] Triggered at step {triggered_at}")
        print(f"  [CAUSE] {trigger_reason}")
    else:
        print(f"  [OK] System remained stable.")
        print(f"  Max EMA: {max(history_ema):.2f}")
        print(f"  Max STD: {max(history_std):.2f}")

    return history_ema, history_std

# ==============================================================================
# SCENARIOS
# ==============================================================================

def main():
    steps = 1000
    
    # 1. Boiling Frog (Slow Drift)
    # Ramps from 1.0 to 6.0 slowly. The old Z-score would miss this (low deviation).
    # The New Hybrid Sentinel should catch it when EMA > 5.0.
    data_frog = [1.0 + (5.5 * i / steps) for i in range(steps)] # Ends at 6.5
    run_simulation("Boiling Frog (Drift)", data_frog)
    
    # 2. Earthquake (Shock)
    # Stable at 2.0, then instant spike to 20.0, then back.
    data_quake = [2.0] * 300 + [20.0] * 10 + [2.0] * 690
    run_simulation("Earthquake (Shock)", data_quake)
    
    # 3. Stable (White Noise)
    # Should NOT trigger. EMA ~ 2.0, Noise +/- 0.5
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, steps)
    data_stable = [2.0 + n for n in noise]
    run_simulation("Stable Regime", data_stable)
    
    # 4. Resumption Shock (Trauma)
    # Starts Wild (Noise +/- 5.0), then settles.
    # Sentinel usually mutes this with Grace Period (external logic),
    # but let's see if the raw Sentinel flags it (it SHOULD, confirming why we need grace period).
    noise_pulse = [np.random.normal(0, 5.0 * ((100-i)/100)) if i < 100 else np.random.normal(0, 0.5) for i in range(steps)]
    data_resumption = [3.0 + abs(n) for n in noise_pulse] # Gradients are magnitudes (abs)
    run_simulation("Resumption Trauma", data_resumption)

if __name__ == "__main__":
    main()
