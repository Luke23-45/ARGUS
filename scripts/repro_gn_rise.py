
import math

class MockTrainer:
    def __init__(self, total_steps):
        self.estimated_stepping_batches = total_steps

class MockWrapper:
    def __init__(self, base_phys_weight=0.2, total_steps=50000):
        self.trainer = MockTrainer(total_steps)
        self.global_step = 0
        self.base_phys_weight = base_phys_weight

    def _get_curr_physics_weight(self) -> float:
        total_steps = getattr(self.trainer, "estimated_stepping_batches", 50000)
        warmup_steps = total_steps * 0.5
        
        current_step = float(self.global_step)
        
        if current_step < warmup_steps:
            progress = current_step / float(max(1, warmup_steps))
            return 0.01 + (self.base_phys_weight - 0.01) * progress
        else:
            return self.base_phys_weight

def main():
    wrapper = MockWrapper(base_phys_weight=0.2, total_steps=50000)
    
    print(f"{'Step':<10} | {'Phys Weight':<15} | {'Gradient Scale Factor':<20}")
    print("-" * 50)
    
    # Simulate steps
    steps_to_check = [0, 1000, 5000, 12500, 25000, 25001, 50000]
    
    base_grad_norm = 1.0 # Assume constant physics error leads to unit gradient
    
    for step in range(50001):
        wrapper.global_step = step
        weight = wrapper._get_curr_physics_weight()
        
        if step in steps_to_check:
            # Gradient contribution is proportional to weight
            # Factor = Current Weight / Initial Weight
            factor = weight / 0.01
            print(f"{step:<10} | {weight:.5f}         | {factor:.1f}x")

    print("\nCONCLUSION:")
    print("The Physics Weight increases from 0.01 to 0.20 over the first 25,000 steps.")
    print("This causes the physics-related Gradient Norm to rise by a factor of 20x,")
    print("even if the model's physics error remains constant.")

if __name__ == "__main__":
    main()
