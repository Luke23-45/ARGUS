
import torch
import torch.nn.functional as F

print("=== Forensic Analysis: The BCE target=2.0 Breakdown ===")

# Model confidence from 0 to 1
probs = torch.tensor([0.1, 0.5, 0.9, 0.99, 1.0])
logits = torch.logit(probs)

# Standard Sepsis Target (1.0)
target_std = torch.full_like(probs, 1.0)
loss_std = F.binary_cross_entropy_with_logits(logits, target_std, reduction='none')
print(f"\nStandard Target (1.0):")
print(f"Probs: {probs.tolist()}")
print(f"Loss:  {loss_std.tolist()}")
print("Result: Loss is always positive and goes to 0 as confidence increases.")

# Shock Target (2.0) - The Source of the Crash
target_shock = torch.full_like(probs, 2.0)
loss_shock = F.binary_cross_entropy_with_logits(logits, target_shock, reduction='none')
print(f"\nShock Target (2.0):")
print(f"Probs: {probs.tolist()}")
print(f"Loss:  {loss_shock.tolist()}")

print("\n!!! THE DISCOVERY !!!")
for p, l in zip(probs, loss_shock):
    if l < 0:
        print(f"Confidence {p*100:.0f}% -> Loss is NEGATIVE ({l.item():.4f})")
    if torch.isinf(l):
        print(f"Confidence 100% -> Loss is {l.item()} (REWARD HACK)")

print("\n--- Gradient Explosion Proof ---")
# Let's check the gradient of the 'Reward Hack'
x = torch.tensor([5.0], requires_grad=True) # High confidence
t = torch.tensor([2.0])
loss = F.binary_cross_entropy_with_logits(x, t)
loss.backward()
print(f"Logit x=5.0 (Conf=99.3%), Loss={loss.item():.4f}, Gradient={x.grad.item():.4f}")

x_higher = torch.tensor([15.0], requires_grad=True) # Very high confidence
loss_higher = F.binary_cross_entropy_with_logits(x_higher, t)
loss_higher.backward()
print(f"Logit x=15.0 (Conf=99.9%), Loss={loss_higher.item():.4f}, Gradient={x_higher.grad.item():.4f}")

print("\nSUMMARY:")
print("1. When target=2.0, loss for a CORRECT prediction is -inf.")
print("2. The gradient (force) is -1.0 ALWAYS while confidence increases.")
print("3. This infinite negative loss incentivized the model to 'explode' its weights to reach more -inf.")
print("4. This is the mechanical source of the 'NaN/Inf' you saw in the logs.")
