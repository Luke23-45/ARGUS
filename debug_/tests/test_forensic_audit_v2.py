
import os
import torch
import torch.nn as nn
from icu.core.gradnorm import GradNormBalancer
from icu.core.cagrad import CAGrad
from icu.models.components.contrastive_loss import AsymmetricContrastiveLoss
import unittest

class ForensicAudit(unittest.TestCase):
    def test_smoking_gun_235_gradnorm_detach(self):
        """
        [SG #235] Verifies if GradNorm correctly computes gradients through losses.
        The current implementation detaches losses before grad computation.
        """
        num_tasks = 2
        p = nn.Parameter(torch.ones(10, requires_grad=True))
        shared_params = [p]
        
        balancer = GradNormBalancer(num_tasks=num_tasks, shared_params=shared_params)
        
        # Mock losses that depend on p
        loss1 = (p ** 2).sum()
        loss2 = (p * 3).sum()
        losses = torch.stack([loss1, loss2])
        
        # Force initial losses to ignore warmup
        balancer.initial_losses.data.copy_(losses.detach())
        
        gn_loss, weights = balancer.update(losses)
        
        # If it works, gn_loss should have a grad_fn linked to balancer.weights
        print(f"GradNorm Loss: {gn_loss.item()}")
        self.assertIsNotNone(gn_loss.grad_fn, "GradNorm Loss has NO grad_fn! (SG #235 Detach Poison)")
        
        # Check if it actually computed non-trivial norms
        # Norm of grad(loss1) = norm(2p) = norm([2,2...]) = sqrt(10 * 4) = 6.32
        # We check internal norm_emas (buffered)
        print(f"Norm 0: {balancer.norm_emas[0]:.4f}, Norm 1: {balancer.norm_emas[1]:.4f}")
        
        if balancer.norm_emas[0] < 1e-4:
            self.fail("FAILURE: GradNorm is blind to gradients! (SG #235)")

    def test_smoking_gun_226_cagrad_drift(self):
        """
        [SG #226] Verifies if CAGrad correctly handles frozen parameters without indexing drift.
        """
        p1 = nn.Parameter(torch.ones(2, requires_grad=True))
        p2 = nn.Parameter(torch.ones(2, requires_grad=False)) # Frozen
        p3 = nn.Parameter(torch.ones(2, requires_grad=True))
        
        base_opt = torch.optim.SGD([p1, p2, p3], lr=0.1)
        optimizer = CAGrad(base_opt)
        
        # Loss only depends on p3
        loss = (p3 * 5).sum()
        optimizer.pc_backward([loss])
        
        # If fix works, idx correctly skips frozen p2
        # final_grad[2:4] now corresponds to p3
        print(f"P1 Grad: {p1.grad}")
        print(f"P3 Grad: {p3.grad}")
        
        # Check P1 (should be 0 because loss was on p3)
        self.assertTrue(torch.allclose(p1.grad, torch.zeros_like(p1)), f"P1 Grad poisoned: {p1.grad}")
        
        # Check P3 (should be [5, 5])
        self.assertIsNotNone(p3.grad, "P3 Grad is None!")
        self.assertTrue(torch.allclose(p3.grad, torch.tensor([5.0, 5.0])), f"P3 Grad shifted or zeroed: {p3.grad}")
        print("SUCCESS: CAGrad is robust to frozen parameters.")

    def test_smoking_gun_220_acl_deadlock(self):
        """
        [SG #220] Verifies if AsymmetricContrastiveLoss returns early on empty ranks.
        If it returns early, it will miss the DDP all_reduce in a cluster.
        """
        acl = AsymmetricContrastiveLoss(d_model=16, num_classes=3)
        
        # Rank X has zero samples
        z_empty = torch.zeros(0, 16, requires_grad=True)
        y_empty = torch.zeros(0, dtype=torch.long)
        
        # We can't easily mock DDP here without a real cluster, but we can check if it returns early.
        # We'll monkeypatch all_reduce to see if it's called.
        all_reduce_called = False
        def mock_all_reduce(tensor, op=None):
            nonlocal all_reduce_called
            all_reduce_called = True
            
        import torch.distributed as dist
        original_dist = None
        if hasattr(dist, 'all_reduce'):
            original_dist = dist.all_reduce
            dist.all_reduce = mock_all_reduce
            
        # Mock ddp init
        def mock_is_initialized(): return True
        orig_is_init = dist.is_initialized
        dist.is_initialized = mock_is_initialized
        
        try:
            acl.train()
            # This should call all_reduce if robust, or return early if buggy
            _ = acl(z_empty, y_empty)
            
            if not all_reduce_called:
                print("FAILURE: ACL Deadlock Vulnerability! Empty rank returned early. (SG #220)")
            else:
                print("SUCCESS: ACL is robust to empty ranks.")
        finally:
            if original_dist: dist.all_reduce = original_dist
            dist.is_initialized = orig_is_init

if __name__ == "__main__":
    unittest.main()
