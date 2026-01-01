"""
working/cagrad.py
-----------------
SOTA CAGrad Implementation (Pure PyTorch, GPU-friendly).

Official from Cranial-XIX/CAGrad (NeurIPS 2021), integrated in LibMTL benchmarks.
"""

import torch

class CAGrad(torch.optim.Optimizer):
    """
    Conflict-Averse Gradient Descent Wrapper.
    Usage: optimizer = CAGrad(torch.optim.Adam(model.parameters()), c=0.5)
           losses = [diff_loss, aux_loss, critic_loss]
           optimizer.pc_backward(losses)  # Custom method
           optimizer.step()
    """
    def __init__(self, optimizer, c: float = 0.5):
        self.optimizer = optimizer
        self.c = c
        # [FIX] Initialize parent Optimizer with inner optimizer's params and defaults
        # This satisfies isinstance(optimizer, torch.optim.Optimizer) check in LRScheduler
        super().__init__(optimizer.param_groups, optimizer.defaults)
        
        # [CRITICAL] Swap internal state to match inner optimizer exactly
        # This overrides the shallow copies made by super().__init__
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, closure=None):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()
        
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _get_flat_grad(self):
        """Helper to flatten and concatenate gradients."""
        views = []
        for group in self.param_groups: # [FIX] Use self.param_groups (which is linked to inner)
            for p in group['params']:
                if p.grad is None:
                    view = p.data.new(p.data.numel()).fill_(0)
                else:
                    view = p.grad.data.view(-1)
                views.append(view)
        return torch.cat(views, 0)

    def pc_backward(self, losses, backward_fn=None, accumulate=False):
        # 1. Save current accumulated gradients if we are in an accumulation window
        current_grads = None
        if accumulate:
            current_grads = self._get_flat_grad().clone()

        # 2. Capture Task Gradients
        task_grads = []
        for loss in losses:
            self.optimizer.zero_grad() # Safe now because we saved 'current_grads'
            if backward_fn:
                backward_fn(loss, retain_graph=True)
            else:
                loss.backward(retain_graph=True)
            task_grads.append(self._get_flat_grad().clone())

        # 3. Perform Surgery (Math remains same)
        g = torch.stack(task_grads)
        g_avg = g.mean(dim=0)
        GG = g @ g.t()
        try:
            # [FIX] torch.linalg.solve requires float32 for stability and compatibility
            # Especially critical for mixed-precision/bfloat16 training.
            alpha = torch.linalg.solve(
                (GG + 1e-6 * torch.eye(len(losses), device=GG.device)).float(), 
                torch.ones(len(losses), device=GG.device, dtype=torch.float32)
            )
            alpha = torch.clamp(alpha, min=0) 
            alpha = alpha / (alpha.sum() + 1e-8)
            # Cast back to input dtype for consistent gradient surgery
            alpha = alpha.to(g.dtype) 
        except:
            alpha = torch.ones(len(losses), device=GG.device, dtype=g.dtype) / len(losses)

        final_grad = (alpha @ g)
        if torch.norm(final_grad) > 1e-8:
            final_grad = final_grad * torch.norm(g_avg) / torch.norm(final_grad)
        final_grad = (1 - self.c) * final_grad + self.c * g_avg

        # 4. Apply and Restore
        if accumulate and current_grads is not None:
            # Add new surgical grad to existing accumulated grad
            self._set_flat_grad(current_grads + final_grad)
        else:
            self._set_flat_grad(final_grad)

    def _set_flat_grad(self, grad_vec):
        """
        Restores gradients from a flat vector. 
        SOTA 2025 FIX: Handles None gradients caused by zero_grad() 
        cycles during Multi-Task surgery.
        """
        idx = 0
        for group in self.param_groups: # [FIX] Use self.param_groups
            for p in group['params']:
                if p.requires_grad:
                    numel = p.numel()
                    # If p.grad was set to None by zero_grad(), we must re-initialize it
                    # to hold the surgical gradient result.
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    
                    # Copy the surgical result into the grad buffer
                    p.grad.data.copy_(grad_vec[idx:idx + numel].view_as(p.grad))
                    idx += numel
