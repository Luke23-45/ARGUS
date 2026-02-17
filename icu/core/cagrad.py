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
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    if p.grad is None:
                        # [v23.0 SOTA FIX] Safe initialization for missing grads
                        view = p.data.new(p.data.numel()).fill_(0)
                    else:
                        view = p.grad.data.view(-1)
                    views.append(view)
        return torch.cat(views, 0)

    def pc_backward(self, losses, backward_fn=None, accumulate=False):
        """
        Conflict-Averse Surgery with DDP Logic.
        [v2026 SOTA FIX] Unified Surgical Consensus (Smoking Gun #Divergence)
        """
        # 1. Save current accumulated gradients
        # Rationale: Captures background accumulation (Diffusion, AGEM) to prevent surgery wipe.
        current_grads = None
        if accumulate:
            current_grads = self._get_flat_grad()

        # 2. Capture Task Gradients
        task_grads = []
        is_dist = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_dist else 1
        
        for loss in losses:
            self.optimizer.zero_grad() 
            if backward_fn:
                backward_fn(loss, retain_graph=True)
            else:
                loss.backward(retain_graph=True)
                
            # [v2026 SOTA FIX] Finite Guard (Smoking Gun #NaN)
            # Rationale: Poisoned tasks must be zeroed to satisfy collective surgery.
            g_i = self._get_flat_grad()
            if not torch.isfinite(g_i).all():
                g_i.zero_()
            
            # [v2026 SOTA FIX] Global Task Consensus (Smoking Gun #DDP-Surgery)
            # Rationale: In DDP, surgery MUST be performed on the GLOBAL task gradients.
            # Otherwise, each rank performs 'local' surgery, breaking parity.
            if is_dist:
                torch.distributed.all_reduce(g_i, op=torch.distributed.ReduceOp.SUM)
                g_i.div_(world_size)
                
            task_grads.append(g_i)

        # 3. Perform Surgery
        g = torch.stack(task_grads)
        g_avg = g.mean(dim=0)
        
        # [v2026 SOTA] Deterministic Surgery
        # Rationale: Since g_i are now global, GG is naturally global. 
        GG = g @ g.t()

        try:
            # [FIX] torch.linalg.solve requires float32 for stability
            alpha = torch.linalg.solve(
                (GG + 1e-6 * torch.eye(len(losses), device=GG.device)).float(), 
                torch.ones(len(losses), device=GG.device, dtype=torch.float32)
            )
            alpha = torch.clamp(alpha, min=0) 
            alpha = alpha / (alpha.sum() + 1e-8)
            alpha = alpha.to(g.dtype) 
        except:
            alpha = torch.ones(len(losses), device=GG.device, dtype=g.dtype) / len(losses)

        # Calculate surgical direction
        final_grad = (alpha @ g)
        
        # [v2026 SOTA FIX] Branchless Renormalization (Zero-Sync)
        f_norm = torch.norm(final_grad)
        g_avg_norm = torch.norm(g_avg)
        
        scaler = torch.where(
            f_norm > 1e-8,
            g_avg_norm / (f_norm + 1e-8), 
            torch.tensor([1.0], device=f_norm.device, dtype=f_norm.dtype)
        )
        final_grad = final_grad * scaler
        final_grad = (1 - self.c) * final_grad + self.c * g_avg

        # 4. Apply and Restore
        if accumulate and current_grads is not None:
            # [v2026 SOTA FIX] Additive Surgery (The Accumulation Recovery)
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
