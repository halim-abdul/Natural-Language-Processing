import math
from typing import Callable, Iterable, Tuple
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-6,
                 weight_decay: float = 0.0, correct_bias: bool = True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter")
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, correct_bias=correct_bias))

    def step(self, closure: Callable = None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                if group["correct_bias"]:
                    step_size = alpha * math.sqrt(1-beta2**t) / (1-beta1**t)
                else:
                    step_size = alpha
                p.data.addcdiv_(m, v.sqrt().add(eps), value=-step_size)
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-alpha * group["weight_decay"])
        return loss
