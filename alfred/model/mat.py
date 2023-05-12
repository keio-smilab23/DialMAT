import torch
import torch.nn as nn


class moment_based_adversarial_perturbation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, m, v, step, rho1, rho2, lr):
        ctx.delta, ctx.m, ctx.v, ctx.rho1, ctx.rho2, ctx.step, ctx.lr = delta, m, v, rho1, rho2, step, lr
        return delta

    @staticmethod
    def backward(ctx, grad_delta):
        new_m = ctx.rho1 * ctx.m + (1 - ctx.rho1) * grad_delta
        new_v = ctx.rho2 * ctx.v + (1 - ctx.rho2) * grad_delta**2
        m_hat = new_m / (1 - ctx.rho1**ctx.step)
        v_hat = new_v / (1 - ctx.rho2**ctx.step)

        d_delta = ctx.lr * m_hat / torch.clamp(v_hat, min=1e-8) ** 0.5

        d_delta_norm = torch.norm(d_delta.view(d_delta.size(0), -1), dim=1).view(-1, 1)
        d_delta_norm = torch.clamp(d_delta_norm, min=1e-8)

        delta_step = -ctx.lr * d_delta / d_delta_norm
        m_step = ctx.lr * (new_m - ctx.m)
        v_step = ctx.lr * (new_v - ctx.v)

        return delta_step, m_step, v_step, None, None, None, None
    

class AdversarialPerturbationAdder(nn.Module):
    def __init__(self, dim_size: int, rho1=0.9, rho2=0.999, lr_reduce=2e-3 / 8e-5, step_size=4):  # param tuning
        super().__init__()
        self.dim_size = dim_size
        self.perturbation = moment_based_adversarial_perturbation.apply
        self.delta = nn.Parameter(torch.zeros(dim_size), requires_grad=True)
        self.m = nn.Parameter(torch.zeros(dim_size), requires_grad=True)
        self.v = nn.Parameter(torch.zeros(dim_size), requires_grad=True)
        self.rho1 = rho1
        self.rho2 = rho2
        self.lr_reduce = lr_reduce
        self.step_size = step_size
        self.global_step = 1
        self._step = 0
        self._input_abs_mean = nn.Parameter(torch.tensor([0]), requires_grad=False)

    def _calc_input_abs_mean(self, inputs: torch.Tensor) -> torch.Tensor:
        alpha = 0.5
        self._input_abs_mean.data = alpha * torch.abs(torch.mean(inputs)) + (1 - alpha) * self._input_abs_mean.data
        return self._input_abs_mean

    def _adaptive_param_reset(self, inputs: torch.Tensor) -> None:
        delta_max = self._calc_input_abs_mean(inputs) * 0.20  # param tuning
        if torch.abs(torch.mean(self.delta)) < delta_max:
            return

        sigma3 = float(delta_max / 5)  # based on 3sigma  # param tuning
        # print("auto reset: delta, m, v, step")
        # print(f"sigma: {sigma3 / 3}")
        # print(f"delta: {float(torch.mean(self.delta.data))} to ", end="")
        torch.nn.init.normal_(self.delta.data, mean=0, std=sigma3)
        # print(f"delta: {float(torch.mean(self.delta.data))}")
        torch.nn.init.zeros_(self.m.data)
        torch.nn.init.zeros_(self.v.data)
        self.global_step = 1
        self._step = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[-1] == self.dim_size
        
        if not self.training:
            return inputs
        
        # auto reset
        self._adaptive_param_reset(inputs)

        # generate perturbation
        p = self.perturbation(self.delta, self.m, self.v, self.global_step, self.rho1, self.rho2, self.lr_reduce)

        # update step
        self._step = (self._step + 1) % self.step_size
        if self._step == 0:
            self.global_step += 1

        return inputs + p.expand(inputs.shape)