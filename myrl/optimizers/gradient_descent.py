import torch


class PolicyGradientOptimizer:

    def __init__(self, parameters, alpha):
        self.parameters = list(parameters)
        self.alpha = alpha

    @torch.no_grad
    def print_parameters(self):
        for idx, parameter in enumerate(self.parameters):
            print(f"P({idx})={parameter}, Grad={parameter.grad}")
    
    @torch.no_grad
    def step(self) -> None:
        for idx, parameter in enumerate(self.parameters):
            # Gradient Descent because the loss is negative
            # i.e., minimizing the loss (indirectly) maximizes expected return
            parameter.add_(parameter.grad, alpha=-self.alpha)
        return

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            # if not parameter.grad:
            #     continue
            #parameter.grad.zero_()
            parameter.grad = None
        return