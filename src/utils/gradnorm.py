import torch


class GradNormController:
    """
    GradNorm: Gradient Normalization for adaptive loss balancing in multitask learning.
    Reference: Chen et al., 2018 — https://arxiv.org/abs/1711.02257

    Maintains per-task weights that are updated each step so that all tasks
    train at a similar rate, preventing any single task from dominating.

    Usage:
        controller = GradNormController(n_tasks=10, alpha=1.5, device=device)
        controller.set_weight_optimizer(lr=1e-2)
        shared_param = model.backbone.layer4[-1].bn2.weight

        # Inside training loop:
        losses = torch.stack([loss_fn_i(pred_i, target_i) for i in range(n_tasks)])
        info = controller.step(losses, shared_param, model_optimizer)
    """

    def __init__(self, n_tasks: int, alpha: float = 1.5, device: str = 'cpu'):
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.initial_losses: torch.Tensor | None = None
        # Start with uniform weights summing to n_tasks
        self.weights = torch.ones(n_tasks, device=device, requires_grad=True)
        self._optimizer: torch.optim.Optimizer | None = None

    def set_weight_optimizer(self, lr: float = 1e-2):
        self._optimizer = torch.optim.Adam([self.weights], lr=lr)

    @property
    def current_weights(self) -> list[float]:
        return self.weights.detach().cpu().tolist()

    def step(
        self,
        losses: torch.Tensor,
        shared_param: torch.Tensor,
        model_optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        One GradNorm step: updates model params and task weights.

        Args:
            losses: per-task losses, shape (n_tasks,). Must still have a live graph.
            shared_param: weight tensor from the last shared layer (e.g. backbone.layer4[-1].bn2.weight).
            model_optimizer: optimizer for the model parameters.

        Returns:
            dict with keys 'total_loss', 'gradnorm_loss', 'weights'.
        """
        assert self._optimizer is not None, "Call set_weight_optimizer() before step()"

        n = self.n_tasks

        if self.initial_losses is None:
            self.initial_losses = losses.detach().clone()

        # Per-task gradient norms at the shared layer.
        # create_graph=True lets gradnorm_loss differentiate back through grad_norms to self.weights.
        grad_norms = torch.stack([
            torch.autograd.grad(
                self.weights[i] * losses[i],
                shared_param,
                retain_graph=True,
                create_graph=True,
            )[0].norm()
            for i in range(n)
        ])

        # Relative inverse training rates r_i: tasks falling behind get a larger target norm.
        mean_norm = grad_norms.detach().mean()
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        r_i = loss_ratios / (loss_ratios.mean() + 1e-8)
        targets = (mean_norm * r_i ** self.alpha).detach()

        # GradNorm loss — L1 distance between actual and target gradient norms.
        gradnorm_loss = (grad_norms - targets).abs().sum()

        # Update task weights only (inputs=[self.weights] keeps model-param grads untouched).
        self._optimizer.zero_grad()
        gradnorm_loss.backward(inputs=[self.weights], retain_graph=True)
        self._optimizer.step()

        # Renormalize so weights sum to n_tasks (preserves scale invariance).
        with torch.no_grad():
            self.weights.data.clamp_(min=1e-4)
            self.weights.data.mul_(n / self.weights.data.sum())

        # Update model params with the renormalized (detached) weights.
        weighted_loss = (self.weights.detach() * losses).sum()
        model_optimizer.zero_grad()
        weighted_loss.backward()
        model_optimizer.step()

        return {
            'total_loss': weighted_loss.item(),
            'gradnorm_loss': gradnorm_loss.item(),
            'weights': self.current_weights,
        }
