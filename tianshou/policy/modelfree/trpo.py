import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Type, Optional
from torch.distributions import kl_divergence


from tianshou.policy import A2CPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as

def conjugate_gradients(A, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    # Note: should be 'b - A(x)', but for x=0, A(x)=0. Change if doing warm start.
    # r = b - A(x)
    # p = r
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        z = A(p)
        alpha = rdotr / torch.dot(p, z)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            print("Early solving cg at {} step".format(i))
            return x
    return x

class TRPOPolicy(A2CPolicy):
    # https://towardsdatascience.com/introducing-k-fac-and-its-application-for-large-scale-deep-learning-4e3f9b443414
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        advantage_normalization: bool = True,
        optim_critic_iters: int = 3,
        max_kl: float = 0.01,
        backtrack_coeff: float = 0.8,
        max_backtracks: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor, critic, optim, dist_fn, **kwargs)
        del self._weight_vf
        del self._weight_ent
        del self._grad_norm
        self._norm_adv = advantage_normalization
        self._optim_critic_iters = optim_critic_iters
        self._max_backtracks = max_backtracks
        self._delta = max_kl
        self._backtrack_coeff = backtrack_coeff  #trpo 0.5
        self.__damping = 0.1

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        batch = self._compute_returns(batch, buffer, indice)
        batch.act = to_torch_as(batch.act, batch.v_s)
        old_log_prob = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                old_log_prob.append(self(b).dist.log_prob(b.act))
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        if self._norm_adv:
            batch.adv = (batch.adv - batch.adv.mean()) / batch.adv.std()
        return batch

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        actor_losses, vf_losses, step_sizes = [], [], []
        for step in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                # optimize actor
                # direction: calculate villia gradient
                dist = self(b).dist # TODO could come from batch
                ratio = (dist.log_prob(b.act) - b.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                actor_loss = - (ratio * b.adv).mean()
                with torch.no_grad():
                    grads = torch.autograd.grad(actor_loss, self.actor.parameters(), retain_graph=True)
                    flattened_grads = torch.cat([grad.view(-1) for grad in grads]).data

                # direction: calculate natural gradient
                # calculate kl divergence
                with torch.no_grad():
                    old_dist = self(b).dist
                kl = kl_divergence(old_dist, dist).mean()
                kl_grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
                flattened_kl_grad = torch.cat([kl_grad.view(-1) for kl_grad in kl_grads])
                def MVP(v):
                    # matrix vector product
                    # caculate second order gradient of kl with respect to theta
                    kl_v = (flattened_kl_grad * v).sum() # why variable
                    kl_grads2 = torch.autograd.grad(kl_v, self.actor.parameters(), retain_graph=True)
                    flat_grad_grad_kl = torch.cat([kl_grad.contiguous().view(-1) for kl_grad in kl_grads2]).data
                    return flat_grad_grad_kl + v * self.__damping

                search_direction = - conjugate_gradients(MVP, flattened_grads, nsteps=10)

                # stepsize: calculate max stepsize constrained by kl bound
                step_size = torch.sqrt(
                    2 * self._delta / (search_direction * MVP(search_direction)).sum(0, keepdim=True))

                # stepsize: linesearch stepsize
                with torch.no_grad():
                    flattened_params = torch.cat([param.data.view(-1) for param in self.actor.parameters()])
                    for i in range(self._max_backtracks):
                        new_flattened_params = flattened_params + step_size * search_direction
                        prev_ind = 0
                        for param in self.actor.parameters():
                            flat_size = int(np.prod(list(param.size())))
                            param.data.copy_(
                                new_flattened_params[prev_ind: prev_ind + flat_size].view(param.size()))
                            prev_ind += flat_size
                        # calculate kl and if in bound, loss actually down
                        new_dist = self(b).dist
                        new_dratio = (new_dist.log_prob(b.act) - b.logp_old).exp().float()
                        new_dratio = new_dratio.reshape(new_dratio.size(0), -1).transpose(0, 1)
                        new_actor_loss = - (new_dratio * b.adv).mean()
                        kl = kl_divergence(old_dist, new_dist).mean()

                        if  kl < 1.5 * self._delta and new_actor_loss < actor_loss:
                            if i!=0:
                                print('Accepting new params at step %d of line search.'%i)
                            break
                        elif i < self._max_backtracks - 1:
                            step_size = step_size * self._backtrack_coeff
                        else:
                            print('Line search failed! Keeping old params.')
                            step_size = 0
                            prev_ind = 0
                            for param in self.actor.parameters():
                                flat_size = int(np.prod(list(param.size())))
                                param.data.copy_(
                                    flattened_params[prev_ind: prev_ind + flat_size].view(param.size()))
                                prev_ind += flat_size

                # optimize citirc
                for _ in range(self._optim_critic_iters):
                    value = self.critic(b.obs).flatten()
                    # TODO use gae or rtg?
                    vf_loss = F.mse_loss(b.returns, value)
                    self.optim.zero_grad()
                    vf_loss.backward()
                    self.optim.step()

                actor_losses.append(actor_loss.item())
                vf_losses.append(vf_loss.item())
                step_sizes.append(step_size.item())
        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {
            "loss/actor": actor_losses,
            "loss/vf": vf_losses,
            "loss/step_sizes": step_sizes,
        }