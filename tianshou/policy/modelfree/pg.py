import torch
import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class PGPolicy(BasePolicy):
    """Implementation of Vanilla Policy Gradient.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Callable[[], torch.distributions.Distribution]
    :param float discount_factor: in [0, 1].

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module],
        optim: torch.optim.Optimizer,
        dist_fn: Callable[[], torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        approximater = None,
        approximater_optim = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model: torch.nn.Module = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert (
            0.0 <= discount_factor <= 1.0
        ), "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.approximater = approximater
        self.approximater_optim = approximater_optim

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each frame.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        #TODO why not put this in learn?
        # batch.returns = self._vanilla_returns(batch)
        # batch.returns = self._vectorized_returns(batch)

        # GAE, including TD(0) and baselined reward to go
        # TODO sort trunctrated
        # return self.compute_episodic_return(
        #     batch, gamma=self._gamma, gae_lambda=1.0, rew_norm=self._rew_norm)
        with torch.no_grad():
            v_s = to_numpy(self.approximater(batch.obs))
            v_s_ = to_numpy(self.approximater(batch.obs_next))
        return self.compute_episodic_return_basic(batch, v_s, v_s_, gamma=self._gamma, gae_lambda=0.97, rew_norm=self._rew_norm)

        # TODO add reward to go method, all reward, baseline method, Q/A function


    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, h = self.model(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)  # type: ignore
        # TODO might be rsample
        # action not bounded now
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            #TODO how to solve truncted 
            # split too slow
            for b in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                b_ = self(b)
                dist = b_.dist #why not b.dist  because no gradient, but they should be same
                # original use
                a = to_torch_as(b.act, b_.logits)
                r = to_torch_as(b.returns, b_.logits)
                log_prob = dist.log_prob(a).reshape(len(r), -1).transpose(0, 1)
                loss = -(log_prob * r).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
  
        def v_fn(batch):
            if batch.info['TimeLimit.truncated'] == False:
                return 0
            else:
                return to_numpy(self.approximater(
                    np.expand_dims(batch.obs, 0)))
        
        train_v_iters = 80
        for _ in range(repeat * train_v_iters):
            with torch.no_grad():
                batch = self._rew_to_go(batch, v_fn)
            self.approximater_optim.zero_grad()
            v = self.approximater(batch.obs).flatten() 
            loss2 = ((to_torch_as(batch.target_v, v)- v)**2).mean()
            loss2.backward()
            self.approximater_optim.step()
        print(loss2)
        return {"loss": losses, "loss2": loss2}

    def _rew_to_go(self, batch, v_fn):
        returns = batch.rew.copy()
        last = 0
        for i in range(len(returns) - 1, -1, -1):
            if not batch.done[i]:
                returns[i] += self._gamma * last
            else:
                returns[i] += v_fn(batch[i])
            last = returns[i]
        batch.target_v = returns
        return batch

    # def _vanilla_returns(self, batch):
    #     returns = batch.rew[:]
    #     last = 0
    #     for i in range(len(returns) - 1, -1, -1):
    #         if not batch.done[i]:
    #             returns[i] += self._gamma * last
    #         last = returns[i]
    #     return returns

    # def _vectorized_returns(self, batch):
    #     # according to my tests, it is slower than _vanilla_returns
    #     # import scipy.signal
    #     convolve = np.convolve
    #     # convolve = scipy.signal.convolve
    #     rew = batch.rew[::-1]
    #     batch_size = len(rew)
    #     gammas = self._gamma ** np.arange(batch_size)
    #     c = convolve(rew, gammas)[:batch_size]
    #     T = np.where(batch.done[::-1])[0]
    #     d = np.zeros_like(rew)
    #     d[T] += c[T] - rew[T]
    #     d[T[1:]] -= d[T[:-1]] * self._gamma ** np.diff(T)
    #     return (c - convolve(d, gammas)[:batch_size])[::-1]
