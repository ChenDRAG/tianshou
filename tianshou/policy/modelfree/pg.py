import torch
import numpy as np
from typing import Any, Dict, List, Type, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as
class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class PGPolicy(BasePolicy):
    """Implementation of Vanilla Policy Gradient.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module],
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_range: Optional[Tuple[float, float]] = None,
        temp_minusmean = 0,
        temp_bound_action=0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_reward = np.inf
        self.epsilon = 1e-8
        if model is not None:
            self.model: torch.nn.Module = model
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization

        self._range = action_range
        self.bound_action_method = temp_bound_action
        self.minus = temp_minusmean

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        # batch.returns = self._vanilla_returns(batch)
        # batch.returns = self._vectorized_returns(batch)
        v_s_ = np.full(indice.shape, self.ret_rms.mean)
        _, un_norm_returns = self.compute_gae_return(
            batch, buffer, indice, v_s_, gamma=self._gamma,
            gae_lambda=1.0, adv_norm=False)
        self.ret_rms.update(un_norm_returns)
        self.mean = un_norm_returns.mean()
        self.std = un_norm_returns.std()
        # TODO check minus okay, think how norm adv
        batch.returns = self.normalize_reward(un_norm_returns)
        return batch

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
            dist = self.dist_fn(logits)
        act = dist.sample()
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        # TODO whether change lr
        self.change_lr(self.optim)
        losses = []
        for _ in range(repeat):
            for b in batch.split(batch_size, merge_last=True):
                self.optim.zero_grad()
                t = self(b)
                dist = t.dist
                a = to_torch_as(b.act, t.act)
                r = to_torch_as(b.returns, t.act)
                log_prob = dist.log_prob(a).reshape(len(r), -1).transpose(0, 1)
                loss = -(log_prob * r).mean()
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        return {
            "loss": losses,
            "loss/gae_std":[np.sqrt(self.ret_rms.var)],
            "loss/gae_mean":[self.ret_rms.mean],
            "loss/real_std":[self.std],
            "loss/real_mean":[self.mean],
        }
    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        if self.minus:
            reward = (reward - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + self.epsilon)
        else:
            reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        return reward
    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.minus:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon) + self.ret_rms.mean
        else:
            return reward * np.sqrt(self.ret_rms.var + self.epsilon)
