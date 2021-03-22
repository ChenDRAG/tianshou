import torch
import numpy as np
from torch import nn
from typing import Any, Dict, List, Type, Tuple, Union, Optional

from tianshou.policy import PGPolicy
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
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

class PPOPolicy(PGPolicy):
    r"""Implementation of Proximal Policy Optimization. arXiv:1707.06347.

    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.nn.Module critic: the critic network. (s -> V(s))
    :param torch.optim.Optimizer optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param float max_grad_norm: clipping gradients in back propagation.
        Default to None.
    :param float eps_clip: :math:`\epsilon` in :math:`L_{CLIP}` in the original
        paper. Default to 0.2.
    :param float vf_coef: weight for value loss. Default to 0.5.
    :param float ent_coef: weight for entropy loss. Default to 0.01.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param float gae_lambda: in [0, 1], param for Generalized Advantage
        Estimation. Default to 0.95.
    :param float dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5,
        where c > 1 is a constant indicating the lower bound.
        Default to 5.0 (set None if you do not want to use it).
    :param bool value_clip: a parameter mentioned in arXiv:1811.02553 Sec. 4.1.
        Default to True.
    :param bool reward_normalization: normalize the returns to Normal(0, 1).
        Default to True.
    :param int max_batchsize: the maximum size of the batch when computing GAE,
        depends on the size of available memory and the memory cost of the
        model; should be as large as possible within the memory constraint.
        Default to 256.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        max_grad_norm: Optional[float] = None,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        action_range: Optional[Tuple[float, float]] = None,
        gae_lambda: float = 0.95,
        dual_clip: Optional[float] = None,
        value_clip: bool = True,
        max_batchsize: int = 256,
        max_repeat: int = 10,
        temp_bound_action=0,
        target_kl: Optional[float] = None,
        norm_adv = True,
        recompute_adv = False,
        enhance_critic = 0,
        **kwargs: Any,
    ) -> None:
        # TODO lr, 3 optional, calculate kl.
        # TODO minibatch clip good?
        super().__init__(None, optim, dist_fn, discount_factor, **kwargs)
        # self.ret_rms = RunningMeanStd(shape=())
        # self.clip_reward = 10.0
        # self.epsilon = 1e-8
        self.ret_rms_rtg = RunningMeanStd(shape=())
        self.ret_rms_rtg_nb = RunningMeanStd(shape=())
        self.ret_rms_l0 = RunningMeanStd(shape=())
        self.clip_num = 0
        self.total_num = 0
        self.max_repeat = max_repeat
        self.target_kl = target_kl
        self._max_grad_norm = max_grad_norm
        self._eps_clip = eps_clip
        self._weight_vf = vf_coef
        self._weight_ent = ent_coef
        self._range = action_range
        self.actor = actor
        self.critic = critic
        self._batch = max_batchsize
        assert 0.0 <= gae_lambda <= 1.0, "GAE lambda should be in [0, 1]."
        self._lambda = gae_lambda
        assert dual_clip is None or dual_clip > 1.0, \
            "Dual-clip PPO parameter should greater than 1.0."
        self._dual_clip = dual_clip
        self._value_clip = value_clip
        self.norm_adv = norm_adv
        self.recompute_adv = recompute_adv
        self.bound_action_method = temp_bound_action
        self.buffer_buf = None
        self.indice_buf = None
        self.enhance_critic = enhance_critic

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        # self.compute_returns(batch, buffer, indice)
        print_loss = []
        if self.enhance_critic > 0:
            for _ in range(self.enhance_critic):
                vf_losses = []
                self.compute_returns(batch, buffer, indice)
                for b in batch.split(64, merge_last=True):
                    value = self.critic(b.obs).flatten()
                    if self._value_clip:
                        v_clip = b.v_s + (value - b.v_s).clamp(-self._eps_clip, self._eps_clip)
                        vf1 = (b.returns - value).pow(2)
                        vf2 = (b.returns - v_clip).pow(2)
                        vf_loss = 0.5 * torch.max(vf1, vf2).mean()
                    else:
                        vf_loss = 0.5 * (b.returns - value).pow(2).mean()
                    vf_losses.append(vf_loss.item())

                    loss = self._weight_vf * vf_loss
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                print_loss.append(np.sum(vf_losses))
            print(print_loss)
        assert self.buffer_buf is None
        assert self.indice_buf is None
        self.buffer_buf = buffer
        self.indice_buf = indice
        old_log_prob = []
        ori_acts = []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                if self.bound_action_method != 2: 
                    old_log_prob.append(self(b).dist.log_prob(to_torch_as(b.act, batch.v_s[0])))
                else:
                    ori_act = to_torch_as((np.log(1 + b.act/self._range[1]) - np.log(1 - b.act/self._range[1])) / 2.0, batch.v_s[0])
                    ori_acts.append(ori_act)
                    old_log_prob.append(self(b).dist.log_prob(ori_act))
        if self.bound_action_method == 2:
           batch.ori_act = torch.cat(ori_acts, dim=0)
        batch.logp_old = torch.cat(old_log_prob, dim=0)
        batch.act = to_torch_as(batch.act, batch.v_s[0])
        return batch

    def compute_returns(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ):
        v_s, v_s_ = [], []
        with torch.no_grad():
            for b in batch.split(self._batch, shuffle=False, merge_last=True):
                v_s_.append(self.critic(b.obs_next))
                v_s.append(self.critic(b.obs))
        batch.v_s_ = torch.cat(v_s_, dim=0).flatten()
        batch.v_s = torch.cat(v_s, dim=0).flatten()
        if self._rew_norm:
            un_norm_v_s = self.unnormalize_reward(to_numpy(batch.v_s).flatten())
            un_norm_v_s_ = self.unnormalize_reward(to_numpy(batch.v_s_).flatten())
        batch.adv, un_norm_returns = self.compute_gae_return(
            batch, buffer, indice, un_norm_v_s_, un_norm_v_s, gamma=self._gamma,
            gae_lambda=self._lambda)

        _, un_norm_returns_rtg = self.compute_gae_return(
            batch, buffer, indice, un_norm_v_s_, un_norm_v_s, gamma=self._gamma,
            gae_lambda=1)
        _, un_norm_returns_rtg_nb = self.compute_gae_return(
            batch, buffer, indice, np.zeros_like(un_norm_v_s_), np.zeros_like(un_norm_v_s), gamma=self._gamma,
            gae_lambda=1)
        _, un_norm_returns_l0 = self.compute_gae_return(
            batch, buffer, indice, un_norm_v_s_, un_norm_v_s, gamma=self._gamma,
            gae_lambda=0)

        if self.norm_adv:
            batch.adv = (batch.adv - batch.adv.mean())/batch.adv.std()
        if self._rew_norm:
            batch.returns = self.normalize_reward(un_norm_returns)
        # end_flag = batch.done.copy()
        # end_flag[np.isin(indice, buffer.unfinished_index())] = True
        # batch.returns = self._rew_to_go(batch, v_s_, end_flag)
        batch.returns = to_torch_as(batch.returns, batch.v_s[0])
        batch.adv = to_torch_as(batch.adv, batch.v_s[0])
        self.ret_rms.update(un_norm_returns)
        # self.ret_rms_rtg.update(un_norm_returns_rtg)
        # self.ret_rms_rtg_nb.update(un_norm_returns_rtg_nb)
        # self.ret_rms_l0.update(un_norm_returns_l0)
        # self.mean = un_norm_returns.mean()
        # self.std = un_norm_returns.std()
        return batch
            

    # def _rew_to_go(self, batch, v_s_, end_flag):
    #     v_s_ = to_numpy(v_s_.flatten())
    #     returns = batch.rew.copy()
    #     last = 0
    #     for i in range(len(returns) - 1, -1, -1):
    #         if not end_flag[i]:
    #             returns[i] += self._gamma * last
    #         else:
    #             returns[i] += v_s_[i]
    #         last = returns[i]
    #     return returns

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
        logits, h = self.actor(batch.obs, state=state, info=batch.info)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        act = dist.sample()
        if self.bound_action_method == 2:
            act = torch.tanh(act) * self._range[1]
        return Batch(logits=logits, act=act, state=h, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        assert batch_size == 64
        self.change_lr(self.optim)
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        # for _ in range(repeat):
        assert self.max_repeat == 10
        for step in range(self.max_repeat):
            if self.recompute_adv and step > 0:
                batch = self.compute_returns(batch, self.buffer_buf, self.indice_buf)
            approx_kls = []
            for b in batch.split(batch_size, merge_last=True):
                dist = self(b).dist
                # ori_act = (torch.log(1 + b.act) - torch.log(1 - b.act)) / 2.0
                if self.bound_action_method != 2:
                    logp = dist.log_prob(b.act)
                else:
                    logp = dist.log_prob(b.ori_act)
                ratio = (logp - b.logp_old).exp().float()
                ratio = ratio.reshape(ratio.size(0), -1).transpose(0, 1)
                surr1 = ratio * b.adv
                surr2 = ratio.clamp(1.0 - self._eps_clip, 1.0 + self._eps_clip) * b.adv
                if self._dual_clip:
                    assert False
                    clip_loss = -torch.max(
                        torch.min(surr1, surr2), self._dual_clip * b.adv
                    ).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()
                clip_losses.append(clip_loss.item())

                # value = self.critic(b.obs).flatten()
                # if self._value_clip:
                #     v_clip = b.v_s + (value - b.v_s).clamp(-self._eps_clip, self._eps_clip)
                #     vf1 = (b.returns - value).pow(2)
                #     vf2 = (b.returns - v_clip).pow(2)
                #     vf_loss = 0.5 * torch.max(vf1, vf2).mean()
                # else:
                #     vf_loss = 0.5 * (b.returns - value).pow(2).mean()
                # vf_losses.append(vf_loss.item())
                e_loss = dist.entropy().mean()
                ent_losses.append(e_loss.item())
                # loss = clip_loss + self._weight_vf * vf_loss \
                #     - self._weight_ent * e_loss
                loss = clip_loss - self._weight_ent * e_loss
                losses.append(loss.item())
                self.optim.zero_grad()
                loss.backward()
                if self._max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()), self._max_grad_norm)
                self.optim.step()
                approx_kls.append(torch.mean(b.logp_old - logp).detach().cpu().numpy())
            
            if self.target_kl and np.mean(approx_kls) > 1.5 * self.target_kl:
                assert False
                print(f"Early stopping at step {step} due to reaching max kl: {np.mean(approx_kls):.3f}")
                break
        
        self.buffer_buf = None
        self.indice_buf = None
        return {
            "loss": losses,
            "loss/clip": clip_losses,
            # "loss/vf": vf_losses,
            "loss/ent": ent_losses,
            # "loss/gae_std":[np.sqrt(self.ret_rms.var)],
            # "loss/gae_mean":[self.ret_rms.mean],
            # "loss/rtg_std":[np.sqrt(self.ret_rms_rtg.var)],
            # "loss/rtg_mean":[self.ret_rms_rtg.mean],
            # "loss/rtg_std_nb":[np.sqrt(self.ret_rms_rtg_nb.var)],
            # "loss/rtg_mean_nb":[self.ret_rms_rtg_nb.mean],
            # "loss/l0_std":[np.sqrt(self.ret_rms_l0.var)],
            # "loss/l0_mean":[self.ret_rms_l0.mean],
            # "loss/real_std":[self.std],
            # "loss/real_mean":[self.mean],
        }

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        #TODO check need mean
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        reward = reward / np.sqrt(self.ret_rms.var + self.epsilon)
        return reward
    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        return reward * np.sqrt(self.ret_rms.var + self.epsilon)
