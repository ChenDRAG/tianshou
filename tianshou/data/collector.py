import gym
import time
import torch
import warnings
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Dict, List, Union, Optional, Callable

from tianshou.policy import BasePolicy
from tianshou.exploration import BaseNoise
from tianshou.data.batch import _create_value
from tianshou.env import BaseVectorEnv, DummyVectorEnv
from tianshou.data import Batch, ReplayBuffer, PrioritizedReplayBuffer, \
    ReplayBufferManager, CachedReplayBuffer, to_numpy


class Collector(object):
    # TODO change doc
    """Collector enables the policy to interact with different types of envs.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
        class. If set to ``None`` (testing phase), it will not store the data.
    :param function preprocess_fn: a function called before the data has been
        added to the buffer, see issue #42 and :ref:`preprocess_fn`, defaults
        to None.
    :param BaseNoise action_noise: add a noise to continuous action. Normally
        a policy already has a noise param for exploration in training phase,
        so this is recommended to use in test collector for some purpose.
    :param function reward_metric: to be used in multi-agent RL. The reward to
        report is of shape [agent_num], but we need to return a single scalar
        to monitor training. This function specifies what is the desired
        metric, e.g., the reward of agent 1 or the average reward over all
        agents. By default, the behavior is to select the reward of agent 1.

    The ``preprocess_fn`` is a function called before the data has been added
    to the buffer with batch format, which receives up to 7 keys as listed in
    :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
    collector resets the environment. It returns either a dict or a
    :class:`~tianshou.data.Batch` with the modified keys and values. Examples
    are in "test/base/test_collector.py".

    Here is the example:
    ::

        policy = PGPolicy(...)  # or other policies if you wish
        env = gym.make('CartPole-v0')
        replay_buffer = ReplayBuffer(size=10000)
        # here we set up a collector with a single environment
        collector = Collector(policy, env, buffer=replay_buffer)

        # the collector supports vectorized environments as well
        envs = DummyVectorEnv([lambda: gym.make('CartPole-v0')
                               for _ in range(3)])
        collector = Collector(policy, envs, buffer=replay_buffer)

        # collect 3 episodes
        collector.collect(n_episode=3)
        # collect 1 episode for the first env, 3 for the third env
        collector.collect(n_episode=[1, 0, 3])
        # collect at least 2 steps
        collector.collect(n_step=2)
        # collect episodes with visual rendering (the render argument is the
        #   sleep time between rendering consecutive frames)
        collector.collect(n_episode=1, render=0.03)

    Collected data always consist of full episodes. So if only ``n_step``
    argument is give, the collector may return the data more than the
    ``n_step`` limitation. Same as ``n_episode`` for the multiple environment
    case.

    .. note::

        Please make sure the given environment has a time limitation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        reward_metric: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        # TODO update training in all test/examples, remove action noise
        super().__init__()
        if not isinstance(env, BaseVectorEnv):
            env = DummyVectorEnv([lambda: env])
        # TODO support or seperate async
        assert env.is_async is False
        self.env = env
        self.env_num = len(env)
        self._save_data = buffer is not None
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = env.action_space
        self._rew_metric = reward_metric or BasicCollector._default_rew_metric
        # avoid creating attribute outside __init__
        self.reset()

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        if buffer is None:
            self.buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert self.buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert self.buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert self.buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to "
                    f"collect {self.env_num} envs, please use {vector_type}("
                    f"total_size={buffer.maxsize}, buffer_num={self.env_num}, "
                    "...) instead.")
            self.buffer = buffer

    @staticmethod
    def _default_rew_metric(
        x: Union[Number, np.number]
    ) -> Union[Number, np.number]:
        # this internal function is designed for single-agent RL
        # for multi-agent RL, a reward_metric must be provided
        assert np.asanyarray(x).size == 1, (
            "Please specify the reward_metric "
            "since the reward is not a scalar.")
        return x

    def reset(self) -> None:
        """Reset all related variables in the collector."""
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(obs={}, act={}, rew={}, done={}, obs_next={},
                          info={}, policy={})
        self.reset_env()
        self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def reset_buffer(self) -> None:
        """Reset the data buffer."""
        self.buffer.reset()

    def reset_env(self) -> None:
        """Reset all of the environment(s)' states and the cache buffers."""
        obs = self.env.reset()
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs).get("obs", obs)
        self.data.obs = obs
        if isinstance(self.buffer, CachedReplayBuffer):
            for buf in self.buffer.cached_buffers:
                buf.reset()

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        state = self.data.state  # it is a reference
        if isinstance(state, torch.Tensor):
            state[id].zero_()
        elif isinstance(state, np.ndarray):
            state[id] = None if state.dtype == np.object else 0
        elif isinstance(state, Batch):
            state.empty_(id)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
    ) -> Dict[str, float]:
        """Collect a specified number of step or episode.

        :param int n_step: how many steps you want to collect.
        :param n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data.
            Default to False.
        :param float render: the sleep time between rendering consecutive
            frames. Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward.
            Default to True (no gradient retaining).

        .. note::

            One and only one collection number specification is permitted,
            either ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` the collected number of episodes.
            * ``n/st`` the collected number of steps.
            * ``rews`` the list of episode reward over collected episodes.
            * ``lens`` the list of episode length over collected episodes.
        """
        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in "
                f"Collector.collect, got n_step = {n_step}, "
                f"n_episode = {n_episode}.")
            assert n_step > 0
            assert n_step % self.env_num == 0, \
                "n_step should be a multiple of #envs"
        else:
            assert isinstance(n_episode, int) and n_episode > 0

        start_time = time.time()

        step_count = 0
        # episode of each environment
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        buffer_ids = list(range(self.env_num))

        while True:
            # restore the state and the input data
            last_state = self.data.state
            if isinstance(last_state, Batch) and last_state.is_empty():
                last_state = None

            # calc the next action / update state / act / policy into self.data
            if random:
                result = Batch(act=[a.sample() for a in self._action_space])
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)

            state = result.get("state", None)
            policy = result.get("policy", Batch())
            act = to_numpy(result.act)
            if state is None:
                # convert None to Batch(), since None is reserved for 0-init
                state = Batch()
            if not (isinstance(state, Batch) and state.is_empty()):
                # save hidden state to policy._state, in order to save into buffer
                policy._state = state

            # TODO discuss and change policy's add_exp_noise behavior
            if self.training and not random and hasattr(self.policy, 'add_exp_noise'):
                act = self.policy.add_exp_noise(act)
            self.data.update(state=state, policy=policy, act=act)

            # step in env
            obs_next, rew, done, info = self.env.step(act)

            result = {"obs_next": obs_next,
                      "rew": rew, "done": done, "info": info}
            if self.preprocess_fn:
                result = self.preprocess_fn(**result)  # type: ignore

            # update obs_next, rew, done, & info into self.data
            self.data.update(result)

            if render:
                self.env.render()
                time.sleep(render)

            # add data into the buffer
            data_t = self.data
            if n_episode and len(cached_buffer_ids) < self.env_num:
                data_t = self.data[cached_buffer_ids]
            if type(self.buffer) == ReplayBuffer:
                data_t = data_t[0]
            # lens, rews, idxs = self.buffer.add(**data_t, index = cached_buffer_ids)
            # rews need to be array for ReplayBuffer
            lens, rews = self.buffer.add(**data_t, index=cached_buffer_ids)
            if type(self.buffer) == ReplayBuffer:
                lens = np.asarray(lens)
                rews = np.asarray(rews)
                # idxs = np.asarray(idxs)
            # collect statistics
            step_count += len(cached_buffer_ids)
            for i in cached_buffer_ids(np.where(lens == 0)[0]):
                episode_count += 1
                episode_lens.append(lens[i])
                episode_rews.append(self._rew_metric(rews[i]))
                # start_idxs.append(idxs[i])

            if sum(done):
                finised_env_ind = np.where(done)[0]
                # now we copy obs_next to obs, but since there might be finished episodes,
                # we have to reset finished envs first.
                obs_reset = self.env.reset(finised_env_ind)
                if self.preprocess_fn:
                    obs_reset = self.preprocess_fn(
                        obs=obs_reset).get("obs", obs_reset)
                self.data.obs_next[finised_env_ind] = obs_reset
                for i in finised_env_ind:
                    self._reset_state(i)
                    if n_episode and n_episode - episode_count < self.env_num:
                        try:
                            cached_buffer_ids.remove(i)
                        except ValueError:
                            pass
            self.data.obs[:] = self.data.obs_next

            if (n_step and step_count >= n_step) or \
               (n_episode and episode_count >= n_episode):
                break

        # generate the statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)
        if n_episode:
            self.reset_env()
        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": np.array(episode_rews),
            "lens": np.array(episode_lens),
            # "idxs": np.array(start_idxs)
        }
