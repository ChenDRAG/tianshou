import free_mjc
import os
import gym
import torch
import datetime
import argparse
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Independent, Normal
from copy import deepcopy

from tianshou.policy import PPOPolicy
from tianshou.utils import BasicLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic
from typing import Any, List, Union, Optional, Callable, Tuple

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


class normalize_train(SubprocVectorEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        wait_num: Optional[int] = None,
        timeout: Optional[float] = None,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env_fns, wait_num=wait_num, timeout=timeout)
        # check TODO
        if training:
            self.obs_rms = RunningMeanStd(shape=self.observation_space[0].shape)
            # self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        # self.clip_reward = clip_reward
        # Returns: discounted rewards
        # self.ret = np.zeros(self.env_num)
        # self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        # self.norm_reward = norm_reward
        self._buffer = np.arange(self.env_num)

    def step(
        self,
        action: np.ndarray,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> List[np.ndarray]:
        if id is None:
            id = self._buffer
        obs_next, rew, done, info = super().step(action, id)
        if self.training:
            self.obs_rms.update(obs_next)
            # self.ret[id] = self.ret[id] * self.gamma + rew
            # self.ret_rms.update(self.ret[id])
        obs_next = self.normalize_obs(obs_next)

        # # when normalise reward is testing
        # if self.training:
        #     normalised_rew = self.normalize_reward(deepcopy(rew))
        #     for i, r in zip(info, normalised_rew):
        #         i['normalised_r'] = r

        # self.ret[id][done] = False
        return obs_next, rew, done, info

    def reset(
        self, id: Optional[Union[int, List[int], np.ndarray]] = None
    ) -> np.ndarray:
        if id is None:
            id = self._buffer
        obs = super().reset(id)
        # self.ret[id] = np.zeros(len(id))
        if self.training:
            self.obs_rms.update(obs)
            # self._update_reward(self.ret)
        return self.normalize_obs(obs)

    # def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
    #     """
    #     Normalize rewards using this VecNormalize's rewards statistics.
    #     Calling this method does not update statistics.
    #     """
    #     if self.norm_reward:
    #         reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
    #     return reward

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        # Avoid modifying by reference the original object
        # obs_ = deepcopy(obs)
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return obs



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Walker2d-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=8000)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=30000)
    parser.add_argument('--step-per-collect', type=int, default=2048)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)# 64 in baselines
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)#ori 100
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume-path', type=str, default=None)
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)#0.5 in sb3
    parser.add_argument('--gae-lambda', type=float, default=0.95)# 0.95 in baselines 0.97 in spinning up
    parser.add_argument('--rew-norm', type=int, default=True)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=True)
    parser.add_argument('--target-kl', type=int, default=0.01)
    parser.add_argument('--max-repeat', type=int, default=10)#10 in baselines /sb3
    parser.add_argument('--temp_rnorm', type=int, default=1)#10 in baselines /sb3
    return parser.parse_args()



def test_ppo(args=get_args()):
    torch.set_num_threads(1)  # TODO we just need only one thread for NN
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low),
          np.max(env.action_space.high))
    # train_envs = gym.make(args.task)
    train_envs = normalize_train(
        [lambda: gym.make(args.task) for _ in range(args.training_num)], gamma=args.gamma, training=True)
    # test_envs = gym.make(args.task)
    test_envs = normalize_train(
        [lambda: gym.make(args.task) for _ in range(args.test_num)], gamma=args.gamma, training=False)
    test_envs.obs_rms = train_envs.obs_rms
        
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    actor = ActorProb(net_a, args.action_shape, max_action=args.max_action, unbounded=True,
                      device=args.device)
    # TODO check unbounded and -0.5
    actor.sigma_param = nn.Parameter(torch.as_tensor(-0.5 * np.ones((actor.output_dim, 1), dtype=np.float32)))
    actor = actor.to(args.device)
    net_c = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation=nn.Tanh, device=args.device)
    critic = Critic(net_c, device=args.device).to(args.device)

    # orthogonal initialization TODO checkout
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(set(
        actor.parameters()).union(critic.parameters()), lr=args.lr, eps=1e-5)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward TODO
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    delta = args.lr / (np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch)
    print(delta)
    def change_lr(optim):
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] - delta
    

    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        max_repeat=args.max_repeat,
        target_kl=args.target_kl,
        # dual_clip=args.dual_clip,
        # dual clip cause monotonically increasing log_std :)
        value_clip=args.value_clip,
        # action_range=[env.action_space.low[0], env.action_space.high[0]],)
        # if clip the action, ppo would not converge :)
        gae_lambda=args.gae_lambda)
    
    policy.change_lr = change_lr

    # collector
    if args.training_num > 1:
        buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(args.buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo', 'seed_' + str(
        args.seed) + '_' + datetime.datetime.now().strftime('%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.repeat_per_collect, args.test_num, args.batch_size,
        step_per_collect=args.step_per_collect, save_fn=save_fn,
        logger=logger, test_in_train=False)

    # Let's watch its performance!
    policy.eval()
    test_envs.seed(args.seed)
    test_collector.reset()
    result = test_collector.collect(n_episode=args.test_num, render=args.render)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_ppo()
