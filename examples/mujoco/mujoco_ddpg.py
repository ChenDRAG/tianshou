import free_mjc
import os
import datetime

import gym
import torch
from torch import nn
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from tianshou.policy import DDPGPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import miniblock
from tianshou.exploration import GaussianNoise
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.data import to_torch, to_torch_as

# from tianshou.utils.net.continuous import Actor, Critic


class Actor(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_layer_size = [],
        max_action = 1.0,
        device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self._max = max_action
        if isinstance(hidden_layer_size, int):
            hidden_layer_size = [hidden_layer_size]
        hidden_layer_size.insert(0, np.prod(obs_shape))
        hidden_layer_size.insert(len(hidden_layer_size), np.prod(action_shape))
        self.last = nn.Linear(hidden_layer_size[-2], hidden_layer_size[-1])
        model = []
        for i in range(len(hidden_layer_size) - 2):
            model += miniblock(
                hidden_layer_size[i], hidden_layer_size[i+1])
        self.model = nn.Sequential(*model)
        
    def forward(
        self,
        s,
        state = None,
        info = {},
    ):
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.reshape(s.size(0), -1)
        logits = self.model(s)
        act = self._max * torch.tanh(self.last(logits))
        return act, None

class Critic(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_shape,
        hidden_layer_size = [],
        max_action = 1.0,
        device = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self._max = max_action
        if isinstance(hidden_layer_size, int):
            hidden_layer_size = [hidden_layer_size]
        hidden_layer_size.insert(0, np.prod(obs_shape) + np.prod(action_shape))
        hidden_layer_size.insert(len(hidden_layer_size), 1)
        self.last = nn.Linear(hidden_layer_size[-2], hidden_layer_size[-1])
        model = []
        for i in range(len(hidden_layer_size) - 2):
            model += miniblock(
                hidden_layer_size[i], hidden_layer_size[i+1])
        self.model = nn.Sequential(*model)
        
    def forward(
        self,
        s,a,
        info = {},
    ):
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        s = s.flatten(1)
        a = to_torch_as(a, s)
        a = a.flatten(1)
        logits = torch.cat([s, a], dim=1)
        logits = self.model(logits)
        q = self.last(logits)
        return q

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-layer-size', type=int, nargs='*', default=[])
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def test_ddpg(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # train_envs = gym.make(args.task)
    train_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)])
    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    actor = Actor(args.state_shape, args.action_shape, args.hidden_layer_size, args.max_action, args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic = Critic(args.state_shape, args.action_shape, args.hidden_layer_size, args.max_action, args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=None)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size),
        action_noise=GaussianNoise(sigma=args.exploration_noise))
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg', 'seed_' + str(
        args.seed) + '_' + datetime.datetime.now().strftime('%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, writer=writer, log_interval=args.log_interval)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')


if __name__ == '__main__':
    test_ddpg()
