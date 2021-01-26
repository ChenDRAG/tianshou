import free_mjc
import os
import datetime

import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PGPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import onpolicy_trainer_basic
from tianshou.data import CachedReplayBuffer, BasicCollector
from torch.distributions import Independent, Normal

from tianshou.utils.net.continuous import Critic
from tianshou.utils import DefaultStepLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=5)
    parser.add_argument('--collect-per-step', type=int, default=5)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=99999)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-tinterval', type=int, default=1)
    parser.add_argument('--log-uinterval', type=int, default=10)
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--hidden-layer-size', type=int, nargs='*', default=[64, 64])#spinging up 256,256  ,TD3 400 300
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def preprocess_fn(**kwargs):
    if 'info' in kwargs:
        for info_dict in kwargs['info']:
            if 'TimeLimit.truncated' not in info_dict:
                info_dict['TimeLimit.truncated'] = False
    return kwargs

def test_pg(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
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
    # 666 doesn't mean anything
    # activation set to tanh to be consistant with spiningup
    # TODO no action range cli in anet?
    anet = Net(
        666, args.state_shape, args.action_shape,
        hidden_layer_size=args.hidden_layer_size,
        device=args.device, activation=torch.nn.Tanh).to(args.device)
    optim = torch.optim.Adam(anet.parameters(), lr=args.lr)


    log_std = torch.nn.Parameter(torch.as_tensor(-0.5 * np.ones(args.action_shape, dtype=np.float32)))
    def dist(logits):
        std = torch.exp(log_std).to(args.device)
        return Independent(Normal(logits, std), 1)
    
    cnet = Net(
        666, args.state_shape,
        hidden_layer_size=args.hidden_layer_size,
        device=args.device, activation=torch.nn.Tanh)
    critic = Critic(cnet, args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

    policy = PGPolicy(anet, optim, dist, args.gamma,
                      reward_normalization=args.rew_norm,
                      approximater = critic,
                      approximater_optim = critic_optim)
    # collector
    cb = CachedReplayBuffer(size = args.buffer_size, cached_buf_n = args.training_num, max_length = 1000)
    train_collector = BasicCollector(
        policy, train_envs, cb, preprocess_fn = preprocess_fn, training=True)
    test_collector = BasicCollector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'vpg', 'seed_' + str(
        args.seed) + '_' + datetime.datetime.now().strftime('%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)
    logger = DefaultStepLogger(writer,
        log_train_interval = args.log_tinterval,
        log_update_interval = args.log_uinterval)

    # trainer
    result = onpolicy_trainer_basic(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, logger=logger)

    if __name__ == '__main__':
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_pg()
