import free_mjc
import os
import datetime

import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DDPGPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.trainer import offpolicy_trainer_basic
from tianshou.data import CachedReplayBuffer, BasicCollector
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.utils import DefaultStepLogger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--buffer-size', type=int, default=1000000)
    parser.add_argument('--hidden-layer-size', type=int, nargs='*', default=[])#spinging up 256,256  ,TD3 400 300
    parser.add_argument('--actor-lr', type=float, default=1e-3)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument("--start-timesteps", type=int, default=25000)#10000 for spinging up
    parser.add_argument('--epoch', type=int, default=250)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--collect-per-step', type=int, default=1)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=256)#100 for spinging up
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-tinterval', type=int, default=1)
    parser.add_argument('--log-uinterval', type=int, default=1000)
    parser.add_argument('--render', type=float, default=0.)
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

def test_ddpg(args=get_args()):
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.exploration_noise = args.exploration_noise * args.max_action
    #TODO 3 noises max action 
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
    anet = Net(666, args.state_shape,
              hidden_layer_size=args.hidden_layer_size, device=args.device)
    actor = Actor(
        anet, args.action_shape,
        args.max_action, args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    cnet = Net(666, args.state_shape,
               args.action_shape, concat=True,
               hidden_layer_size=args.hidden_layer_size,
               device=args.device)
    critic = Critic(cnet, args.device).to(args.device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    policy = DDPGPolicy(
        actor, actor_optim, critic, critic_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        ignore_done=True)

    # collector
    cb = CachedReplayBuffer(size = args.buffer_size, cached_buf_n = args.training_num, max_length = 1000)
    cb_test = CachedReplayBuffer(size = args.buffer_size, cached_buf_n = args.test_num, max_length = 1000)#TODO this is not necessary
    train_collector = BasicCollector(
        policy, train_envs, cb, preprocess_fn = preprocess_fn, training=True)
    test_collector = BasicCollector(policy, test_envs, cb_test)
    train_collector.collect(n_step=args.start_timesteps, random=True)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ddpg', 'seed_' + str(
        args.seed) + '_' + datetime.datetime.now().strftime('%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)
    logger = DefaultStepLogger(writer,
        log_train_interval = args.log_tinterval,
        log_update_interval = args.log_uinterval)

    # trainer
    result = offpolicy_trainer_basic(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, update_per_step = args.update_per_step,
        logger=logger)

    if __name__ == '__main__':
        # Let's watch its performance!
        policy.eval()
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=args.test_num,
                                        render=args.render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


if __name__ == '__main__':
    test_ddpg()
