# import pickle, numpy as np
# from tianshou.data import CachedReplayBuffer
# buf = CachedReplayBuffer(size=20, cached_buf_n = 2, max_length = 3)

# for i in range(3):
#     buf.add(obs=[i]*2, act=[i]*2, rew=[i]*2, done=[i]*2, obs_next=[i + 1]*2, info=[{}]*2)
# print(buf.obs)
# print(buf.done)

# # but there are only three valid items, so len(buf) == 3.
# print(len(buf))
# buf2 = CachedReplayBuffer(size=20, cached_buf_n = 2, max_length = 3)
# for i in range(15):
#     buf2.add(obs=[i]*2, act=[i]*2, rew=[i]*2, done=[i]*2, obs_next=[i + 1]*2, info=[{}]*2)
# print(len(buf2))
# print(buf2.obs)

# # move buf2's result into buf (meanwhile keep it chronologically)
# buf.update(buf2)
# print(buf)
# # get a random sample from buffer
# # the batch_data is equal to buf[incide].
# batch_data, indice = buf.sample(batch_size=4)
# print(batch_data.obs == buf[indice].obs)
# print(len(buf))
import free_mjc
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import TD3Policy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer, CachedReplayBuffer, BasicCollector
from tianshou.utils.net.continuous import Actor, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Ant-v2')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=2400)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=2)
    parser.add_argument('--test-num', type=int, default=5)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def test_td3(args=get_args()):
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
    net = Net(args.layer_num, args.state_shape, device=args.device)
    actor = Actor(
        net, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net = Net(args.layer_num, args.state_shape,
              args.action_shape, concat=True, device=args.device)
    critic1 = Critic(net, args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net, args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    policy = TD3Policy(
        actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
        action_range=[env.action_space.low[0], env.action_space.high[0]],
        tau=args.tau, gamma=args.gamma,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        policy_noise=args.policy_noise,
        update_actor_freq=args.update_actor_freq,
        noise_clip=args.noise_clip,
        reward_normalization=True, ignore_done=True)
    



    # collector
    cb = CachedReplayBuffer(size = 20, cached_buf_n = 2, max_length = 1000)
    train_collector = BasicCollector(
        policy, train_envs, cb)
    train_collector.collect(n_step = 2)

    # cb = ReplayBuffer(args.buffer_size)
    # train_collector = Collector(
    #     policy, train_envs, cb)
    # train_collector.collect(n_step=2)
    # test_collector = Collector(policy, test_envs)


if __name__ == '__main__':
    test_td3()
