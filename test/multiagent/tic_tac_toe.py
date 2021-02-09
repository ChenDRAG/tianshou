import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy, RandomPolicy, \
    MultiAgentPolicyManager

from tic_tac_toe_env import TicTacToeEnv


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='a smaller gamma favors earlier win')
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--board_size', type=int, default=6)
    parser.add_argument('--win_size', type=int, default=4)
    parser.add_argument('--win_rate', type=float, default=0.9,
                        help='the expected winning rate')
    parser.add_argument('--watch', default=False, action='store_true',
                        help='no training, '
                             'watch the play of pre-trained models')
    parser.add_argument('--agent_id', type=int, default=2,
                        help='the learned agent plays as the'
                             ' agent_id-th player. choices are 1 and 2.')
    parser.add_argument('--resume_path', type=str, default='',
                        help='the path of agent pth file '
                             'for resuming from a pre-trained agent')
    parser.add_argument('--opponent_path', type=str, default='',
                        help='the path of opponent agent pth file '
                             'for resuming from a pre-trained agent')
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_known_args()[0]
    return args


def get_agents(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer]:
    env = TicTacToeEnv(args.board_size, args.win_size)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    if agent_learn is None:
        # model
        net = Net(args.state_shape, args.action_shape,
                  hidden_sizes=args.hidden_sizes, device=args.device
                  ).to(args.device)
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        agent_learn = DQNPolicy(
            net, optim, args.gamma, args.n_step,
            target_update_freq=args.target_update_freq)
        if args.resume_path:
            agent_learn.load_state_dict(torch.load(args.resume_path))

    if agent_opponent is None:
        if args.opponent_path:
            agent_opponent = deepcopy(agent_learn)
            agent_opponent.load_state_dict(torch.load(args.opponent_path))
        else:
            agent_opponent = RandomPolicy()

    if args.agent_id == 1:
        agents = [agent_learn, agent_opponent]
    else:
        agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents)
    return policy, optim


def train_agent(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[dict, BasePolicy]:
    def env_func():
        return TicTacToeEnv(args.board_size, args.win_size)
    train_envs = DummyVectorEnv([env_func for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim = get_agents(
        args, agent_learn=agent_learn,
        agent_opponent=agent_opponent, optim=optim)

    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    if not hasattr(args, 'writer'):
        log_path = os.path.join(args.logdir, 'tic_tac_toe', 'dqn')
        writer = SummaryWriter(log_path)
        args.writer = writer
    else:
        writer = args.writer

    def save_fn(policy):
        if hasattr(args, 'model_save_path'):
            model_save_path = args.model_save_path
        else:
            model_save_path = os.path.join(
                args.logdir, 'tic_tac_toe', 'dqn', 'policy.pth')
        torch.save(
            policy.policies[args.agent_id - 1].state_dict(),
            model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= args.win_rate

    def train_fn(epoch, env_step):
        policy.policies[args.agent_id - 1].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        policy.policies[args.agent_id - 1].set_eps(args.eps_test)

    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer,
        test_in_train=False)

    return result, policy.policies[args.agent_id - 1]


def watch(
    args: argparse.Namespace = get_args(),
    agent_learn: Optional[BasePolicy] = None,
    agent_opponent: Optional[BasePolicy] = None,
) -> None:
    env = TicTacToeEnv(args.board_size, args.win_size)
    policy, optim = get_agents(
        args, agent_learn=agent_learn, agent_opponent=agent_opponent)
    policy.eval()
    policy.policies[args.agent_id - 1].set_eps(args.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
