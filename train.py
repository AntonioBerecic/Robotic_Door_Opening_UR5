import mujoco_py
import gym
import argparse
from gym import envs
import os
import torch
from rl.td3.train_td3 import train_td3
import numpy as np
import multiprocessing
# Reduce CPU/BLAS thread usage to limit CPU contention and RAM usage
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
multiprocessing.set_start_method('spawn', force=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    parser.add_argument('--env', type=str, help='Environment', required=True)
    parser.add_argument('--render', dest='render', action='store_true',
                    help='Enable openai gym real-time rendering')
    parser.add_argument('--process', type=int, default=1,
                    help='Process count for parallel exploration')
    parser.add_argument('--model', dest='path', type=str, default=None,
                help='Moddel weights location')
    parser.add_argument('--model_id', dest='model_id', type=int, default=0,
            help='Moddel weights id (step for saving the model)')
    parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
            help='Load a pretrained model and finetune it')
    parser.add_argument('--seed', dest='seed', type=int, default=1234,
            help='Random seed')
    parser.add_argument('--alg', dest='alg', type=str, default='td3',
                help='Choose algorithm type')
    parser.add_argument('--fast', dest='fast', action='store_true', default=False,
            help='Use a faster TD3 training preset for iteration/debug runs')
    parser.add_argument('--max_episodes', type=int, default=None,
            help='Override total training episodes across all workers')
    parser.add_argument('--max_steps', type=int, default=None,
            help='Override max steps per episode')
    parser.add_argument('--update_itr', type=int, default=None,
            help='Override TD3 gradient updates per episode')
    parser.add_argument('--explore_steps', type=int, default=None,
            help='Override initial random exploration steps')
    parser.add_argument('--eval_interval', type=int, default=None,
            help='Override checkpoint interval in episodes')
    parser.add_argument('--action_range', type=float, default=None,
            help='Override policy action range (needed for some legacy checkpoints)')
    parser.add_argument('--debug_ft', action='store_true', default=False,
            help='Print UR5 wrist force/torque readings during --test evaluation')
    parser.add_argument('--debug_ft_every', type=int, default=10,
            help='Print force/torque debug output every N evaluation steps')
    parser.add_argument('--ft_log', type=str, default=None,
            help='Optional CSV path for all force/torque evaluation samples')
    ft_group = parser.add_mutually_exclusive_group()
    ft_group.add_argument('--use_ft', dest='ft_mode', action='store_const',
            const='raw', help='Add six raw, unscaled F/T values to the observation')
    ft_group.add_argument('--no_ft', dest='ft_mode', action='store_const',
            const='none', help='Exclude F/T values from the observation (default)')
    ft_group.add_argument('--normalized_ft', dest='ft_mode', action='store_const',
            const='normalized',
            help='Use legacy scaled F/T observations for old checkpoints')
    parser.set_defaults(ft_mode='none')
    parser.add_argument('--fixed_gripper_flexion', type=float, default=None,
            help='Optionally fix UR5 finger flexion; default uses movable fingers')
    parser.add_argument('--legacy_reward', action='store_true', default=False,
            help='Evaluate an old Panda or UR5 checkpoint with its legacy reward')
    args = parser.parse_args()

    if args.legacy_reward and args.env not in (
            'pandaopendoorfktactile', 'ur5opendoorfktactile'):
        parser.error('--legacy_reward is available only for Panda and UR5')
    if args.legacy_reward and not args.test:
        parser.error('--legacy_reward is evaluation-only; add --test')

    ROBOT_ENVS = ['pandaopendoorfktactile', 'ur5opendoorfktactile']
    envs = envs.registry.all()

    if args.env in ROBOT_ENVS:
        from environment import envs
        worker_env_kwargs = {}
        if args.env == 'ur5opendoorfktactile':
            worker_env_kwargs.update(
                fixed_gripper_flexion=args.fixed_gripper_flexion,
                ft_mode=args.ft_mode,
                legacy_reward=args.legacy_reward,
            )
        elif args.env == 'pandaopendoorfktactile':
            worker_env_kwargs.update(legacy_reward=args.legacy_reward)
        env_kwargs = dict(worker_env_kwargs)
        if args.render:
            env_kwargs.update(has_renderer=True, has_offscreen_renderer=False)
        env = envs[args.env](**env_kwargs)
    else:
        print('Environment {} not exists!'.format(args.env))
    print('Environment Name:', args.env)
    if args.env == 'ur5opendoorfktactile':
        print('UR5 F/T observation mode:', args.ft_mode)
        print('UR5 reward mode:',
              'legacy-20260831' if args.legacy_reward else 'current')
    elif args.env == 'pandaopendoorfktactile':
        print('Panda reward mode:',
              'legacy-20260714' if args.legacy_reward else 'current')
    print('Observation space: {}  Action space: {}'.format(env.observation_space, env.action_space))
    obs = np.asarray(env.reset(), dtype=np.float32)
    act = np.asarray(env.action_space.sample(), dtype=np.float32)

    print("obs shape:", obs.shape, "declared:", env.observation_space.shape)
    print("act shape:", act.shape, "declared:", env.action_space.shape)
    print("obs finite:", np.isfinite(obs).all(), "act finite:", np.isfinite(act).all())

    obs2, reward, done, info = env.step(act)
    obs2 = np.asarray(obs2, dtype=np.float32)

    print("step obs shape:", obs2.shape)
    print("step obs finite:", np.isfinite(obs2).all(), "reward finite:", np.isfinite(reward), "done:", done)

    assert obs.shape == env.observation_space.shape
    assert act.shape == env.action_space.shape
    assert obs2.shape == env.observation_space.shape
    assert np.isfinite(obs).all()
    assert np.isfinite(act).all()
    assert np.isfinite(obs2).all()
    assert np.isfinite(reward)

    if args.alg=='td3':
        td3_overrides = {}
        if args.fast:
            td3_overrides.update({
                'max_episodes': 3000,
                'max_steps': 500,
                'update_itr': 10,
                'explore_steps': 2000,
                'eval_interval': 250,
            })
        for key in ['max_episodes', 'max_steps', 'update_itr', 'explore_steps',
                    'eval_interval', 'action_range']:
            value = getattr(args, key)
            if value is not None:
                td3_overrides[key] = value
        train_td3(env, envs, args.train, args.test, args.finetune, args.path,
                  args.model_id, args.render, args.process, args.seed,
                  td3_overrides, debug_ft=args.debug_ft,
                  debug_ft_every=args.debug_ft_every, ft_log=args.ft_log,
                  worker_env_kwargs=worker_env_kwargs)
    else:
        print("Algorithm type is not implemented!")
