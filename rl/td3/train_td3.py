import os
import torch
# Limit PyTorch threads to reduce CPU/BLAS contention and RAM usage
torch.set_num_threads(1)
import queue
import random
import datetime
import numpy as np
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager

from rl.buffers import ReplayBuffer
from utils.load_params import load_params
from utils.common_func import rand_params
from rl.td3.td3 import TD3_Trainer, collector_worker
from rl.policy_networks import DPG_PolicyNetwork
# from rl.td3.td3_test import TD3_Trainer, worker, cpu_worker


def _validate_evaluation_state(state, env, step):
    """Reject invalid observations without treating raw F/T units as state blow-up."""
    state = np.asarray(state)
    assert np.all(np.isfinite(state)), f"Non-finite state at step {step}"

    # With --use_ft the final six values are unscaled forces [N] and torques
    # [Nm]. Forces above 100 N are possible during contact and must not be
    # compared with the limit intended for joint/pose observations.
    checked_state = state[:-6] if getattr(env, 'ft_mode', 'none') == 'raw' else state
    max_value = float(np.max(np.abs(checked_state)))
    assert max_value < 100, (
        f"State explosion at step {step}: max non-raw-F/T value={max_value:.2f}"
    )


def train_td3(env, envs, train, test, finetune, path, model_id, render, process,
              seed, param_overrides=None, debug_ft=False, debug_ft_every=10,
              ft_log=None, worker_env_kwargs=None):
    torch.manual_seed(seed)  # Reproducibility
    np.random.seed(seed)
    random.seed(seed)

    # hyper-parameters for RL training
    try: # custom env
        env_name = env.name 
    except: # gym env
        env_name = env.spec.id

    num_workers = process # or: mp.cpu_count()
    worker_env_kwargs = dict(worker_env_kwargs or {})
    if num_workers < 1:
        raise ValueError("process count must be at least 1")
    # Seconds avoid two runs with the same seed overwriting one another.
    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = './data/weights/{}{}'.format(prefix, seed)
    if not os.path.exists(model_path) and train:
        os.makedirs(model_path)
    print('Model Path: ', model_path)

    # load other default parameters
    param_names = ['max_steps', 'max_episodes', 'action_range', 'batch_size', 'explore_steps', 'update_itr', 'eval_interval',
        'explore_noise_scale', 'eval_noise_scale', 'reward_scale', 'gamma', 'soft_tau', 'hidden_dim', 'noise_decay',
        'policy_target_update_interval', 'q_lr', 'policy_lr', 'replay_buffer_size', 'randomized_params', 'deterministic']
    param_values = dict(zip(param_names, load_params('td3', env_name, param_names)))
    if param_overrides:
        unknown = set(param_overrides) - set(param_values)
        if unknown:
            raise ValueError("Unknown TD3 parameter override(s): {}".format(sorted(unknown)))
        param_values.update(param_overrides)
        print('TD3 parameter overrides: ', param_overrides)
    [max_steps, max_episodes, action_range, batch_size, explore_steps, update_itr, eval_interval, explore_noise_scale, eval_noise_scale, reward_scale,
        gamma, soft_tau, hidden_dim, noise_decay, policy_target_update_interval, q_lr, policy_lr, replay_buffer_size, randomized_params, DETERMINISTIC] = \
        [param_values[name] for name in param_names]
    if not action_range:
        action_range = env.action_space.high[0]  # mujoco env gives the range of action and it is symmetric
    # Training collectors need a process-safe proxy. Testing only needs a local
    # placeholder buffer and should not leave a manager process behind.
    manager = None
    if train:
        mp.set_start_method('spawn', force=True)
        BaseManager.register('ReplayBuffer', ReplayBuffer)
        manager = BaseManager()
        manager.start()
        replay_buffer = manager.ReplayBuffer(replay_buffer_size)
    else:
        replay_buffer = ReplayBuffer(replay_buffer_size)

    action_space = env.action_space
    state_space = env.observation_space

    machine_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    print('Learner device:', machine_type)
    td3_trainer=TD3_Trainer(replay_buffer, state_space, action_space, hidden_dim, q_lr, policy_lr,\
        policy_target_update_interval=policy_target_update_interval, action_range=action_range, machine_type=machine_type )

    if train: 
        if finetune is True:
            td3_trainer.load_model('./data/weights/'+ path +'/{}_td3'.format(model_id))
        # Collectors only read this shared CPU policy. The learner periodically
        # publishes its latest CUDA/CPU policy into it under a process lock.
        collector_policy = DPG_PolicyNetwork(
            state_space, action_space, hidden_dim, action_range, machine_type='cpu')
        collector_policy.load_state_dict({
            key: value.detach().cpu()
            for key, value in td3_trainer.policy_net.state_dict().items()
        })
        collector_policy.share_memory()
        policy_lock = mp.Lock()
        episode_counter = mp.Value('i', 0)

        rewards_queue=mp.Queue()  # used for get rewards from all processes and plot the curve
        processes=[]
        rewards=[]
        success = []

        base_episodes = max_episodes // num_workers
        extra_episodes = max_episodes % num_workers
        worker_episode_counts = [
            base_episodes + (1 if i < extra_episodes else 0)
            for i in range(num_workers)
        ]
        print('Total training episodes: ', max_episodes, '| Workers: ', num_workers,
              '| Episodes per worker: ', worker_episode_counts)

        for i, worker_max_episodes in enumerate(worker_episode_counts):
            if worker_max_episodes <= 0:
                continue
            process = mp.Process(target=collector_worker, args=(
                i, collector_policy, policy_lock, episode_counter, envs, env_name,
                rewards_queue, replay_buffer, worker_max_episodes, max_steps,
                explore_steps, num_workers, noise_decay, explore_noise_scale,
                action_range, render, randomized_params, worker_env_kwargs,
                seed))
            process.daemon=True  # all processes closed when the main stops
            processes.append(process)

        [p.start() for p in processes]
        expected_rewards = sum(worker_episode_counts)
        while len(rewards) < expected_rewards:  # keep getting episode rewards from workers
            try:
                r = rewards_queue.get(timeout=30)
            except queue.Empty:
                if not any(p.is_alive() for p in processes):
                    break
                continue
            episode_reward, episode_success, _episode_steps = r
            rewards.append(episode_reward)
            success.append(episode_success)

            # A single learner performs every optimizer step. This preserves one
            # coherent TD3 policy/Q/target set regardless of collector count.
            if replay_buffer.get_length() >= batch_size:
                for _ in range(update_itr):
                    td3_trainer.update(
                        batch_size, eval_noise_scale=eval_noise_scale,
                        reward_scale=reward_scale, gamma=gamma, soft_tau=soft_tau)
                with policy_lock:
                    collector_policy.load_state_dict({
                        key: value.detach().cpu()
                        for key, value in td3_trainer.policy_net.state_dict().items()
                    })

            if len(rewards)%20==0 and len(rewards)>0:
                # plot(rewards)
                np.save('log/'+prefix+'td3_rewards', rewards)
                np.save('log/'+prefix+'td3_success', success)
            if len(rewards) % eval_interval == 0:
                td3_trainer.save_model(model_path + '/{}_td3'.format(len(rewards)))
                print("Saved checkpoint:", model_path + '/{}_td3'.format(len(rewards)))

        [p.join() for p in processes]  # finished at the same time
        failed = [p.exitcode for p in processes if p.exitcode not in (0, None)]
        if failed or len(rewards) != expected_rewards:
            manager.shutdown()
            raise RuntimeError(
                "TD3 collector failure: exit codes={}, received episodes={}/{}"
                .format(failed, len(rewards), expected_rewards))

        td3_trainer.save_model(model_path)
        np.save('log/'+prefix+'td3_rewards', rewards)
        np.save('log/'+prefix+'td3_success', success)
        manager.shutdown()
        
    if test:
        import csv
        import time
        if debug_ft_every < 1:
            raise ValueError("debug_ft_every must be at least 1")
        if (debug_ft or ft_log) and not hasattr(env, 'get_ft_sensor_data'):
            raise ValueError("Force/torque debug is only supported by the UR5 environment")

        ft_file = None
        ft_writer = None
        if ft_log:
            log_parent = os.path.dirname(os.path.abspath(ft_log))
            os.makedirs(log_parent, exist_ok=True)
            ft_file = open(ft_log, 'w', newline='')
            ft_writer = csv.writer(ft_file)
            ft_writer.writerow([
                'episode', 'step', 'fx_N', 'fy_N', 'fz_N',
                'tx_Nm', 'ty_Nm', 'tz_Nm', 'force_magnitude_N',
                'torque_magnitude_Nm', 'fx_obs', 'fy_obs', 'fz_obs',
                'tx_obs', 'ty_obs', 'tz_obs', 'reward',
                'door_hinge_angle_rad', 'done', 'success',
            ])
            print('F/T CSV log:', os.path.abspath(ft_log))

        model_path = './data/weights/'+ path +'/{}_td3'.format(str(model_id))
        print('Load model from: ', model_path)
        td3_trainer.load_model(model_path)
        # print(env.action_space.high, env.action_space.low)

        no_DR = False
        dist_threshold = 0.02
        dist_threshold_max = 0.07
        if no_DR:
            randomized_params=None
        print(randomized_params)
        evaluation_rewards = []
        evaluation_successes = []
        evaluation_lengths = []
        evaluation_final_angles = []
        try:
            if debug_ft:
                print('F/T debug units: force=N, torque=Nm; obs=raw/[5000, 500].')
                print('Values are expressed in the ft_sensor_site coordinate frame.')
            for eps in range(10):
                if not no_DR:
                    param_dict, param_vec = rand_params(env, randomized_params)
                    state = env.reset(**param_dict)
                    print('Randomized parameters value: ', param_dict)
                else:
                    state = env.reset()

                if render:
                    env.render()
                episode_reward = 0
                episode_success = False
                if render:
                    time.sleep(1)
                s_list = []
                for step in range(max_steps):
                    # Detect state explosion instead of silently clipping
                    _validate_evaluation_state(state, env, step)
                    action = td3_trainer.policy_net.get_action(state, noise_scale=0.0)

                    next_state, reward, done, info = env.step(action)
                    episode_success = (
                        episode_success or bool(info.get('success', False)))
                    if render:
                        env.render()
                        time.sleep(0.005)
                    episode_reward += reward

                    if debug_ft or ft_writer:
                        ft = env.get_ft_sensor_data()
                        force_mag = float(np.linalg.norm(ft['force']))
                        torque_mag = float(np.linalg.norm(ft['torque']))
                        hinge = float(env.sim.data.get_joint_qpos('hinge0'))
                        if debug_ft and step % debug_ft_every == 0:
                            print(
                                'FT ep={:02d} step={:04d} | '
                                'F[N]=[{:+8.2f} {:+8.2f} {:+8.2f}] |F|={:8.2f} | '
                                'T[Nm]=[{:+7.2f} {:+7.2f} {:+7.2f}] |T|={:7.2f} | '
                                'obs={}'.format(
                                    eps, step, *ft['force'], force_mag,
                                    *ft['torque'], torque_mag,
                                    np.array2string(ft['normalized'], precision=5,
                                                    suppress_small=True)))
                        if ft_writer:
                            ft_writer.writerow([
                                eps, step, *ft['raw'], force_mag, torque_mag,
                                *ft['normalized'], reward, hinge, bool(done),
                                bool(info.get('success', False)),
                            ])

                    state = next_state
                    s_list.append(state)
                    if done:
                        break

                episode_length = step + 1
                final_door_angle = abs(float(
                    env.sim.data.get_joint_qpos('hinge0')))
                print(
                    'Episode:', eps,
                    '| Episode Reward:', episode_reward,
                    '| Success:', episode_success,
                    '| Steps:', episode_length,
                    '| Final door angle: {:.4f} rad ({:.2f} deg)'.format(
                        final_door_angle, np.degrees(final_door_angle)))
                evaluation_rewards.append(episode_reward)
                evaluation_successes.append(episode_success)
                evaluation_lengths.append(episode_length)
                evaluation_final_angles.append(final_door_angle)
                # np.save('data/s.npy', s_list)

            episode_count = len(evaluation_rewards)
            success_count = int(np.sum(evaluation_successes))
            mean_final_angle = float(np.mean(evaluation_final_angles))
            print('\n=== EVALUATION SUMMARY ===')
            print('Evaluation episodes: {}'.format(episode_count))
            print('Average episode reward: {:.6f}'.format(
                np.mean(evaluation_rewards)))
            print('Successful openings: {}/{} ({:.2f}%)'.format(
                success_count, episode_count,
                100.0 * success_count / episode_count))
            print('Average episode length: {:.2f} steps'.format(
                np.mean(evaluation_lengths)))
            print('Average final door angle: {:.4f} rad ({:.2f} deg)'.format(
                mean_final_angle, np.degrees(mean_final_angle)))
        finally:
            if ft_file is not None:
                ft_file.close()
