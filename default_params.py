def get_hyperparams(env_name):
    if 'pandaopendoorfk' in env_name or 'ur5opendoorfk' in env_name:
        hyperparams_dict={
        'alg_name': 'td3',
        # 'max_steps': 300,
        'max_steps': 1000,
        'max_episodes': 10000,
        'action_range': 0.15,  # Increased from 0.05 to allow more aggressive exploration
        # 'action_range': 0.1,  # on joint
        'batch_size': 128,
        'explore_steps': 5000,  # Add random exploration phase to bootstrap learning
        'update_itr': 50,  # iterative update
        'eval_interval': 500,
        'explore_noise_scale': 0.1,  # Increase exploration noise for better coverage
        'eval_noise_scale': 0.02,  # noisy evaluation trick
        'reward_scale': None, # keep shaped reward magnitudes meaningful
        'gamma': 0.99, # reward discount
        'soft_tau': 1e-2,  # soft udpate coefficient
        'hidden_dim': 512,
        'noise_decay': 0.9999, # decaying exploration noise
        'policy_target_update_interval': 5, # delayed update
        'q_lr': 3e-4,
        'policy_lr': 3e-4,
        'replay_buffer_size': 200000,
        'randomized_params': [ 'table_position_offset_x','table_position_offset_y'],
        #'randomized_params':[],
        #'randomized_params': ['knob_friction', 'hinge_stiffness', 'hinge_damping', 'hinge_frictionloss', 'door_mass', 'knob_mass', 'table_position_offset_x', 'table_position_offset_y'],  # choose in: 'all', None, or a list of parameter keys
        'deterministic': True,
        }

    else:
        raise NotImplementedError

    print('Hyperparameters: ', hyperparams_dict)
    return hyperparams_dict
