import numpy as np
from gym import spaces
from .syspath_robolite import *
from robosuite.environments.ur5_open_door import ur5opendoorfktactile as UR5OpenDoorBase

import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
from utils.common_func import Drawer


class ur5opendoorfktactile(UR5OpenDoorBase):
    def __init__(self, use_tactile=False, full_obs=False,
                 fixed_gripper_flexion=None, ft_mode="none", **kwargs):
        gripper_type = kwargs.pop("gripper_type", "RobotiqThreeFingerGripper")
        super().__init__(use_tactile=use_tactile, full_obs=full_obs,
                         ft_mode=ft_mode,
                         gripper_type=gripper_type,
                         fixed_gripper_flexion=fixed_gripper_flexion, **kwargs)

        self.tactile_dim = 30
        self.use_tactile = use_tactile
        self.full_obs = full_obs
        self.ft_mode = ft_mode
        self.name = self.__class__.__name__.lower()
        self.fixed_gripper = fixed_gripper_flexion is not None

        # Legacy training uses a seventh action for the movable gripper. Fixed
        # flexion remains available only as an explicit experimental option.
        act_dim = 6 if self.fixed_gripper else 7
        self.action_space = spaces.Box(
            low=-np.ones(act_dim, dtype=np.float32),
            high=np.ones(act_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.plot_tactile = False
        if self.plot_tactile and self.use_tactile:
            self.drawer = Drawer()
            self.drawer.start()

        self.last_obs = None
        self.dist_threshold = 0.03
        self.gripper_width_threshold = 0.07
        self.p_tactile_signal_flip = 0.005
        self.gripper_action_scale = 0.05
        sample_obs = np.asarray(self._final_obs(super().reset()), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones_like(sample_obs, dtype=np.float32),
            high=np.inf * np.ones_like(sample_obs, dtype=np.float32),
            dtype=np.float32,
        )

    def _final_obs(self, di):
        return np.asarray(di['task_state'], dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        expected_dim = 6 if self.fixed_gripper else 7
        if action.shape != (expected_dim,):
            raise ValueError(
                "UR5 action must contain exactly {} commands".format(expected_dim)
            )
        if self.fixed_gripper:
            # The wrapped environment internally expects a seventh gripper
            # command. Zero leaves the fixed position targets unchanged.
            norm_action = np.concatenate((action, np.zeros(1, dtype=np.float32)))
        else:
            # Compatibility mode for old policies that learned a seventh,
            # movable-gripper action.
            norm_action = np.concatenate(
                (action[:-1], action[-1:] * self.gripper_action_scale)
            )
        obs, reward, done, info = super().step(norm_action)

        if self.plot_tactile and self.use_tactile:
            self.drawer.add_value(obs['tactile'].copy())
            self.drawer.render()

        self.last_obs = self._final_obs(obs)

        if self.use_tactile:
            rands = np.random.uniform(0, 1, size=self.tactile_dim)
            idx = np.where(rands < self.p_tactile_signal_flip)
            obs_to_flip = self.last_obs[-self.tactile_dim:][idx]
            self.last_obs[-self.tactile_dim:][idx] = -1 * obs_to_flip + 1

        return self.last_obs, reward, done, info

    def reset(self, **kwargs):
        return self._final_obs(super().reset(**kwargs))
