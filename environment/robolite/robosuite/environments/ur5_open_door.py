from collections import OrderedDict
import numpy as np
import copy
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.ur5 import UR5Env
from gym.envs.robotics.rotations import quat2euler, quat_mul
from robosuite.utils import transform_utils as T
from robosuite.models.arenas import TableCabinetArena
from robosuite.models.tasks import TableTopTask, UniformRandomSamplerObjectSpecific
from robosuite.class_wrappers import change_dof


class ur5opendoorfktactile(change_dof(UR5Env, 7, 7)):
    minimal_offset = 1e-5
    parameters_spec = {
        **UR5Env.parameters_spec,
        'knob_friction':           [0.8, 1.0],
        'hinge_stiffness':         [0.1, 0.8],
        'hinge_damping':           [0.1, 0.3],
        'hinge_frictionloss':      [0.0, 1.0],
        'door_mass':               [50, 150],
        'knob_mass':               [2, 10],
        'table_size_0':            [0.8, 0.8+1e-5],
        'table_size_1':            [1.8, 1.8+1e-5],
        'table_size_2':            [0.9, 0.9+1e-5],
        'table_position_offset_x': [-0.05, 0.05],
        'table_position_offset_y': [-0.05, 0.05],
    }

    def reset_props(self, knob_friction=0.8, hinge_stiffness=0.1, hinge_damping=0.1,
                    hinge_frictionloss=0.1, door_mass=100., knob_mass=5.,
                    table_size_0=0.8, table_size_1=1.8, table_size_2=0.9,
                    table_position_offset_x=0.05, table_position_offset_y=0.05, **kwargs):
        self.knob_friction      = knob_friction
        self.hinge_stiffness    = hinge_stiffness
        self.hinge_damping      = hinge_damping
        self.hinge_frictionloss = hinge_frictionloss
        self.door_mass          = door_mass
        self.knob_mass          = knob_mass
        self.table_full_size    = (table_size_0, table_size_1, table_size_2)
        self.table_position_offset = np.array([table_position_offset_x, table_position_offset_y, 0.])
        super().reset_props(**kwargs)
        self.params_dict.update({
            'knob_friction': knob_friction, 'hinge_stiffness': hinge_stiffness,
            'hinge_damping': hinge_damping, 'hinge_frictionloss': hinge_frictionloss,
            'door_mass': door_mass, 'knob_mass': knob_mass,
            'table_size_0': table_size_0, 'table_size_1': table_size_1, 'table_size_2': table_size_2,
            'table_position_offset_x': table_position_offset_x,
            'table_position_offset_y': table_position_offset_y,
        })

    def __init__(self, use_object_obs=True, use_tactile=False, full_obs=False,
                 ft_mode="none",
                 legacy_reward=False,
                 reward_shaping=True, placement_initializer=None,
                 object_obs_process=True, fixed_gripper_flexion=0.1, **kwargs):
        self.use_object_obs     = use_object_obs
        self.use_tactile        = use_tactile
        self.full_obs           = full_obs
        if ft_mode not in ("none", "raw", "normalized"):
            raise ValueError(
                "ft_mode must be 'none', 'raw', or 'normalized'"
            )
        self.ft_mode            = ft_mode
        self.legacy_reward      = bool(legacy_reward)
        self.reward_shaping     = reward_shaping
        self.object_obs_process = object_obs_process
        if (fixed_gripper_flexion is not None and
                not 0.0 <= fixed_gripper_flexion <= np.pi):
            raise ValueError("fixed_gripper_flexion must be between 0 and pi")
        self.fixed_gripper_flexion = (
            None if fixed_gripper_flexion is None
            else float(fixed_gripper_flexion)
        )
        self.placement_initializer = placement_initializer or \
            UniformRandomSamplerObjectSpecific(
                x_ranges=[[-0.03,-0.02],[0.09,0.1]],
                y_ranges=[[-0.05,-0.04],[-0.05,-0.04]],
                ensure_object_boundary_in_range=False, z_rotation=None)
        self.table_full_size       = (0.8, 0.8, 0.8)
        self.table_friction        = (0., 0.005, 0.0001)
        self.knob_friction         = 0.8
        self.hinge_stiffness       = 0.1
        self.hinge_damping         = 0.1
        self.hinge_frictionloss    = 0.1
        self.table_position_offset = np.array([0., 0., 0.])
        self.door_mass             = 100.
        self.knob_mass             = 5.
        self.previous_door_open_angle = 0.0
        self.grasp_state = False
        self.grasp_rewarded = False
        super().__init__(gripper_visualization=True, **kwargs)

    def _get_reference(self):
        super()._get_reference()
        self.l_finger_geom_ids = [
            self.sim.model.geom_name2id(name)
            for name in self.gripper.left_finger_geoms
        ]
        self.r_finger_geom_ids = [
            self.sim.model.geom_name2id(name)
            for name in self.gripper.right_finger_geoms
        ]
        self.knob_geom_id = self.sim.model.geom_name2id("cabinet_knob")

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])
        self.mujoco_arena = TableCabinetArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction)
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()
        central_pos = np.array([-0.84, 0.5, -0.03]) + self.table_position_offset
        self.mujoco_arena.set_origin(central_pos)
        self.mujoco_objects = None
        self.model = TableTopTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects,
            initializer=self.placement_initializer, visual_objects=[])
        self.mujoco_arena.knob_geom.set("friction", str(self.knob_friction)+" 0 0")
        self.mujoco_arena.door_hinge.set("stiffness",    str(self.hinge_stiffness))
        self.mujoco_arena.door_hinge.set("damping",      str(self.hinge_damping))
        self.mujoco_arena.door_hinge.set("frictionloss", str(self.hinge_frictionloss))
        self.mujoco_arena.door_inertial.set("mass",      str(self.door_mass))
        self.mujoco_arena.knob_link_inertial.set("mass", str(self.knob_mass))

    def _reset_internal(self):
        super()._reset_internal()
        self.previous_door_open_angle = 0.0
        self.grasp_state = False
        self.grasp_rewarded = False
        self.sim.forward()
        ini_pos = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
        noise   = np.random.uniform(-0.0, 0.0, 6)
        self.sim.data.qpos[self._ref_joint_pos_indexes] = ini_pos + noise
        if self.fixed_gripper_flexion is None:
            # Legacy 7D policies start with an open, movable gripper.
            self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = 0.0
            self.sim.data.ctrl[-self.gripper.dof:] = 0.0
        else:
            # Keep the three fingers in a fixed, compliant half-closed pose. The
            # new policy controls only the six UR5 joints.
            movement = np.array([0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.])
            fixed_gripper_pose = movement * self.fixed_gripper_flexion
            self.sim.data.qpos[self._ref_gripper_joint_pos_indexes] = fixed_gripper_pose
            self.sim.data.ctrl[-self.gripper.dof:] = fixed_gripper_pose
        eef_rot = self.sim.data.get_body_xmat("right_hand").reshape((3, 3))
        self.world_rot_in_eef = copy.deepcopy(eef_rot.T)

    def get_gripper_state(self):
        qpos = np.abs(self.sim.data.qpos[self._ref_gripper_joint_pos_indexes])
        if self.gripper_type == "RobotiqThreeFingerGripper" and len(qpos) > 1:
            # The first and fifth joints are palm spread joints; the closing action
            # drives the finger flexion joints, so use those for grasp detection.
            flexion = np.delete(qpos, [0, 4]) if len(qpos) > 4 else qpos[1:]
            return float(np.max(flexion))
        return float(qpos[0])

    def reward(self, action=None):
        if self.legacy_reward:
            return self._legacy_reward(action)

        self.door_open_angle = abs(self.sim.data.get_joint_qpos("hinge0"))
        distance = np.linalg.norm(self.get_hand2knob_dist_vec())

        desired_orientation = np.array([0.0, 0.0, -np.pi / 2.0])
        finger_orientation = self.get_finger_ori()
        orientation_delta = np.concatenate((
            np.sin(desired_orientation) - np.sin(finger_orientation),
            np.cos(desired_orientation) - np.cos(finger_orientation),
        ))
        self.orientation_error = float(np.linalg.norm(orientation_delta))
        self.pose_error = float(distance + 0.25 * self.orientation_error)

        reward_dist = 0.0
        reward_ori = 0.0
        if self.door_open_angle < 0.02:
            reward_dist = -0.1 - 0.1 * np.tanh(distance)
            reward_ori = -0.1 - 0.1 * np.tanh(self.orientation_error)

        opening_progress = (
            self.door_open_angle - self.previous_door_open_angle)
        self.previous_door_open_angle = self.door_open_angle

        # Standing still receives an extra penalty. Closing is penalized both
        #by the negative 50*opening_progress term and by this fixed penalty.

        nonpositive_progress_penalty = ( -0.05 if opening_progress <= 0.0 else 0.0)

        reward = (50.0 * opening_progress + nonpositive_progress_penalty
                  + 0.4 * reward_dist
                  + 0.05 * reward_ori - 0.01)
        self.success = self._check_success()
        if self.success:
            reward += 100.0
        return reward

    def _legacy_reward(self, action=None):
        """Reward used to train the 2026-08-31 16:56 UR5 checkpoint."""
        self.door_open_angle = abs(
            self.sim.data.get_joint_qpos("hinge0"))
        distance = np.linalg.norm(self.get_hand2knob_dist_vec())
        desired_orientation = np.array([0.0, 0.0, -np.pi / 2.0])
        finger_orientation = self.get_finger_ori()
        orientation_delta = np.concatenate((
            np.sin(desired_orientation) - np.sin(finger_orientation),
            np.cos(desired_orientation) - np.cos(finger_orientation),
        ))
        self.orientation_error = float(np.linalg.norm(orientation_delta))
        self.pose_error = float(distance + 0.25 * self.orientation_error)

        reward_dist = 0.0
        reward_ori = 0.0
        if self.door_open_angle < 0.02:
            reward_dist = -0.1 - 0.1 * np.tanh(
                distance)
            reward_ori = -0.1 - 0.1 * np.tanh(
                self.orientation_error)

        reward_door_open = 0.5 * self.door_open_angle
        reward = (reward_door_open + 0.4 * reward_dist +
                  0.05 * reward_ori - 0.01)
        self.success = self._check_success()
        if self.success:
            reward += 100.0
        return reward

    def _check_success(self): return self.door_open_angle >= 1.55

    def _post_action(self, action):
        reward, _, info = super()._post_action(action)
        success = bool(getattr(self, "success", self._check_success()))
        timeout = (self.timestep >= self.horizon) and not self.ignore_done
        self.done = bool(success or timeout)
        info["success"] = success
        info["timeout"] = bool(timeout)
        info["door_open_angle"] = float(self.door_open_angle)
        info["distance_to_knob"] = float(
            np.linalg.norm(self.get_hand2knob_dist_vec())
        )
        info["orientation_error"] = float(self.orientation_error)
        info["pose_error"] = float(self.pose_error)
        return reward, self.done, info

    def _get_tactile_signals(self, contact_threshold=1e-3, Binary=True):
        tf = self.sim.data.sensordata[7::3]
        return np.where(np.abs(tf) > contact_threshold, 1, 0) if Binary else tf

    def _get_tactile_singals(self, *a, **k): return self._get_tactile_signals(*a, **k)

    def get_ft_sensor_data(self):
        """Return raw and observation-scaled wrist force/torque readings."""
        force_id = self.sim.model.sensor_name2id("ft_force")
        torque_id = self.sim.model.sensor_name2id("ft_torque")
        force_adr = self.sim.model.sensor_adr[force_id]
        torque_adr = self.sim.model.sensor_adr[torque_id]

        force = self.sim.data.sensordata[force_adr:force_adr + 3].copy()
        torque = self.sim.data.sensordata[torque_adr:torque_adr + 3].copy()
        raw = np.concatenate((force, torque))
        normalized = raw / np.array([5000., 5000., 5000., 500., 500., 500.])
        return {
            "force": force,
            "torque": torque,
            "raw": raw,
            "normalized": normalized,
        }

    def step(self, action): return super().step(action)

    def _get_observation(self):
        di = OrderedDict()
        if self.use_object_obs:
            di["eef_pos_in_world"]   = self.get_hand_pos()
            di["joint_pos_in_world"] = self.sim.data.qpos[self._ref_joint_pos_indexes]
            di["joint_vel_in_world"] = self.sim.data.qvel[self._ref_joint_vel_indexes]
            di["knob_pos_in_world"]  = self.get_knob_pos()
            di["knob_pos_to_eef"]    = di["knob_pos_in_world"] - di["eef_pos_in_world"]
            di["door_hinge_angle"]   = [self.sim.data.get_joint_qpos("hinge0")]
            di["gripper_width"]      = [self.get_gripper_state()]
            task_state = np.concatenate([di["joint_pos_in_world"], di["gripper_width"],
                                         di["knob_pos_to_eef"],    di["door_hinge_angle"]])
        di["task_state_no_tactile"] = task_state

        # New F/T experiments can select six raw or normalized sensor values.
        # Normalized mode also preserves compatibility with older 17D models.
        if self.ft_mode != "none":
            ft_data = self.get_ft_sensor_data()
            ft_signal = ft_data[self.ft_mode]
            di["ft_sensor"] = ft_signal
            task_state = np.concatenate((task_state, ft_signal))

        if self.use_tactile:
            di["tactile"] = self._get_tactile_signals()
            task_state    = np.concatenate((task_state, di["tactile"]))
        di["task_state"] = task_state
        return di

    def _check_contact(self): return False
    def world2eef(self, w): return self.world_rot_in_eef.dot(w)

    def get_finger_ori(self):
        # Generički - koristi zadnji link prvog prsta (radi za Robotiq 85 i 3-finger)
        if self.gripper_type == "RobotiqThreeFingerGripper":
            fq = self.sim.data.get_body_xquat("finger_1_link_3")
        else:
            fq = self.sim.data.get_body_xquat("robotiq_85_right_finger_tip_link")
        hq = self.sim.data.get_body_xquat("right_hand")
        return quat2euler(quat_mul(fq, hq))

    def get_hand_pos(self):
        if self.fixed_gripper_flexion is None:
            # Preserve the observation convention used by legacy 7D models.
            return self.sim.data.get_body_xpos("right_hand")
        # Unlike the wrist origin, this point follows the actual space between
        # the fingers when the wrist joints rotate.
        return self.sim.data.site_xpos[self.eef_site_id]

    def get_knob_pos(self):          return self.sim.data.get_geom_xpos("center_cabinet_knob")
    def get_hand2knob_dist_vec(self): return self.get_hand_pos() - self.get_knob_pos()

    def _gripper_visualization(self):
        if self.gripper_visualization:
            self.sim.model.site_rgba[self.eef_site_id] = np.zeros(4)
