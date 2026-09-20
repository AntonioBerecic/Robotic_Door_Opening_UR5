from collections import OrderedDict
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv
from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import UR5


class UR5Env(MujocoEnv):
    parameters_spec = {
        'link1_mass': [3.5, 4.0], 'link2_mass': [8.0, 9.0], 'link3_mass': [2.0, 2.5],
        'link4_mass': [1.0, 1.5], 'link5_mass': [1.0, 1.5], 'link6_mass': [0.1, 0.3],
        'joint1_damping': [0.06, 0.14], 'joint2_damping': [0.06, 0.14],
        'joint3_damping': [0.06, 0.14], 'joint4_damping': [0.06, 0.14],
        'joint5_damping': [0.06, 0.14], 'joint6_damping': [0.006, 0.014],
        'joint1_armature': [0.0, 0.5], 'joint2_armature': [0.0, 0.5],
        'joint3_armature': [0.0, 0.5], 'joint4_armature': [0.0, 0.5],
        'joint5_armature': [0.0, 0.5], 'joint6_armature': [0.0, 0.5],
        'actuator_velocity_joint1_kv': [30.0, 50.0],
        'actuator_velocity_joint2_kv': [30.0, 50.0],
        'actuator_velocity_joint3_kv': [30.0, 50.0],
        'actuator_velocity_joint4_kv': [30.0, 50.0],
        'actuator_velocity_joint5_kv': [30.0, 50.0],
        'actuator_velocity_joint6_kv': [30.0, 50.0],
    }

    def __init__(self, gripper_type="RobotiqGripper", gripper_visualization=False,
                 use_indicator_object=False, has_renderer=False, has_offscreen_renderer=False,
                 render_collision_mesh=False, render_visual_mesh=True, control_freq=20,
                 horizon=1000, ignore_done=False, use_camera_obs=False, camera_name="frontview",
                 camera_height=256, camera_width=256, camera_depth=False):
        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object
        self.params_dict = {}
        super().__init__(has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh, render_visual_mesh=render_visual_mesh,
            control_freq=control_freq, horizon=horizon, ignore_done=ignore_done,
            use_camera_obs=use_camera_obs, camera_name=camera_name,
            camera_height=camera_height, camera_width=camera_width, camera_depth=camera_depth)

    def reset_props(self, **kwargs):
        defaults = {
            'link1_mass': 3.7, 'link2_mass': 8.393, 'link3_mass': 2.275,
            'link4_mass': 1.219, 'link5_mass': 1.219, 'link6_mass': 0.1879,
            'joint1_damping': 0.1, 'joint2_damping': 0.1, 'joint3_damping': 0.1,
            'joint4_damping': 0.1, 'joint5_damping': 0.1, 'joint6_damping': 0.01,
            'joint1_armature': 0.0, 'joint2_armature': 0.0, 'joint3_armature': 0.0,
            'joint4_armature': 0.0, 'joint5_armature': 0.0, 'joint6_armature': 0.0,
            'actuator_velocity_joint1_kv': 40.0, 'actuator_velocity_joint2_kv': 40.0,
            'actuator_velocity_joint3_kv': 40.0, 'actuator_velocity_joint4_kv': 40.0,
            'actuator_velocity_joint5_kv': 40.0, 'actuator_velocity_joint6_kv': 40.0,
        }
        assert all(k in defaults for k in kwargs), "Invalid parameter key"
        self.params_dict.update(defaults, **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot = UR5()
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)
        if bool(self.params_dict):
            p = self.params_dict
            link_names = ["shoulder_link","upper_arm_link","forearm_link",
                          "wrist_1_link","wrist_2_link","wrist_3_link"]
            link_keys  = ["link1","link2","link3","link4","link5","link6"]
            for ln, lk in zip(link_names, link_keys):
                lie = self.mujoco_robot.root.find(".//body[@name='{}']".format(ln)).find("./inertial[@mass]")
                lie.set("mass", str(p["{}_mass".format(lk)]))
            for jn in self.mujoco_robot._joints:
                je = self.mujoco_robot.root.find(".//joint[@name='{}']".format(jn))
                je.set("damping",   str(p["{}_damping".format(jn)]))
                je.set("armature",  str(p["{}_armature".format(jn)]))
                av = self.mujoco_robot.root.find(".//velocity[@joint='{}']".format(jn))
                av.set("kv", str(p["actuator_velocity_{}_kv".format(jn)]))

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos
        if self.has_gripper:
            self.sim.data.qpos[self._ref_joint_gripper_actuator_indexes] = self.gripper.init_qpos

    def _get_reference(self):
        super()._get_reference()
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints]
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints]
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints]
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.joint_name2id(j)
                for j in self.gripper.joints
                if j in self.sim.model.joint_names]
        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(a)
            for a in self.sim.model.actuator_names if a.startswith("vel")]
        self.eef_site_id     = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def _pre_action(self, action, rescale=False):
        assert len(action) == self.dof
        low, high = self.action_spec
        action = np.clip(action, low, high)
        if self.has_gripper:
            arm     = action[:self.mujoco_robot.dof]
            gripper = self.gripper.format_action(
                action[self.mujoco_robot.dof:self.mujoco_robot.dof + self.gripper.dof])
            action  = np.concatenate([arm, gripper])
        arm_ctrl = action[:self.mujoco_robot.dof]
        gripper_ctrl_raw = action[self.mujoco_robot.dof:]
        self.sim.data.ctrl[:self.mujoco_robot.dof] = arm_ctrl
        # Ako gripper akcija ima manji dof od aktuatora, broadcast zadnje vrijednosti
        if len(gripper_ctrl_raw) < self.gripper.dof:
            import numpy as _np
            gripper_ctrl_full = _np.tile(gripper_ctrl_raw[-1], self.gripper.dof)
        else:
            gripper_ctrl_full = gripper_ctrl_raw[:self.gripper.dof]
        self.sim.data.ctrl[self.mujoco_robot.dof:] += gripper_ctrl_full
        self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = \
            self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._gripper_visualization()
        return ret

    def _get_observation(self):
        di = super()._get_observation()
        di["joint_pos"] = np.array([self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes])
        di["joint_vel"] = np.array([self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes])
        robot_states = [np.sin(di["joint_pos"]), np.cos(di["joint_pos"]), di["joint_vel"]]
        if self.has_gripper:
            di["gripper_qpos"] = np.array([self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes])
            di["eef_pos"]  = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(self.sim.data.get_body_xquat("right_hand"), to="xyzw")
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])
        di["robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def action_spec(self):
        return np.ones(self.dof) * -1., np.ones(self.dof) * 1.

    @property
    def dof(self):
        return self.mujoco_robot.dof + (self.gripper.dof if self.has_gripper else 0)

    def _gripper_visualization(self):
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self): return False

    @property
    def _joint_positions(self): return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self): return self.sim.data.qvel[self._ref_joint_vel_indexes]