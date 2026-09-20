import numpy as np
from robosuite.models.robots.robot import Robot
from robosuite.utils.mjcf_utils import xml_path_completion, array_to_string


class UR5(Robot):
    def __init__(self):
        super().__init__(xml_path_completion("robots/ur5/robot.xml"))
        self.bottom_offset = np.array([0, 0, -0.850])
        self.set_joint_damping()
        self._model_name = "ur5"
        self._init_qpos = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])

    def set_base_xpos(self, pos):
        node = self.worldbody.find("./body[@name='base_link']")
        node.set("pos", array_to_string(pos - self.bottom_offset))

    def set_joint_damping(self, damping=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01))):
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("damping", array_to_string(np.array([damping[i]])))

    def set_joint_frictionloss(self, friction=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01))):
        body = self._base_body
        for i in range(len(self._link_body)):
            body = body.find("./body[@name='{}']".format(self._link_body[i]))
            joint = body.find("./joint[@name='{}']".format(self._joints[i]))
            joint.set("frictionloss", array_to_string(np.array([friction[i]])))

    @property
    def dof(self): return 6

    @property
    def joints(self): return ["joint{}".format(x) for x in range(1, 7)]

    @property
    def init_qpos(self): return self._init_qpos

    @property
    def contact_geoms(self): return ["link{}_collision".format(x) for x in range(1, 7)]

    @property
    def _base_body(self): return self.worldbody.find("./body[@name='base_link']")

    @property
    def _link_body(self):
        return ["shoulder_link", "upper_arm_link", "forearm_link",
                "wrist_1_link", "wrist_2_link", "wrist_3_link"]

    @property
    def _joints(self): return ["joint1","joint2","joint3","joint4","joint5","joint6"]