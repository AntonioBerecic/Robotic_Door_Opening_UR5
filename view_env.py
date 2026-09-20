import sys; sys.path.insert(0, '.')
import time
import numpy as np

from environment.ur5opendoorfktactile import ur5opendoorfktactile


env = ur5opendoorfktactile(
    gripper_type='RobotiqThreeFingerGripper',
    has_renderer=True,
    render_visual_mesh=True,
)

obs = env.reset()
env.viewer.set_camera(camera_id=0)

sim = env.sim


# =========================================================
# ARM DEFINITIONS
# =========================================================
arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

print("=== ARM JOINT INFO ===")
arm_joint_meta = {}
for jn in arm_joint_names:
    jid = sim.model.joint_name2id(jn)
    bid = sim.model.jnt_bodyid[jid]
    qadr = sim.model.jnt_qposadr[jid]
    arm_joint_meta[jn] = {"jid": jid, "bid": bid, "qadr": qadr}
    print(
        jn,
        "body=", sim.model.body_id2name(bid),
        "axis(local)=", sim.model.jnt_axis[jid],
        "pos(local)=", sim.model.jnt_pos[jid],
        "qpos_adr=", qadr,
    )


# =========================================================
# JOINT / ACTUATOR DUMP
# =========================================================
print("\n=== ALL JOINTS ===")
for i in range(sim.model.njnt):
    print(i, sim.model.joint_id2name(i))

print("\n=== ALL ACTUATORS ===")
for i in range(sim.model.nu):
    print(i, sim.model.actuator_id2name(i), "trnid=", sim.model.actuator_trnid[i])


# =========================================================
# GRIPPER DEFINITIONS
# =========================================================
gripper_joint_names = [
    "palm_finger_1_joint",
    "finger_1_joint_1",
    "finger_1_joint_2",
    "finger_1_joint_3",
    "palm_finger_2_joint",
    "finger_2_joint_1",
    "finger_2_joint_2",
    "finger_2_joint_3",
    "finger_middle_joint_1",
    "finger_middle_joint_2",
    "finger_middle_joint_3",
]

gripper_act_names = [
    "gripper_palm_finger_1_joint",
    "gripper_finger_1_joint_1",
    "gripper_finger_1_joint_2",
    "gripper_finger_1_joint_3",
    "gripper_palm_finger_2_joint",
    "gripper_finger_2_joint_1",
    "gripper_finger_2_joint_2",
    "gripper_finger_2_joint_3",
    "gripper_finger_middle_joint_1",
    "gripper_finger_middle_joint_2",
    "gripper_finger_middle_joint_3",
]

print("\n=== GRIPPER JOINT INFO ===")
gripper_joint_meta = {}
for jn in gripper_joint_names:
    jid = sim.model.joint_name2id(jn)
    qadr = sim.model.jnt_qposadr[jid]
    rng = sim.model.jnt_range[jid] if sim.model.jnt_limited[jid] else None
    gripper_joint_meta[jn] = {"jid": jid, "qadr": qadr}
    print(
        jn,
        "axis(local)=", sim.model.jnt_axis[jid],
        "qpos_adr=", qadr,
        "range=", rng
    )

print("\n=== GRIPPER ACTUATOR INFO ===")
gripper_act_meta = {}
for an in gripper_act_names:
    aid = sim.model.actuator_name2id(an)
    ctrlrange = sim.model.actuator_ctrlrange[aid]
    trnid = sim.model.actuator_trnid[aid]
    gripper_act_meta[an] = {"aid": aid}
    print(
        an,
        "id=", aid,
        "ctrlrange=", ctrlrange,
        "trnid=", trnid
    )


# =========================================================
# HELPERS
# =========================================================
def zero_ctrl():
    sim.data.ctrl[:] = 0


def render_steps(n=180, dt=0.01):
    for _ in range(n):
        sim.step()
        env.render()
        time.sleep(dt)


def set_arm_q(q):
    qpos = sim.data.qpos.copy()
    for i, jn in enumerate(arm_joint_names):
        qpos[arm_joint_meta[jn]["qadr"]] = q[i]
    sim.data.qpos[:] = qpos
    sim.forward()


def print_arm_body_states(tag=""):
    watch_bodies = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "right_hand",
    ]
    print(f"\n--- ARM BODY STATES: {tag} ---")
    for bn in watch_bodies:
        bid = sim.model.body_name2id(bn)
        xpos = sim.data.body_xpos[bid].copy()
        print(bn, "xpos=", np.round(xpos, 5))


def print_gripper_joint_states(tag=""):
    print(f"\n--- GRIPPER JOINT STATES: {tag} ---")
    for jn in gripper_joint_names:
        qadr = gripper_joint_meta[jn]["qadr"]
        print(jn, "qpos=", round(float(sim.data.qpos[qadr]), 5))


def hold_pose(label, q, steps=160):
    print(f"\n==============================")
    print("POSE:", label)
    print("q =", np.round(q, 4))
    set_arm_q(q)
    zero_ctrl()
    print_arm_body_states(label)
    render_steps(steps)


def set_actuator_ctrl(act_name, value):
    aid = gripper_act_meta[act_name]["aid"]
    lo, hi = sim.model.actuator_ctrlrange[aid]
    sim.data.ctrl[aid] = np.clip(value, lo, hi)


def actuator_limits(act_name):
    aid = gripper_act_meta[act_name]["aid"]
    lo, hi = sim.model.actuator_ctrlrange[aid]
    return float(lo), float(hi)


def set_group_ctrl(act_names, alpha):
    """
    alpha in [-1, 1]
    alpha > 0  -> ide prema pozitivnom kraju ctrlrange
    alpha < 0  -> ide prema negativnom kraju ctrlrange
    """
    zero_ctrl()
    for an in act_names:
        lo, hi = actuator_limits(an)
        if alpha >= 0:
            val = alpha * hi
        else:
            val = (-alpha) * lo
        set_actuator_ctrl(an, val)


def print_group_ctrls(act_names):
    for an in act_names:
        aid = gripper_act_meta[an]["aid"]
        print(an, "ctrl=", round(float(sim.data.ctrl[aid]), 5))


# =========================================================
# ARM TEST POSES
# =========================================================
arm_poses = [
    ("neutral",      np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("joint1 +0.6",  np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("joint1 -0.6",  np.array([-0.6, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("joint2 +0.6",  np.array([0.0, 0.6, 0.0, 0.0, 0.0, 0.0])),
    ("joint2 -0.6",  np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0])),
    ("joint3 +0.6",  np.array([0.0, 0.0, 0.6, 0.0, 0.0, 0.0])),
    ("joint3 -0.6",  np.array([0.0, 0.0, -0.6, 0.0, 0.0, 0.0])),
    ("joint4 +0.6",  np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.0])),
    ("joint4 -0.6",  np.array([0.0, 0.0, 0.0, -0.6, 0.0, 0.0])),
    ("joint5 +0.6",  np.array([0.0, 0.0, 0.0, 0.0, 0.6, 0.0])),
    ("joint5 -0.6",  np.array([0.0, 0.0, 0.0, 0.0, -0.6, 0.0])),
    ("joint6 +0.6",  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6])),
    ("joint6 -0.6",  np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.6])),
]


# =========================================================
# GRIPPER GROUPS
# =========================================================
finger1_acts = [
    "gripper_finger_1_joint_1",
    "gripper_finger_1_joint_2",
    "gripper_finger_1_joint_3",
]

finger2_acts = [
    "gripper_finger_2_joint_1",
    "gripper_finger_2_joint_2",
    "gripper_finger_2_joint_3",
]

fingerm_acts = [
    "gripper_finger_middle_joint_1",
    "gripper_finger_middle_joint_2",
    "gripper_finger_middle_joint_3",
]

spread_acts = [
    "gripper_palm_finger_1_joint",
    "gripper_palm_finger_2_joint",
]

all_close_acts = finger1_acts + finger2_acts + fingerm_acts


# =========================================================
# GRIPPER TEST FUNCTIONS
# =========================================================
def test_single_gripper_actuator(act_name, label, alpha=0.8, steps=180):
    print(f"\n==============================")
    print("GRIPPER TEST:", label)
    zero_ctrl()
    set_group_ctrl([act_name], alpha)
    print_group_ctrls([act_name])
    render_steps(steps)
    print_gripper_joint_states(label)


def test_gripper_group(act_names, label, alpha=0.8, steps=220):
    print(f"\n==============================")
    print("GRIPPER GROUP TEST:", label)
    set_group_ctrl(act_names, alpha)
    print_group_ctrls(act_names)
    render_steps(steps)
    print_gripper_joint_states(label)


# =========================================================
# MAIN LOOP
# =========================================================
while True:
    # ---------------------------------
    # ARM
    # ---------------------------------
    hold_pose("neutral", np.array([0, 0, 0, 0, 0, 0]), steps=120)

    for label, q in arm_poses:
        hold_pose(label, q, steps=140)

    # ---------------------------------
    # GRIPPER - SINGLE SEGMENT DEBUG
    # ---------------------------------
    test_single_gripper_actuator("gripper_finger_1_joint_1", "finger1 segment1 close", alpha=+0.8)
    test_single_gripper_actuator("gripper_finger_1_joint_1", "finger1 segment1 open",  alpha=-0.8)

    test_single_gripper_actuator("gripper_finger_1_joint_2", "finger1 segment2 close", alpha=+0.8)
    test_single_gripper_actuator("gripper_finger_1_joint_2", "finger1 segment2 open",  alpha=-0.8)

    test_single_gripper_actuator("gripper_finger_1_joint_3", "finger1 segment3 close", alpha=+0.8)
    test_single_gripper_actuator("gripper_finger_1_joint_3", "finger1 segment3 open",  alpha=-0.8)

    # ---------------------------------
    # GRIPPER - WHOLE HAND
    # ---------------------------------
    test_gripper_group(all_close_acts, "close all fingers", alpha=+0.8, steps=240)
    test_gripper_group(all_close_acts, "open all fingers",  alpha=-0.8, steps=240)

    # ---------------------------------
    # GRIPPER - SPREAD
    # ---------------------------------
    test_gripper_group(spread_acts, "spread side fingers",   alpha=+0.8, steps=220)
    test_gripper_group(spread_acts, "unspread side fingers", alpha=-0.8, steps=220)

    # ---------------------------------
    # SMALL PAUSE
    # ---------------------------------
    zero_ctrl()
    render_steps(120)