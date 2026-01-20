import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------- Model ----------------
MODEL_PATH = r"C:\Users\Nagham\Documents\mujoco_models\ball.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# ---------------- IDs ----------------
ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
joint_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")

# ---------------- Simulation settings ----------------
sim_time = 5.0
dt = model.opt.timestep
steps = int(sim_time / dt)

# Logs
time_log = []
joint_angle_log = []
ball_rel_x_log = []

# Reset simulation
mujoco.mj_resetData(model, data)

# ---------------- Simulation loop ----------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(steps):
        t = i * dt

        # Bidirectional smooth motion
        if 1.0 < t < 2.0:
            data.ctrl[1] = 0.8 * (t - 1.0)
        elif 2.0 <= t < 3.0:
            data.ctrl[1] = 0.8 * (3.0 - t)
        elif 3.0 < t < 4.0:
            data.ctrl[1] = -0.8 * (t - 3.0)
        elif 4.0 <= t < 5.0:
            data.ctrl[1] = -0.8 * (5.0 - t)
        else:
            data.ctrl[1] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()

        # Logging
        time_log.append(t)
        joint_angle_log.append(data.qpos[joint_x_id])
        ball_rel_x_log.append(
            data.xpos[ball_id][0] - data.xpos[plate_id][0]
        )

        time.sleep(dt)

# ---------------- Plot 1: Arm joint response ----------------
plt.figure()
plt.plot(time_log, joint_angle_log)
plt.xlabel("Time [s]")
plt.ylabel("Joint X Angle [rad]")
plt.title("Arm Joint Angle – Bidirectional Motion")
plt.grid()
plt.show()

# ---------------- Plot 2: Ball response ----------------
plt.figure()
plt.plot(time_log, ball_rel_x_log)
plt.xlabel("Time [s]")
plt.ylabel("Ball X Position Relative to Plate [m]")
plt.title("Ball Response to Bidirectional Plate Tilt")
plt.grid()
plt.show()
