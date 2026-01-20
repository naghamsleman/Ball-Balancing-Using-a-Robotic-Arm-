import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------- XML path ----------
XML_MODEL = r"""
<mujoco model="ball_balancing_arm">
    <compiler angle="degree" coordinate="local"/>
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <default>
        <joint damping="1"/>
        <geom friction="0.6 0.2 0.1" density="1000"/>
    </default>

    <worldbody>

        <!-- Base -->
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.1 0.1 0.05" rgba="0.3 0.3 0.3 1"/>

            <!-- Link 1 -->
            <body name="link1" pos="0 0 0.05">
                <joint name="joint_y"
                       type="hinge"
                       axis="0 1 0"
                       limited="true"
                       range="-25 25"/>

                <geom type="box"
                      size="0.02 0.02 0.15"
                      pos="0 0 0.15"
                      rgba="0.6 0.6 0.6 1"/>

                <!-- Link 2 -->
                <body name="link2" pos="0 0 0.3">
                    <joint name="joint_x"
                           type="hinge"
                           axis="1 0 0"
                           limited="true"
                           range="-45 45"/>

                    <geom type="box"
                          size="0.02 0.02 0.1"
                          pos="0 0 0.1"
                          rgba="0.7 0.7 0.7 1"/>

                    <!-- Plate with walls -->
                    <body name="plate" pos="0 0 0.2">
                        <!-- Plate surface -->
                        <geom type="box"
                              size="0.15 0.15 0.01"
                              rgba="0.2 0.6 0.8 1"/>

                        <!-- Walls -->
                        <geom type="box" size="0.15 0.005 0.03" pos="0 0.145 0.03"/>
                        <geom type="box" size="0.15 0.005 0.03" pos="0 -0.145 0.03"/>
                        <geom type="box" size="0.005 0.15 0.03" pos="0.145 0 0.03"/>
                        <geom type="box" size="0.005 0.15 0.03" pos="-0.145 0 0.03"/>
                    </body>

                </body>
            </body>
        </body>

        <!-- Ball -->
        <body name="ball" pos="0 0 0.7">
            <joint type="free"/>
            <geom type="sphere"
                  size="0.02"
                  rgba="0.9 0.1 0.1 1"/>
        </body>

    </worldbody>

    <!-- Actuators -->
    <actuator>
        <motor joint="joint_y" ctrlrange="-3 3" gear="150"/>
        <motor joint="joint_x" ctrlrange="-3 3" gear="150"/>
    </actuator>

</mujoco>
"""

# ============================================================
# 2) Build model + IDs (robust mapping to match XML)
# ============================================================
model = mujoco.MjModel.from_xml_string(XML_MODEL)
data = mujoco.MjData(model)

dt = model.opt.timestep
SIM_TIME = 10.0
steps = int(SIM_TIME / dt)

# ---------- IDs ----------
ball_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

joint_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
joint_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")

joint_x_adr = model.jnt_qposadr[joint_x_id]
joint_y_adr = model.jnt_qposadr[joint_y_id]

# ---------- Actuator indices (بدون افتراض ترتيب ctrl) ----------
act_x = None
act_y = None
for a in range(model.nu):
    if model.actuator_trnid[a, 0] == joint_x_id:
        act_x = a
    if model.actuator_trnid[a, 0] == joint_y_id:
        act_y = a

if act_x is None and act_y is None:
    raise RuntimeError("No actuators found for joint_x / joint_y. Check <actuator> in XML.")

# ---------- Helper: position of ball along PLATE X axis ----------
def plate_x_coord():
    # ball and plate positions in world
    p_ball  = data.xpos[ball_id].copy()
    p_plate = data.xpos[plate_id].copy()

    # plate rotation matrix in world: reshape 9 -> 3x3
    R = data.xmat[plate_id].reshape(3, 3)  # world_R_plate

    # plate X axis in world is first column of R
    x_axis_world = R[:, 0]

    # coordinate of relative vector projected on plate X axis
    rel = p_ball - p_plate
    x_on_plate = float(np.dot(rel, x_axis_world))
    return x_on_plate

# ---------- Simple metrics ----------
def compute_metrics(t, err, band=0.01):
    t = np.asarray(t, float)
    err = np.asarray(err, float)
    peak = float(np.max(np.abs(err)))
    rms  = float(np.sqrt(np.mean(err**2)))
    tail = max(1, int(0.10 * len(err)))
    ss   = float(np.mean(np.abs(err[-tail:])))
    # settling time: first time stays within band until end
    inside = np.abs(err) <= band
    st = None
    for k in range(len(err)):
        if inside[k] and np.all(inside[k:]):
            st = float(t[k])
            break
    return peak, rms, ss, st

# ---------- Reset + initial disturbance ----------
mujoco.mj_resetData(model, data)

# find ball free joint qpos address (7 values)
ball_free_adr = None
for j in range(model.njnt):
    if (model.jnt_bodyid[j] == ball_id) and (model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
        ball_free_adr = model.jnt_qposadr[j]
        break
if ball_free_adr is None:
    raise RuntimeError("Ball must have a free joint (<joint type='free'/>).")

# put ball slightly off center (so control has something to do)
data.qpos[ball_free_adr + 0] += 0.06  # 6 cm
mujoco.mj_forward(model, data)

# ---------- Tiny calibration: choose which actuator controls plate-X, and sign ----------
# Idea: apply a small positive command and see how plate-X coordinate changes tendency.
def test_actuator(act_idx, u_test=0.3, test_time=0.2):
    # save state
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    x0 = plate_x_coord()

    data.ctrl[:] = 0.0
    data.ctrl[act_idx] = u_test
    n = int(test_time / dt)
    for _ in range(n):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    x1 = plate_x_coord()
    dx = x1 - x0

    # restore state
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    return dx

candidates = []
if act_x is not None:
    candidates.append(act_x)
if act_y is not None:
    candidates.append(act_y)

# pick actuator with stronger effect on plate-X coordinate
effects = [(a, abs(test_actuator(a))) for a in candidates]
act_use = max(effects, key=lambda z: z[1])[0]

# sign: if +u makes x increase, then to reduce positive x we need negative u
dx_sign = test_actuator(act_use)
SIGN = -1.0 if dx_sign > 0 else 1.0

print(f"Using actuator index: {act_use}  |  SIGN={SIGN:+.1f}")
# ---------- PID gains (ابدأ بهالقيم) ----------
Kp = 1.0
Ki = 0.3
Kd = 2.0
I_LIM = 0.1

# ---------- PID states ----------
x_d = 0.0
e_prev = 0.0
I = 0.0

# ---------- Logs for plots ----------
t_log, x_log, jx_log, jy_log, u_log = [], [], [], [], []

# ---------- Run simulation ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(steps):
        t = i * dt

        # 1) measure ball coordinate along plate X axis
        x = plate_x_coord()

        # 2) error
        e = x_d - x

        # 3) PID
        I += e * dt
        I = float(np.clip(I, -I_LIM, I_LIM))
        D = (e - e_prev) / dt
        e_prev = e

        u = SIGN * (Kp * e + Ki * I + Kd * D)
        u = float(np.clip(u, -1.0, 1.0))

        # 4) apply ONLY one actuator (X-axis control), set others to zero
        data.ctrl[:] = 0.0
        data.ctrl[act_use] = u

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        # logs
        t_log.append(t)
        x_log.append(x)
        jx_log.append(float(data.qpos[joint_x_adr]))
        jy_log.append(float(data.qpos[joint_y_adr]))
        u_log.append(u)

print("Done.")

# ---------- Metrics + Plots ----------
err = np.array(x_log) - x_d
band = 0.025
peak, rms, ss, st = compute_metrics(t_log, err, band=band)

print("\nMetrics (X regulation):")
print(f"Peak deviation [m]     : {peak:.6f}")
print(f"RMS error [m]          : {rms:.6f}")
print(f"Steady-state error [m] : {ss:.6f}")
print(f"Settling time [s]      : {st if st is not None else 'Not settled'}")

plt.figure()
plt.plot(t_log, x_log, label="Ball coord along plate X [m]")
plt.axhline(+band, linestyle="--", label="band")
plt.axhline(-band, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("Ball Position (Plate X axis)")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(t_log, jy_log, label="joint_y [rad]")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.title("Joint Angle")
plt.grid(True)
plt.legend()

plt.show()
