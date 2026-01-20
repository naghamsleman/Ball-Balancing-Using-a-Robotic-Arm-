# Ball-Balancing-Using-a-Robotic-Arm-
Ball Balancing Using a Robotic Arm (MuJoCo Simulation)

Project Overview
This project presents a simulation-based implementation of a ball balancing system using a robotic arm.
A flat plate is attached to the end-effector of a two-degree-of-freedom robotic arm, and a ball is placed on top of the plate.
By tilting the plate through controlled joint motions, the ball moves due to gravity and contact forces.

The main objective of this project is to build a realistic simulation model, analyze the physical behavior of the system,
and demonstrate the challenges of controlling an inherently unstable system.
Both open-loop experiments and closed-loop PID control are used to study the system response.

The simulation is implemented using the MuJoCo physics engine and Python.


System Description
- Robotic Arm: A serial robotic arm with two rotational joints.
- Plate: Rigidly attached to the second link of the arm.
- Ball: Modeled as a free rigid body.
- Simulation Engine: MuJoCo.

The plate orientation is controlled directly through joint commands.
Inverse kinematics is not required.


Project Structure

Ball Balancing Using a Robotic Arm in Simulation/
- src/
  - simulation.py
  - control.py
- models/
  - ball.xml
  - ball_with_pid.xml
- plots/
  - results.png
- README.txt
- requirements.txt


Requirements
- Python 3.8 or higher
- MuJoCo
- NumPy
- Matplotlib


How to Run the Project

1. Install the required libraries:
pip install -r requirements.txt

2. Visualize the simulation model:
python src/simulation.py

3. Run experimental simulations:
python src/control.py


Experiments and Results

Open-loop experiments apply smooth joint motions to tilt the plate.
The ball moves due to gravity and drifts away from the center, showing the unstable nature of the system.

Closed-loop PID control improves stability and keeps the ball closer to the desired position.


Report
A detailed explanation is provided in the Robot Programming project report.


Authors
Nagham Sleman – Simulation modeling, XML implementation, visualization
Hazem Afif – Control design, PID implementation, performance evaluation


Notes
This project is simulation-based and intended for educational purposes.
