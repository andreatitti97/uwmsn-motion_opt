# uwmsn-motion_opt

This package contains the motion-planning and optimization layer for the UWMSN project. It computes control policies for AUV teams operating under acoustic communication constraints and limited network connectivity.

## Purpose

`uwmsn-motion_opt` implements the planning logic for the underwater monitoring scenario. It consumes target estimates and neighboring state information, then computes feasible control policies for the current AUV while accounting for communication-disrupted information exchange.

## Included algorithms and entry points

The project includes several optimization and control variants, including:

- `src/distr_opt.py`: distributed optimization entry point.
- `src/decentralized_opt.py`: decentralized optimization node.
- `src/distr_opt_MTT.py`: multi-target distributed optimization.
- `src/distr_opt_PSO.py`: PSO-based optimization variant.
- `src/dec_MPC.py`: decentralized MPC variant.
- `src/bnb.py`: branch-and-bound optimization implementation.
- `src/bnb_MTT.py`: multi-target BnB variant.
- `src/PSO_planner.py`: particle-swarm planner used by the optimization workflow.
- `include/uwmsn-motion_opt/*.py`: helper modules and planning support code.

## Role in the project workflow

This package receives state estimates and policy updates, solves the current optimization problem, and publishes the resulting control policy back into the ROS network for execution and simulation.

## Requirements

- ROS 1 catkin environment
- `rospy`
- `std_msgs`
- `numpy`
- `scipy` (when plotting or auxiliary numerical routines are used)
- `matplotlib` for visualization utilities
- `uwmsn_msgs` or the equivalent message layer used in the project

## Installation

Place this package inside the `src` folder of your catkin workspace and rebuild:

```bash
cd ~/ros1_ws
catkin_make
source devel/setup.bash
```

## Usage

Start the optimization node for a specific AUV:

```bash
roslaunch uwmsn-motion_opt distr_opt.launch auvID:=1 auvNum:=4
```

If you are using the multi-target configuration, prefer the multi-target launch files and align the AUV count and parameters with the simulator setup.

## Reproducibility

To reproduce a result, keep the following fixed for a given run:

- AUV count
- target count
- communication topology
- packet-loss parameters
- optimization algorithm variant
- launch arguments

This package is meant to be evaluated together with the simulator and communication layer as one research pipeline.

## Citation

If you use this package in research or teaching, please cite the project paper describing the motion optimization strategy under intermittent communication.

> A. Tiranti, P. Di Lillo, F. Wanderlingh, E. Simetti, M. Baglietto, and G. Antonelli, "Motion Optimization Strategy for Passive Acoustic Monitoring With a Team of AUVs Considering Intermittent Communication," IEEE Journal of Oceanic Engineering, vol. 50, no. 4, 2025. doi: 10.1109/JOE.2025.3586242.

## Notes

- The package targets ROS 1 and expects a catkin workspace.
- It is designed to be used together with `uwmsn-sim` and `uw-communication`.
- The code is research-oriented and can be used directly for controlled academic experimentation.
