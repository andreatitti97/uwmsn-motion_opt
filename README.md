# uwmsn-motion_opt

This package contains the motion-planning and optimization layer for the UWMSN project. It computes control policies for AUV teams operating under acoustic communication constraints and limited network connectivity.

## Purpose

`uwmsn-motion_opt` implements the planning logic for the underwater monitoring scenario. It consumes target estimates and neighboring state information, then computes feasible policies for the current AUV while accounting for communication-disrupted information exchange.

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

## Citation

If you use this package in research or teaching, please cite the project paper describing the motion optimization strategy under intermittent communication.

> Tiranti, A., et al. "Motion optimization strategy for passive acoustic monitoring with a team of AUVs considering intermittent communication." Please cite the published paper appropriately in any derived work.

A placeholder for the second paper can be added here once its final bibliographic information is ready.

## Notes

- The package targets ROS 1 and expects a catkin workspace.
- It is designed to be used together with `uwmsn-sim` and `uw-communication`.
- The code remains research-oriented and is intended to be cleaned further before the first public GitHub release.
