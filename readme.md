# DiffXPBD

This is a trajectory optimization project (still updating). Implementation is based on [Taichi](https://github.com/taichi-dev/taichi).

## Methods

* Objective: reach a given target position
* Input: control forces per frame per node
* Optimization method: gradient descent with line-search
* Forward simulation: XPBD
* Backward computation: Adjoint Method

## Usage

* `asset/input.json` sets initial and target position (.obj) and fixed points
* In `main.py`
  * Set `b_display` to `False` to start optimization and save results
  * Set `b_display` to `True` to display forward simulation

## Logs

#### Problems

* too large spring stiffness (1e10) --> $cond(A)\gg1$ --> CG doesn't converge
* large term 会崩
* short term
  * 仿真最后一帧和 target 差太多的时候，会先在第一帧来一个很大的力把布料拉很远，然后再慢慢回去
  * 中间不能保持一个 valid cloth 形态

#### Impl

* SNode 个数有上限，所以不能每帧开一个，要把所有帧的数据放到一个大 field 里
* safe normalization

## TODO

* control force visualization