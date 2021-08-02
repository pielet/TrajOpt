# TrajOpt

This is a trajectory optimization project (still updating). Implementation is based on [Taichi](https://github.com/taichi-dev/taichi).

## Methods

* Objective: reach a given target position
  * [x] regularization
  * [ ] smoothness
* Control parameters: forces per frame per node
* Optimization method
  * [x] gradient descent with line-search
  * [ ] L-BFGS
  * [ ] Gauss-Newton
* Forward simulation
  * [x] XPBD
  * [ ] Newton's Method
* Backward computation: Adjoint Method

## Usage

* `asset/input.json` sets initial and target position (.obj) and fixed points
* In `main.py`
  * Set `b_display` to `False` to start optimization and save results
  * Set `b_display` to `True` to display forward simulation
