# TrajOpt

This is a trajectory optimization project (still updating). Implementation is based on [Taichi](https://github.com/taichi-dev/taichi).

## Methods

* Objective: reach a given target position
  * [x] regularization
  * [ ] smoothness
* Control parameters: forces per frame per node
* Optimization method
  * [x] gradient descent with line-search
  * [x] Step and projection
  * [ ] L-BFGS
  * [ ] Gauss-Newton
* Forward simulation
  * [x] XPBD
  * [x] Newton's Method
* Backward computation: Adjoint Method

## Usage

* `asset/input.json` sets initial and target position (.obj) and fixed points
* In `main.py`
  * Set `b_display` to `False` to start optimization and save results
  * Set `b_display` to `True` to display forward simulation

## Before running

You must fix some bugs in [Tina](https://github.com/taichi-dev/taichi_three) manually .

* See this [pull request](https://github.com/taichi-dev/taichi_three/pull/41/commits/aebdda53d8b99e9ba8260fbc876ea9ad600222e9)

* As self-defined keyboard events will shelter Tina's GUI handler, I change Tina's camera control to purely mouse control. In `Tina/util/control.py`:

  ```python
  print('[Tina] Hint: LMB to orbit, MMB to pan, RMB to zoom')
  class Control:
      def on_lmb_drag(self, delta, origin):
          if self.gui.is_pressed(self.gui.CTRL):
              delta = delta * 0.2
          if self.gui.is_pressed(self.gui.SHIFT):
              delta = delta * 5
          self.on_orbit(delta, origin)
  
      def on_mmb_drag(self, delta, origin):
          if self.gui.is_pressed(self.gui.CTRL):
              delta = delta * 0.2
          if self.gui.is_pressed(self.gui.SHIFT):
              delta = delta * 5
          self.on_pan(delta, origin)
  
      def on_rmb_drag(self, delta, origin):
          if self.gui.is_pressed(self.gui.CTRL):
              delta = delta * 0.2
          if self.gui.is_pressed(self.gui.SHIFT):
              delta = delta * 5
          self.on_zoom(delta[1] * 10, origin)
  ```

  