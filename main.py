import os
import time

import numpy as np
import taichi as ti
from time import localtime, strftime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ClothModel import ClothModel
from LinearSolver import LinearSolver
from ClothSim import ClothSim
from Optimizer import Optimizer

ti.init(arch=ti.cpu, debug=False)

# display/output params
b_display = False
b_visualize = False
vis_path = "checkpoints\\2021-12-27 13-57-30\\epoch_100"
b_save_every_epoch = True
b_save_npy = True
b_verbose = False

# input/output path
input_json = "assets/input.json"
if not b_display:
    output_path = os.path.join("checkpoints", strftime("%Y-%m-%d %H-%M-%S", localtime()))
    os.makedirs(output_path)

# implicit Euler linear solver params
ie_solver = "direct"  # direct, cg
ie_cg_precond = "Jacobi"
ie_cg_iter_ratio = 1.0
ie_cg_err = 1e-12

# simulation params
n_frame = 80
dt = 0.01
sim_med = "implicit"  # implicit, symplectic, XPBD
iter = 500 if sim_med == "XPBD" else 3  # only for implicit Euler / XPBD
sim_err = 1e-12  # only for implicit Euler / XPBD
use_attach = True
use_spring = True
use_stretch = False
use_bend = False
k_spring = 500
k_attach = 1e5

# optimization params
opt_med = "Gauss-Newton"  # gradient, projected Newton, SAP, Gauss-Newton, L-BFGS
n_epoch = 50
## force based opt params
opt_sim_med = "implicit"  # implicit, symplectic
desc_rate = 0.01  # start step size
regl_coef = 1e-4
## position based opt params
b_soft_con = False if opt_med == "SAP" or opt_med == "projected Newton" else False
init_med = "load"  # static, solve (0 physical loss), fix (0 constrain loss), load (mixed loss)
### Gauss-Newton linear solver params
b_matrix_free = False
gn_integration = "implicit"  # implicit, symplectic
gn_solver = "direct"  # direct, L-BFGS
gn_cg_precond = "None"
gn_cg_iter_ratio = 1.0
gn_cg_err = 1e-12

# create scene (allocate memory)
cloth_model = ClothModel(input_json)

ie_cg_iter = int(ie_cg_iter_ratio * 3 * cloth_model.n_vert)
ie_linear_solver = LinearSolver(cloth_model.n_vert, ie_cg_precond, ie_cg_iter, ie_cg_err)

cloth_sim = ClothSim(cloth_model, dt, sim_med, iter, sim_err, ie_linear_solver,
                     use_spring, use_stretch, use_bend, use_attach,
                     k_attach=k_attach, k_spring=k_spring, b_verbose=b_verbose)

opt = Optimizer(cloth_model, cloth_sim, ie_linear_solver, opt_med, n_frame,
                opt_sim_med, desc_rate, regl_coef,
                init_med, b_soft_con, gn_integration, gn_solver, b_matrix_free, gn_cg_precond, gn_cg_iter_ratio, gn_cg_err,
                b_verbose=b_verbose)

# init data
cloth_model.initialize()
cloth_sim.initialize()
opt.initialize()

if b_display:
    window = ti.ui.Window("Loopy cloth", (640, 640))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()

    camera = ti.ui.make_camera()
    camera.position(-1.8, 0.5, 0.4)
    camera.lookat(0.0, 0.0, 0.0)

    scene.mesh(cloth_sim.x, cloth_model.face_field, two_sided=True)
    scene.set_camera(camera)
    scene.point_light((0, 0.5, 0), (1, 1, 1))
    canvas.scene(scene)
    window.show()

def display():
    b_stop = False
    frame_i = 0
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break
            elif window.event.key == ti.ui.SPACE:
                b_stop = ~b_stop
        if not b_stop:
            frame_i += 1
            print(f"[frame {frame_i}]")
            opt.tmp_vec.fill(0.0)
            iter, err = cloth_sim.step(opt.tmp_vec, opt.tmp_vec)
            print("%d, %.1e" % (iter, err))

        scene.mesh(opt.tmp_vec, cloth_model.face_field, two_sided=True)
        scene.set_camera(camera)
        scene.point_light((0, 0.5, 0), (1, 1, 1))
        canvas.scene(scene)
        window.show()

        time.sleep(dt * 2)


def optimize():
    for i in range(n_epoch + 1):
        print(f"====================== epoch {i} =====================")

        c = (cloth_sim.mass / n_frame / dt ** 2) ** 2 / regl_coef

        if i == 0:
            opt.compute_init_loss()
            if opt_med == "Gauss-Newton":
                if not b_soft_con:
                    print(f"[init] loss: {opt.loss}")
                else:
                    print(f"[init] loss: {opt.loss} (loopy: {opt.loopy_loss} / constrain: {opt.constrain_loss / c} / scaled_constrain: {opt.constrain_loss})")
            else:
                print(f"[init] loss: {opt.loss} (force: {opt.force_loss} / constrain: {opt.constrain_loss})")
        else:
            b_converge = opt.one_iter()
            if opt_med == "Gauss-Newton":
                if not b_soft_con:
                    print(f"[epoch {i}] loss: {opt.loss}")
                else:
                    print(f"[init] loss: {opt.loss} (loopy: {opt.loopy_loss} / constrain: {opt.constrain_loss / c} / scaled_constrain: {opt.constrain_loss})")
            elif opt_med == "gradient":
                print(f"[epoch {i}] loss: {opt.loss} (force: {opt.force_loss / regl_coef} / constrain: {opt.constrain_loss / (cloth_sim.mass / n_frame / dt ** 2) ** 2})")
            else:
                print(f"[epoch {i}] loss: {opt.loss} (force: {opt.force_loss} / constrain: {opt.constrain_loss} / lambda: {opt.L} / loopy_loss: {opt.loopy_loss})")

        if b_save_every_epoch or i == 0 or i == n_epoch:
            epoch_dir = os.path.join(output_path, f"epoch_{i}")
            os.makedirs(epoch_dir)

            if opt_med == "Gauss-Newton":
                opt.compute_virtual_force()

            opt.save_trajectory(epoch_dir, b_save_npy)
            opt.save_frame(epoch_dir)

    visualize(np.array(opt.loss_list), np.array(opt.loss_per_frame))

def visualize(loss, loss_per_frame):
    # loss_per_frame = np.roll(loss_per_frame, 2, axis=1)
    loss_per_frame = np.roll(loss_per_frame, 1, axis=1)[:, :-2]

    fig, ax1 = plt.subplots()

    if opt_med == "Gauss-Newton" and not b_soft_con:
        color = 'tab:blue'
        ax1.set_xlabel('iter')
        ax1.set_ylabel('loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        # ax1.plot(loss[:, 0])
        ax1.plot(loss)
    else:
        color = 'tab:blue'
        ax1.set_xlabel('iter')
        ax1.set_ylabel('loss', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.plot(loss[:, 1], color=color)

        ax2 = ax1.twinx()

        color = 'tab:orange'
        ax2.set_ylabel('constrain', color=color)  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(loss[:, 2], color=color)

        fig.tight_layout()

    plt.savefig(os.path.join(output_path, "loss.png"))
    plt.close()

    fig, axis = plt.subplots(1, 1)
    axis.set_xlim(0, n_frame - 2)
    axis.set_ylim(0, loss_per_frame.max())
    data = axis.plot(np.arange(n_frame - 2))[0]

    def update(i):
        data.set_ydata(loss_per_frame[i])
        axis.set_title(f"epoch_{i}")
        axis.set_ylim(loss_per_frame[i].min(), loss_per_frame[i].max())

    total_time = 5
    loss_ani = FuncAnimation(fig, update, frames=(n_epoch+1), interval=(total_time * 1000 / n_epoch))
    loss_ani.save(os.path.join(output_path, "loss_per_frame.gif"))


if __name__ == "__main__":
    if b_display:
        display()
    elif b_visualize:
        loss = np.load(os.path.join(vis_path, "loss.npy"))
        loss_per_frame = np.load(os.path.join(vis_path, "loss_per_frame.npy"))
        visualize(loss, loss_per_frame)
    else:
        optimize()
