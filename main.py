import os
import taichi as ti
import tina
from time import gmtime, strftime

from ClothModel import ClothModel
from LinearSolver import LinearSolver
from ClothSim import ClothSim
from Optimizer import Optimizer

ti.init(arch=ti.cpu, debug=False, excepthook=True)

# display/output params
b_display = False
b_save_frame = True
b_save_trajectory = True
b_verbose = False

# input/output path
input_json = "assets/input.json"
if not b_display:
    output_path = os.path.join("checkpoints", strftime("%Y-%m-%d %H-%M-%S", gmtime()))
    os.makedirs(output_path)

# linear solver params
cg_precond = "Jacobi"
cg_iter_ratio = 1.0
cg_err = 1e-12

# simulation params
n_frame = 100
dt = 0.005
sim_med = "Newton"
iter = 500 if sim_med == "XPBD" else 5
sim_err = 1e-6
use_attach = True
use_spring = True
use_stretch = False
use_bend = False
k_spring = 1e3
k_attach = 1e10

# optimization params
opt_med = "gradient"
n_epoch = 50
desc_rate = 0.01  # start step size
regl_coef = 10.0

# create scene (allocate memory)
cloth_model = ClothModel(input_json)

cg_iter = int(cg_iter_ratio * 3 * cloth_model.n_vert)
linear_solver = LinearSolver(cloth_model.n_vert, cg_precond, cg_iter, cg_err)

cloth_sim = ClothSim(cloth_model, dt, sim_med, iter, sim_err, linear_solver,
                     use_spring, use_stretch, use_bend, use_attach,
                     k_attach=k_attach, k_spring=k_spring)

opt = Optimizer(cloth_model, cloth_sim, linear_solver, opt_med, n_frame,
                desc_rate, regl_coef,
                b_verbose=b_verbose)

if b_display:
    scene = tina.Scene(res=960)  # it allocates memory
    cloth_model.bind_scene(scene)  # it calls kernel function

# init data
cloth_sim.initialize()
opt.initialize()

if b_display:
    gui = ti.GUI("DiffXPBD", scene.res)


def display():
    b_stop = False
    frame_i = 0
    while gui.running:
        if gui.get_event():
            if gui.event.key == gui.ESCAPE:
                break
            elif gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
                b_stop = ~b_stop
        if not b_stop:
            frame_i += 1
            print(f"[frame {frame_i}]")
            opt.tmp_vec.fill(0.0)
            iter, err = cloth_sim.step(opt.tmp_vec, opt.tmp_vec)
            print("%d, %.1e" % (iter, err))
            cloth_model.update_scene(opt.tmp_vec.to_numpy())
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()


def optimize():
    b_converge = False
    for i in range(n_epoch + 1):
        print(f"====================== epoch {i} =====================")

        epoch_dir = os.path.join(output_path, f"epoch_{i}")
        os.makedirs(epoch_dir)

        if i == 0:
            opt.forward()
            opt.compute_loss()
        else:
            b_converge = opt.backward()

        print(f"[epoch {i}] loss: {opt.loss} ({opt.x_loss} / {opt.f_loss})")

        if b_save_trajectory:
            opt.save_trajectory(epoch_dir)
        if b_save_frame:
            opt.save_frame(epoch_dir)

        if b_converge:
            break


if __name__ == "__main__":
    if b_display:
        display()
    else:
        optimize()
