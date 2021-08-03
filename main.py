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
n_frame = 50
dt = 0.005
sim_med = "Newton"
iter = 500 if sim_med == "XPBD" else 5
err = 1e-6
use_attach = True
use_spring = True
use_stretch = False
use_bend = False
k_spring = 1e3
k_attach = 1e10

# optimization params
op_med = "gradient"
n_epoch = 10
desc_rate = 1.0  # start step size
regl_coef = 0.1

# create scene (allocate memory)
cloth_model = ClothModel(input_json)

cg_iter = int(cg_iter_ratio * 3 * cloth_model.n_vert)
linear_solver = LinearSolver(cloth_model.n_vert, cg_precond, cg_iter, cg_err)

cloth_sim = ClothSim(cloth_model, dt, sim_med, iter, err, linear_solver,
                     use_spring, use_stretch, use_bend, use_attach,
                     k_attach=k_attach, k_spring=k_spring)

opt = Optimizer(cloth_model, cloth_sim, linear_solver, op_med, n_frame,
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
    g_stop = False


def display():
    global g_stop
    frame_i = 0
    while gui.running:
        if gui.get_event():
            if gui.event.key == gui.ESCAPE:
                break
            elif gui.event.key == gui.SPACE and gui.event.type == gui.PRESS:
                g_stop = ~g_stop
        if not g_stop:
            frame_i += 1
            print(f"[frame {frame_i}]")
            opt.tmp_vec.fill(0.0)
            iter, err = cloth_sim.step(opt.tmp_vec, opt.tmp_vec)
            print("%d, %.1ef" % (iter, err))
            cloth_model.update_scene(opt.tmp_vec.to_numpy())
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()


def optimize():
    opt.forward()
    for i in range(n_epoch):
        print(f"====================== epoch {i} =====================")

        epoch_dir = os.path.join(output_path, "epoch_%i" % i)
        os.makedirs(epoch_dir)

        b_converge = opt.backward()
        print("[epoch %i] loss: %.1f (%.1f / %.1f)" % (i, opt.loss, opt.x_loss, opt.f_loss))

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
