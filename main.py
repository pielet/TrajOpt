import os
import taichi as ti
import tina
from threading import Thread
from time import gmtime, strftime

from ClothModel import ClothModel
from ClothSim import ClothSim
from Optimizer import Optimizer

ti.init(arch=ti.cpu, debug=True, excepthook=True)

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

# simulation params
n_frame = 3
dt = 0.005
XPBD_iter = 100
use_attach = True
use_spring = True
use_stretch = False
use_bend = False
k_spring = 1000
k_attach = 1e5

# optimization params
n_epoch = 50
desc_rate = 0.5
smoothness = 0.1
cg_precond = "None"

# create scene (allocate memory)
cloth_model = ClothModel(input_json)
cloth_sim = ClothSim(cloth_model, dt, XPBD_iter,
                     use_spring, use_stretch, use_bend, use_attach,
                     k_attach=k_attach, k_spring=k_spring)
opt = Optimizer(cloth_model, cloth_sim, desc_rate, smoothness, n_frame, cg_precond=cg_precond, b_verbose=b_verbose)

if b_display:
    scene = tina.Scene()  # it allocates memory
    cloth_model.bind_scene(scene)  # it calls kernel function

# init data
cloth_sim.initialize()
opt.initialize()

if b_display:
    gui = ti.GUI("DiffXPBD")
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
            print("[frame %i]" % frame_i)
            opt.tmp_vec.fill(0.0)
            cloth_sim.XPBD_step(opt.tmp_vec, opt.tmp_vec)
            cloth_model.update_scene(opt.tmp_vec.to_numpy())
        scene.input(gui)
        scene.render()
        gui.set_image(scene.img)
        gui.show()


def optimize():
    opt.forward()
    for i in range(n_epoch):
        print("=================== epoch %i ==================" % i)

        epoch_dir = os.path.join(output_path, "epoch_%i" % i)
        os.makedirs(epoch_dir)

        b_converge = opt.backward()
        print("[epoch %i] loss: %.2ef (%.1ef / %.1ef)" % (i, opt.loss, opt.x_loss, opt.f_loss))

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
