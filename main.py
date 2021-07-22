import os
import taichi as ti
import tina
from threading import Thread
from time import gmtime, strftime

from ClothModel import ClothModel
from ClothSim import ClothSim
from Optimizer import Optimizer

ti.init(arch=ti.cpu, debug=True, excepthook=True)

# input/output path
input_model = "assets/init.obj"
target_model = "assets/target.obj"

output_path = os.path.join("checkpoints", strftime("%Y-%m-%d %H-%M-%S", gmtime()))
os.makedirs(output_path)

# display/output params
b_display = False
b_save_frame = True
b_save_trajectory = True

# simulation params
n_frame = 3
dt = 0.01
XPBD_iter = 10
use_spring = True
use_stretch = False
use_bend = False
k_spring = 100

# optimization params
n_epoch = 20
desc_rate = 0.5
regl_coef = 0.1
cg_precond = "None"

# create scene (allocate memory)
scene = tina.Scene()
cloth_model = ClothModel(input_model, target_model, output_path)
cloth_sim = ClothSim(cloth_model, dt, XPBD_iter, use_spring, use_stretch, use_bend, k_spring=k_spring)
opt = Optimizer(cloth_model, cloth_sim, desc_rate, regl_coef, n_frame, cg_precond=cg_precond)

# init data
cloth_model.bind_scene(scene)
cloth_sim.initialize()
opt.initialize()

gui = ti.GUI()


def run_epoch():
    opt.forward()
    for i in range(n_epoch):
        print("=================== epoch %i ==================" % i)

        epoch_dir = os.path.join(output_path, "epoch_%i" % i)
        os.makedirs(epoch_dir)

        b_converge = opt.backward()
        print("[epoch %i] loss: %.2ef" % (i, opt.loss))

        if b_save_trajectory:
            opt.save_trajectory(epoch_dir)
        if b_save_frame:
            opt.save_frame(epoch_dir)

        if b_converge:
            break


def main():
    sim_p = Thread(target=run_epoch)
    sim_p.start()
    # while gui.running and not gui.get_event(gui.ESCAPE):
    #     scene.input(gui)
    #     scene.render()
    #     gui.set_image(scene.img)
    #     gui.show()
    sim_p.join()


if __name__ == "__main__":
    main()
