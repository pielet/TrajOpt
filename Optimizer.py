import os
import numpy as np
import taichi as ti
from FieldUtil import *


@ti.data_oriented
class Optimizer:
    OptimizeMethod = {
        "gradient": 0,
        "SAP": 1,
        "Quasi-Newton": 2
    }

    def __init__(self,
                 cloth_model, cloth_sim, linear_solver, method, n_frame,
                 desc_rate, smoothness,
                 ls_alpha=0.5, ls_gamma=0.03,
                 b_verbose=False):

        assert n_frame >= 3

        self.b_verbose = b_verbose

        self.cloth_model = cloth_model
        self.cloth_sim = cloth_sim

        self.method = self.OptimizeMethod[method]

        self.desc_rate = desc_rate
        self.epsilon = smoothness
        self.ls_alpha = ls_alpha
        self.ls_gamma = ls_gamma

        self.n_frame = n_frame
        self.dt = cloth_sim.dt
        self.n_vert = cloth_sim.n_vert

        self.trajectory = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        self.control_force = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        # 3 lambda are enough for computing
        # but if we want to do line-search and have better initial guess for CG, we'd better remember them all
        self.adjoint_vec = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))

        # line-search
        self.loss = 0.0
        self.x_loss = 0.0
        self.f_loss = 0.0
        self.step_size = 1.0
        self.gradient = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        self.tentetive_ctrl_f = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))

        self.target = ti.Vector.field(3, ti.f32, self.n_vert)
        self.tmp_vec = ti.Vector.field(3, ti.f32, self.n_vert)  # to access one frame data
        self.sum = ti.field(ti.f32, ())

        self.linear_solver = linear_solver
        self.b = linear_solver.b

    def initialize(self):
        self.mass = self.cloth_sim.mass
        self.trajectory.fill(0.0)
        self.control_force.fill(0.0)
        self.adjoint_vec.fill(0.0)
        self.gradient.fill(0.0)
        self.tentetive_ctrl_f.fill(0.0)
        self.target.from_numpy(self.cloth_model.target_verts)

    @ti.kernel
    def get_frame(self, vec: ti.template(), fi: ti.i32):
        for i in self.tmp_vec:
            self.tmp_vec[i] = vec[fi, i]

    @ti.kernel
    def set_frame(self, vec: ti.template(), fi: ti.i32):
        for i in self.tmp_vec:
            vec[fi, i] = self.tmp_vec[i]

    def save_trajectory(self, output_dir):
        """
        Save a whole trajectory under checkpoint/timestamp/epoch_i/trajectory.ply
        """
        writer = ti.PLYWriter((self.n_frame + 1) * self.n_vert, (self.n_frame + 1) * self.cloth_model.n_face, "tri")

        np_traj = np.vstack([self.cloth_model.verts, self.trajectory.to_numpy().reshape(-1, 3)])
        faces = np.tile(self.cloth_model.faces.flatten(), self.n_frame + 1) + \
                np.repeat(self.n_vert * np.arange(self.n_frame + 1), 3 * self.cloth_model.n_face)
        np_force = np.vstack([self.control_force.to_numpy().reshape(-1, 3), np.zeros((self.n_vert, 3))])

        writer.add_vertex_pos(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2])
        writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
        writer.add_faces(faces)
        writer.export(os.path.join(output_dir, "trajectory.ply"))

    def save_frame(self, output_dir):
        """
        Save the model per frame under checkpoint/timestamp/epoch_i/frame_i.ply
        """
        for i in range(self.n_frame + 1):
            writer = ti.PLYWriter(self.n_vert, self.cloth_model.n_face, "tri")

            if i == 0:
                np_verts = self.cloth_model.verts
            else:
                self.get_frame(self.trajectory, i - 1)
                np_verts = self.tmp_vec.to_numpy()

            if i == self.n_frame:
                np_force = np.zeros((self.n_vert, 3))
            else:
                self.get_frame(self.control_force, i)
                np_force = self.tmp_vec.to_numpy()

            writer.add_vertex_pos(np_verts[:, 0], np_verts[:, 1], np_verts[:, 2])
            writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
            writer.add_faces(self.cloth_model.faces)
            writer.export_frame(i, os.path.join(output_dir, "frame"))

    def __forward(self, control_force):
        print("[start forward]", end=" ")

        # reset initial states
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)

        frame, err = 0, 0
        for i in range(self.n_frame):
            print(f"frame {i}")
            self.get_frame(control_force, i)
            fi, ei = self.cloth_sim.step(self.tmp_vec, self.tmp_vec)  # FIXME: assume ext_f and x_next can be the same vector
            frame += fi
            err += ei
            self.set_frame(self.trajectory, i)

        print("avg.iter: %i, avg.err: %.1e" % (frame / self.n_frame, err / self.n_frame))

    def forward(self):
        """ Run frames of simulation """
        self.__forward(self.control_force)

    def __compute_loss(self, control_force):
        """
        Compute loss (forward must be called before)
        L = 1/2 * (M / n_f / h^2)^2 * ||q_{t+1} - q*||^2 + 1/2 * \epsilon * ||p||^2
        """
        # regularization term
        reduce(self.sum, control_force, control_force)
        f_loss = 0.5 * self.epsilon * self.sum[None]

        # target position term
        self.get_frame(self.trajectory, self.n_frame - 1)
        axpy(-1.0, self.target, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        x_loss = 0.5 * (self.mass / self.n_frame / self.dt ** 2) ** 2 * self.sum[None]

        return x_loss + f_loss, x_loss, f_loss

    def compute_loss(self):
        """ Compute init loss """
        self.loss, self.x_loss, self.f_loss = self.__compute_loss(self.control_force)

    @ti.kernel
    def __compute_gradient(self):
        # dLdp = \epsilon * p + M / (n_f^2 * h^2) * \lambda
        for I in ti.grouped(self.gradient):
            self.gradient[I] = self.epsilon * self.control_force[I] + \
                               self.mass / (self.n_frame * self.dt) ** 2 * self.adjoint_vec[I]

    def line_search(self):
        """
        Find a proper step size
        This method will invoke forward simulation several times, so don't need to call forward() anymore
        """
        # compute line search threshold
        self.__compute_gradient()
        reduce(self.sum, self.gradient, self.gradient)
        threshold = self.ls_gamma * self.desc_rate * self.sum[None]

        # line-search
        step_size = min(1.0, self.step_size / self.ls_alpha)  # use step size from last epoch as initial guess
        while True:
            self.tentetive_ctrl_f.copy_from(self.control_force)
            axpy(-step_size * self.desc_rate, self.gradient, self.tentetive_ctrl_f)

            # update trajectory
            self.__forward(self.tentetive_ctrl_f)
            cur_loss, cur_x_loss, cur_f_loss = self.__compute_loss(self.tentetive_ctrl_f)

            print("step size: %f  loss: %.1f (%.1f / %.1f)  threshold: %.1f" % (step_size, cur_loss, cur_x_loss, cur_f_loss, self.loss + step_size * threshold))

            if cur_loss < self.loss + step_size * threshold or step_size < 1e-5:
                break
            step_size *= self.ls_alpha

        # commit control force
        self.step_size = step_size
        self.loss, self.x_loss, self.f_loss = cur_loss, cur_x_loss, cur_f_loss
        self.control_force.copy_from(self.tentetive_ctrl_f)

        return self.x_loss < 1e-6

    def __compute_constrain(self):
        """ Compute constrain value: C = 1/2 ||q_{t+1} - q*||_M^2 """
        self.get_frame(self.trajectory, self.n_frame - 1)
        axpy(-1.0, self.target, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        return 0.5 * self.mass * self.sum[None]

    def SAP(self):
        """
        Perform one step SAP: p^{k+1} = p^{k} - c(p) / ||\grad C(p)||^2 * \grad c(p)
        \grad c(p) = -h^2 * \lambda
        """
        # update control forces
        cons = self.__compute_constrain()
        reduce(self.sum, self.adjoint_vec, self.adjoint_vec)
        dL = cons / (self.dt ** 2 * self.sum[None])
        axpy(-dL, self.adjoint_vec, self.control_force)

        # compute new loss
        self.__forward(self.control_force)
        self.loss, self.x_loss, self.f_loss = self.__compute_loss(self.control_force)

        return self.x_loss < 1e-6

    def backward(self):
        """ Update control forces """

        # compute lambda
        print("[compute lambda]", end=" ")

        iter, err = 0, 0
        for i in range(self.n_frame - 1, -1, -1):
            # prepare A
            self.get_frame(self.trajectory, i)
            self.cloth_sim.compute_hessian(self.tmp_vec)

            # prepare b
            if i == self.n_frame - 1:  # b_t = M * (q_t - q*)
                self.b.copy_from(self.tmp_vec)
                axpy(-1.0, self.target, self.b)
                scale(self.b, self.mass)
            elif i == self.n_frame - 2:  # b_{t-1} = 2 * M * lambda_t
                self.b.fill(0.0)
                self.get_frame(self.adjoint_vec, i + 1)
                axpy(2 * self.mass, self.tmp_vec, self.b)
            else:  # b_i = 2 * M * lambda_{i+1} - M * lambda_{i+2}
                self.b.fill(0.0)
                self.get_frame(self.adjoint_vec, i + 1)
                axpy(2 * self.mass, self.tmp_vec, self.b)
                self.get_frame(self.adjoint_vec, i + 2)
                axpy(-self.mass, self.tmp_vec, self.b)

            # linear solve
            self.get_frame(self.adjoint_vec, i)
            iter_i, err_i = self.linear_solver.conjugate_gradient(self.tmp_vec)
            self.set_frame(self.adjoint_vec, i)
            iter += iter_i
            err += err_i

        print("CG avg.iter: %d, avg.err: %.1e" % (iter / self.n_frame, err / self.n_frame))

        # update control forces
        if self.method == self.OptimizeMethod["gradient"]:
            print("[start line-search]")
            b_converge = self.line_search()
        elif self.method == self.OptimizeMethod["SAP"]:
            print("[start SAP]")
            b_converge = self.SAP()

        return b_converge
