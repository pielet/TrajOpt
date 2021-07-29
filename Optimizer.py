import os
import numpy as np
import taichi as ti
from FieldUtil import *


@ti.data_oriented
class Optimizer:
    CG_PRECOND_METHED = {
        "None": 0,
        "Jacobi": 1
    }

    def __init__(self,
                 cloth_model, cloth_sim,
                 desc_rate, smoothness, n_frame,
                 ls_alpha=0.5, ls_gamma=0.03,
                 cg_precond="Jacobi",
                 cg_iter=100,
                 cg_err=1e-6,
                 b_verbose=False):

        assert n_frame >= 3

        self.b_verbose = b_verbose

        self.cloth_model = cloth_model
        self.cloth_sim = cloth_sim

        self.desc_rate = desc_rate
        self.epsilon = smoothness
        self.ls_alpha = ls_alpha
        self.ls_gamma = ls_gamma

        self.n_frame = n_frame
        self.dt = cloth_sim.dt
        self.n_vert = cloth_sim.n_vert

        self.trajectory = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        self.control_force = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        # In face 3 lambda are enough for computing
        # but if we want to do line-search and have better initial guess for CG, we'd better remember them all
        self.adjoint_vec = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))

        # line-search
        self.loss = None
        self.x_loss = 0.0
        self.f_loss = 0.0
        self.step_size = 1.0
        self.gradient = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))
        self.tentetive_ctrl_f = ti.Vector.field(3, ti.f32, (n_frame, self.n_vert))  # FIXME: can reduce to one Vector.field (w/ more complicate line-search)

        self.tmp_vec = ti.Vector.field(3, ti.f32, self.n_vert)

        # linear solver variables
        self.cg_precond = self.CG_PRECOND_METHED[cg_precond]
        self.cg_iter = cg_iter
        self.cg_err = cg_err
        self.alpha = ti.field(ti.f32, shape=())
        self.beta = ti.field(ti.f32, shape=())
        self.rTz = ti.field(ti.f32, shape=())
        self.sum = ti.field(ti.f32, shape=())
        self.res = ti.field(ti.f32, shape=())

        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.inv_A_diag = ti.Vector.field(3, ti.f32, self.n_vert)
        self.b = ti.Vector.field(3, ti.f32, self.n_vert)
        self.r = ti.Vector.field(3, ti.f32, self.n_vert)
        self.z = ti.Vector.field(3, ti.f32, self.n_vert)
        self.p = ti.Vector.field(3, ti.f32, self.n_vert)
        self.Ap = ti.Vector.field(3, ti.f32, self.n_vert)

    def initialize(self):
        self.mass = self.cloth_sim.mass
        self.trajectory.fill(0.0)
        self.control_force.fill(0.0)
        self.adjoint_vec.fill(0.0)
        self.gradient.fill(0.0)
        self.tentetive_ctrl_f.fill(0.0)

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
        print("[start forward]")

        # reset initial states
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)

        for i in range(self.n_frame):
            self.get_frame(control_force, i)
            self.cloth_sim.XPBD_step(self.tmp_vec, self.tmp_vec)  # FIXME: assume ext_f and x_next can be the same vector
            self.set_frame(self.trajectory, i)
            if self.b_verbose: print("frame %i" % i)

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
        self.b.from_numpy(self.cloth_model.target_verts)
        self.get_frame(self.trajectory, self.n_frame - 1)
        axpy(-1.0, self.tmp_vec, self.b)
        reduce(self.sum, self.b, self.b)
        x_loss = 0.5 * (self.mass / self.n_frame / self.dt ** 2) ** 2 * self.sum[None]

        return x_loss + f_loss, x_loss, f_loss

    @ti.kernel
    def __compute_gradient(self):
        # dLdp = \epsilon * p + h^2 * \lambda
        for I in ti.grouped(self.gradient):
            self.gradient[I] = self.epsilon * self.control_force[I] + self.dt ** 2 * self.adjoint_vec[I]

    def line_search(self):
        """
        Find a proper step size.
        This method will invoke forward simulation several times, so don't need to call forward() explicitly anymore
        """
        # compute line search threshold
        if not self.loss:  # first epoch
            self.loss, self.x_loss, self.f_loss = self.__compute_loss(self.control_force)
            print("[init loss] %f (%f / %f)" % (self.loss, self.x_loss, self.f_loss))
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

            print("step size: %f  loss: %.1f (%.1f / %.1f)" % (step_size, cur_loss, cur_x_loss, cur_f_loss))

            if cur_loss < self.loss + step_size * threshold:  # right step size
                break
            step_size *= self.ls_alpha
            if step_size < 1e-5:
                break

        # commit control force
        self.step_size = step_size
        self.loss, self.x_loss, self.f_loss = cur_loss, cur_x_loss, cur_f_loss
        self.control_force.copy_from(self.tentetive_ctrl_f)

        return self.x_loss < 1e-2

    def backward(self):
        """ Update control forces """
        # compute lambda
        print("[start backward]")
        for i in range(self.n_frame - 1, -1, -1):
            # prepare A
            self.get_frame(self.trajectory, i)
            self.cloth_sim.compute_hessian(self.tmp_vec)

            # prepare b
            if i == self.n_frame - 1:  # b_t = M^2 / h^4 / n_f^2 * (q_t - q*)
                self.b.from_numpy(self.cloth_model.target_verts)
                axpy(-1.0, self.tmp_vec, self.b)
                scale(self.b, -self.mass ** 2 / self.dt ** 4 / self.n_frame ** 2)
                # print("last b:")
                # print_field(self.b)
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
            self.conjugate_gradient(self.tmp_vec)
            self.set_frame(self.adjoint_vec, i)
            if self.b_verbose: print("lambda %i" % i)

        # update control forces (gradient descent with line search)
        print("[start line-search]")
        b_converge = self.line_search()

        return b_converge

    @ti.kernel
    def update_preconditioner(self):
        A = ti.static(self.cloth_sim.hessian)
        for i in range(self.n_vert):
            for j in ti.static(range(3)):
                self.inv_A_diag[i][j] = 1.0 / A[i, i][j, j]

    @ti.kernel
    def compute_Ap(self, x: ti.template()):
        A = ti.static(self.cloth_sim.hessian)
        for i, j in A:
            self.Ap[i] += A[i, j] @ x[j]

    def conjugate_gradient(self, x):
        # r = b - Ax (x's initial value is lambda from last epoch)
        self.r.copy_from(self.b)
        self.Ap.fill(0.0)
        self.compute_Ap(x)
        axpy(-1.0, self.Ap, self.r)

        # z and p
        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.update_preconditioner()
            element_wist_mul(self.inv_A_diag, self.r, self.z)
        else:
            self.z.copy_from(self.r)
        self.p.copy_from(self.z)

        # rTz
        reduce(self.rTz, self.r, self.z)
        # print("CG iter -1: %.1ef" % self.rTz[None])

        for i in range(self.cg_iter):
            self.Ap.fill(0.0)
            self.compute_Ap(self.p)

            # alpha
            reduce(self.sum, self.p, self.Ap)
            self.alpha[None] = self.rTz[None] / self.sum[None]

            # update x and r(z)
            axpy(self.alpha[None], self.p, x)
            axpy(-self.alpha[None], self.Ap, self.r)

            reduce(self.res, self.r, self.r)
            # print("CG iter % i: %.1ef" % (i, self.res[None]))
            if self.res[None] < self.cg_err:
                if self.b_verbose:
                    print("[CG converge] iter: %i, err: %.1ef" % (i, self.res[None]))
                break

            if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
                element_wist_mul(self.inv_A_diag, self.r, self.z)
            else:
                self.z.copy_from(self.r)

            # beta
            reduce(self.sum, self.r, self.z)
            self.beta[None] = self.sum[None] / self.rTz[None]
            self.rTz[None] = self.sum[None]

            scale(self.p, self.beta[None])
            axpy(1.0, self.z, self.p)
        else:
            if self.b_verbose:
                print("[CG not converge] err: %.1ef" % self.res[None])
