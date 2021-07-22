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
                 desc_rate, epsilon, n_frame,
                 ls_alpha=0.5, ls_gamma=0.03,
                 cg_precond="Jacobi",
                 cg_iter=100,
                 cg_err=1e-6,
                 b_display=False):

        assert n_frame >= 3

        self.cloth_model = cloth_model
        self.cloth_sim = cloth_sim

        self.desc_rate = desc_rate
        self.epsilon = epsilon
        self.ls_alpha = ls_alpha
        self.ls_gamma = ls_gamma

        self.n_frame = n_frame
        self.dt = cloth_sim.dt
        self.n_vert = cloth_sim.n_vert
        self.mass = cloth_sim.mass

        self.b_display = b_display

        self.trajectory = [ti.Vector.field(3, ti.f32, self.n_vert) for _ in range(n_frame)]
        self.control_force = [ti.Vector.field(3, ti.f32, self.n_vert) for _ in range(n_frame)]
        # In face 3 lambda are enough for computing
        # but if we want to do line-search and have better initial guess for CG, we'd better remember them all
        self.adjoint_vec = [ti.Vector.field(3, ti.f32, self.n_vert) for _ in range(n_frame)]

        # line-search
        self.loss = None
        self.step_size = 1.0
        self.gradient = [ti.Vector.field(3, ti.f32, self.n_vert) for _ in range(n_frame)]
        self.tentetive_ctrl_f = [ti.Vector.field(3, ti.f32, self.n_vert) for _ in range(n_frame)]  # FIXME: can reduce to one Vector.field (w/ more complicate line-search)

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
        for i in range(self.n_frame):
            self.trajectory[i].fill(0.0)
            self.control_force[i].fill(0.0)
            self.adjoint_vec[i].fill(0.0)
            self.gradient[i].fill(0.0)
            self.tentetive_ctrl_f[i].fill(0.0)

    def save_trajectory(self, output_dir):
        """
        Save a whole trajectory under checkpoint/timestamp/epoch_i/trajectory.ply
        """
        writer = ti.PLYWriter(self.n_frame * self.n_vert, self.n_frame * self.cloth_model.n_face, "tri")
        np_traj = np.vstack([x.to_numpy() for x in self.trajectory]).reshape(-1, 3)
        faces = np.tile(self.cloth_model.faces.flatten(), self.n_frame) + \
                np.repeat(self.n_vert * np.arange(self.n_frame), 3 * self.cloth_model.n_face)
        writer.add_vertex_pos(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2])
        writer.add_faces(faces)
        writer.export(os.path.join(output_dir, "trajectory.ply"))

    def save_frame(self, output_dir):
        """
        Save the model per frame under checkpoint/timestamp/epoch_i/frame_i.ply
        """
        for i in range(self.n_frame):
            writer = ti.PLYWriter(self.n_vert, self.cloth_model.n_face, "tri")
            np_verts = self.trajectory[i].to_numpy()
            writer.add_vertex_pos(np_verts[:, 0], np_verts[:, 1], np_verts[:, 2])
            writer.add_faces(self.cloth_model.faces)
            writer.export(os.path.join(output_dir, "frame_%i.ply" % i))

    def __forward(self, control_force):
        print("start forward")

        # reset initial states
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)

        for i in range(self.n_frame):
            self.cloth_sim.XPBD_step(control_force[i], self.trajectory[i])
            if self.b_display:
                self.cloth_model.update_scene(self.trajectory[i].to_numpy())
            print("[frame %i]" % i)

    def forward(self):
        """ Run frames of simulation """
        self.__forward(self.control_force)

    def __loss(self, control_force):
        """
        Compute loss (forward must be called before)
        L = 1/2 * M^2 / h^4 * ||q_{t+1} - q*||^2 + 1/2 * \epsilon * ||p||^2
        """
        loss = 0.0

        # regularization term
        for i in range(self.n_frame):
            reduce(self.sum, control_force[i], control_force[i])
            loss += self.sum[None]
        loss *= 0.5 * self.epsilon

        # target position term
        self.b.from_numpy(self.cloth_model.target_verts)
        axpy(-1.0, self.trajectory[self.n_frame - 1], self.b)
        reduce(self.sum, self.b, self.b)
        loss += 0.5 * self.mass ** 2 / self.dt ** 4 * self.sum[None]

        return loss

    @ti.kernel
    def __compute_gradient(self, g: ti.template(), ctrl_f: ti.template(), L: ti.template()):
        # dLdp = \epsilon * p + h^2 * \lambda
        for i in range(self.n_vert):
            g[i] = self.epsilon * ctrl_f[i] + self.dt ** 2 * L[i]

    def line_search(self):
        """ Find a proper step size.
        This method will invoke forward simulation several times, so don't need to call forward() explicitly anymore
        """
        # compute line search threshold
        if not self.loss:  # first epoch
            self.loss = self.__loss(self.control_force)
        threshold = 0.0
        for i in range(self.n_frame):
            # compute dLdp
            self.__compute_gradient(self.gradient[i], self.control_force[i], self.adjoint_vec[i])
            reduce(self.sum, self.gradient[i], self.gradient[i])
            threshold += self.sum[None]
        threshold *= self.ls_gamma * self.desc_rate

        # line-search
        step_size = min(1.0, self.step_size / self.ls_alpha)  # use step size from last epoch as initial guess
        b_converge = False
        while True:
            for i in range(self.n_frame):
                self.tentetive_ctrl_f[i].copy_from(self.control_force[i])
                axpy(-step_size * self.desc_rate, self.gradient[i], self.tentetive_ctrl_f[i])

            # update trajectory
            self.__forward(self.tentetive_ctrl_f)
            cur_loss = self.__loss(self.tentetive_ctrl_f)

            print("step size: %f  loss: %.3ef" % (step_size, cur_loss))

            if cur_loss < self.loss + step_size * threshold:  # right step size
                break
            step_size *= self.ls_alpha
            if step_size < 1e-5:
                b_converge = True
                break

        # commit control force
        self.step_size = step_size
        self.loss = cur_loss
        for i in range(self.n_frame):
            self.control_force[i].copy_from(self.tentetive_ctrl_f[i])

        return b_converge

    def backward(self):
        """ Update control forces """
        # compute lambda
        for i in range(self.n_frame - 1, -1, -1):
            # prepare A
            self.cloth_sim.compute_hessian(self.trajectory[i])

            # prepare b
            if i == self.n_frame - 1:  # b_t = M^2 / h^4 * (q_t - q*)
                self.b.from_numpy(self.cloth_model.target_verts)
                M2h4 = self.mass ** 2 / self.dt ** 4
                scale(self.b, -M2h4)
                axpy(M2h4, self.trajectory[i], self.b)
                # print("last b:")
                # print_field(self.b)
            elif i == self.n_frame - 2:  # b_{t-1} = 2 * M * lambda_t
                self.b.fill(0.0)
                axpy(2 * self.mass, self.adjoint_vec[i + 1], self.b)
            else:  # b_i = 2 * M * lambda_{i+1} - M * lambda_{i+2}
                self.b.fill(0.0)
                axpy(2 * self.mass, self.adjoint_vec[i + 1], self.b)
                axpy(-self.mass, self.adjoint_vec[i + 2], self.b)

            # linear solve
            self.conjugate_gradient(self.adjoint_vec[i])

        # update control forces (gradient descent with line search)
        print("start line-search")
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
            if self.res[None] < self.cg_err:
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
            print("[CG not converge] err: %f" % self.res[None])
