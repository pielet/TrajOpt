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
                 desc_rate, epsilon, n_frame,
                 cloth_model, cloth_sim,
                 b_display,
                 b_save_frame=True,
                 cg_precond="Jacobi",
                 cg_iter=100,
                 cg_err=1e-6):

        assert n_frame >= 3

        self.desc_rate = desc_rate
        self.epsilon = epsilon

        self.n_frame = n_frame
        self.dt = cloth_sim.dt
        self.n_vert = cloth_sim.n_vert
        self.mass = cloth_sim.mass

        self.cloth_model = cloth_model
        self.cloth_sim = cloth_sim

        self.b_display = b_display
        self.b_save_frame = b_save_frame

        self.trajectory = [ti.Vector.field(3, float, self.n_vert) for _ in range(n_frame)]
        self.control_force = [ti.Vector.field(3, float, self.n_vert) for _ in range(n_frame)]

        self.adjoint_vec = [ti.Vector.field(3, float, self.n_vert) for _ in range(3)]  # 3 lambda are enough

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
            self.inv_A_diag = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vert)

    def initialize(self):
        for i in range(self.n_frame):
            self.trajectory[i].fill(0.0)
            self.control_force[i].fill(0.0)

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

    def save_frame(self, output_dir, frame, verts):
        writer = ti.PLYWriter(self.n_vert, self.cloth_model.n_face, "tri")
        writer.add_vertex_pos(verts[:, 0], verts[:, 1], verts[:, 2])
        writer.add_faces(self.cloth_model.faces)
        writer.export(os.path.join(output_dir, "frame_%i.ply" % frame))

    def forward(self, epoch, output_dir):
        """ Run frames of simulation and record trajectory """
        # reset initial states
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)

        for i in range(self.n_frame):
            self.cloth_sim.XPBD_step(self.control_force[i], self.trajectory[i])

            # save updated positions
            np_verts = self.trajectory[i].to_numpy()
            if self.b_display:
                self.cloth_model.update_scene(np_verts)
            if self.b_save_frame:
                self.save_frame(output_dir, i, np_verts)

            print("[epoch %i][frame %i]" % (epoch, i))

        print("Finish forward.")

    def loss(self):
        """
        Compute loss (self.forward must be called before)
        L = 1/2 * M^2 / h^4 * ||q_{t+1} - q*||^2 + 1/2 * \epsilon * ||p||^2
        """
        loss = 0.0

        # regularization term
        for i in range(self.n_frame):
            reduce(self.sum, self.control_force[i], self.control_force[i])
            loss += self.sum[None]
        loss *= 0.5 * self.epsilon

        # target position term
        self.b.from_numpy(self.cloth_model.target_verts)
        axpy(-1.0, self.trajectory[self.n_frame - 1], self.b)
        reduce(self.sum, self.b, self.b)
        loss += 0.5 * self.mass ** 2 / self.dt ** 4 * self.sum[None]

        return loss

    @ti.kernel
    def update_control_force(self, control_force: ti.template()):
        # dLdp = \epsilon * p + h^2 * \lambda
        for i in range(self.n_vert):
            control_force[i] -= self.desc_rate * (self.epsilon * control_force[i] + self.dt ** 2 * self.adjoint_vec[0][i])

    def backward(self):
        """ Update control forces """
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
                axpy(2 * self.mass, self.adjoint_vec[1], self.b)
            else:  # b_{i+1} = 2 * M * lambda_i - M * lambda_{i+1}
                self.b.fill(0.0)
                axpy(2 * self.mass, self.adjoint_vec[1], self.b)
                axpy(-self.mass, self.adjoint_vec[2], self.b)

            # linear solve
            self.conjugate_gradient()

            # update control force
            self.update_control_force(self.control_force[i])

            # move lambda
            # print("lambda: ")
            # print_field(self.adjoint_vec[0])
            # print("control force: ")
            # print_field(self.control_force[i])
            self.adjoint_vec[2].copy_from(self.adjoint_vec[1])
            self.adjoint_vec[1].copy_from(self.adjoint_vec[0])


    @ti.kernel
    def update_preconditioner(self):
        A = ti.static(self.cloth_sim.hessian)
        for i in range(self.n_vert):
            for j in ti.static(range(3)):
                self.inv_A_diag[i][j] = 1.0 / A[i, i][j, j]

    @ti.kernel
    def compute_Ap(self):
        A = ti.static(self.cloth_sim.hessian)
        for i, j in A:
            self.Ap[i] += A[i, j] @ self.p[j]

    def conjugate_gradient(self):
        x = ti.static(self.adjoint_vec[0])  # alias

        x.fill(0.0)  # TODO: maybe some better initial guess
        self.r.copy_from(self.b)
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
            self.compute_Ap()

            # alpha
            reduce(self.sum, self.p, self.Ap)
            self.alpha[None] = self.rTz[None] / self.sum[None]

            # update x and r(z)
            axpy(self.alpha[None], self.p, x)
            axpy(-self.alpha[None], self.Ap, self.r)

            reduce(self.res, self.r, self.r)
            if self.res[None] < self.cg_err:
                print("[CG converge] iter: %i, err: %f" % (i, self.res[None]))
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
