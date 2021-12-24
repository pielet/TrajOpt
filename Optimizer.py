import os
import numpy as np
import taichi as ti
from FieldUtil import *
from LinearSolver import LinearSolver

@ti.data_oriented
class Optimizer:
    OptimizeMethod = {
        "gradient": 0,             # f (soft constrain)
        "projected Newton": 1,   # f (linearized hard constrain)
        "SAP": 2,          # f (linearized constrain)
        "Gauss-Newton": 3  # x
    }

    def __init__(self,
                 cloth_model, cloth_sim, linear_solver, opt_med, n_frame,
                 sim_med, desc_rate, smoothness,
                 init_med, opt_integration_med, linear_solver_type, b_matrix_free, cg_precond, cg_iter_ratio, cg_err,
                 ls_alpha=0.5, ls_gamma=0.03,
                 b_verbose=False):

        assert n_frame >= 3

        self.b_verbose = b_verbose
        self.linear_solver_type = linear_solver_type
        self.b_matrix_free = b_matrix_free

        self.cloth_model = cloth_model
        self.cloth_sim = cloth_sim

        self.method = self.OptimizeMethod[opt_med]
        self.init_med = init_med
        self.opt_integration_med = opt_integration_med

        self.n_frame = n_frame
        self.dt = cloth_sim.dt
        self.n_vert = cloth_sim.n_vert
        self.cur_epoch = 0

        self.x0 = ti.Vector.field(3, ti.f32, self.n_vert)
        self.x1 = ti.Vector.field(3, ti.f32, self.n_vert)
        self.gravity_force = ti.Vector.field(3, ti.f32, self.n_vert)

        self.trajectory = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.control_force = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)

        # forced based optimization
        self.sim_med = sim_med
        self.desc_rate = desc_rate
        self.epsilon = smoothness

        ## adjoint method
        self.adjoint_vec = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.linear_solver = linear_solver
        self.b = linear_solver.b

        # position based optimization
        self.ELeq = ti.Vector.field(3, ti.f32, self.n_vert)

        self.hess = ti.Matrix.field(3, 3, ti.f32)
        if self.b_matrix_free:
            ti.root.dense(ti.i, n_frame).pointer(ti.jk, self.n_vert).place(self.hess)
        else:
            ti.root.pointer(ti.ij, n_frame * self.n_vert).place(self.hess)
            self.tmp_hess = ti.Matrix.field(3, 3, ti.f32)
            self.tmp_hess_pointer = ti.root.pointer(ti.ij, self.n_vert)
            self.tmp_hess_pointer.place(self.tmp_hess)
        if self.linear_solver_type == "cg":
            self.x_linear_solver = LinearSolver((n_frame - 2) * self.n_vert, cg_precond,
                                                int(cg_iter_ratio * (n_frame - 2) * self.n_vert * 3), cg_err)

        self.x_bar = ti.Vector.field(3, ti.f32, self.n_vert)
        self.extend_x = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.extend_Ax = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.x_ascent_dir = ti.Vector.field(3, ti.f32, (n_frame - 2) * self.n_vert)

        # line-search
        self.ls_alpha = ls_alpha
        self.ls_gamma = ls_gamma

        self.loss = 0.0
        self.force_loss = 0.0
        self.constrain_loss = 0.0
        self.step_size = 1.0
        self.gradient = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.ascent_dir = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)
        self.tentative = ti.Vector.field(3, ti.f32, n_frame * self.n_vert)

        self.tmp_vec = ti.Vector.field(3, ti.f32, self.n_vert)  # to access one frame data
        self.sum = ti.field(ti.f32, ())

        self.sparsity = 0
        self.loss_list = []
        self.loss_per_frame = []

    def initialize(self):
        self.mass = self.cloth_sim.mass
        self.control_force.fill(0.0)
        self.adjoint_vec.fill(0.0)
        self.ascent_dir.fill(0.0)
        self.gradient.fill(0.0)
        self.tentative.fill(0.0)
        self.gravity_force.from_numpy(self.mass * np.stack([self.cloth_sim.gravity.to_numpy()] * self.n_vert))

        # count sparsity
        A_nz = self.cloth_sim.sparsity
        sparse_set_zero(self.tmp_hess)
        sparse_ATA(self.n_vert, self.cloth_sim.hessian, self.cloth_sim.hessian_pointer, self.tmp_hess)
        ATA_nz = sparsity(self.tmp_hess, self.tmp_hess_pointer)
        print(f"A sparsity: {A_nz}, ATA sparsity: {ATA_nz}")

        self.sparsity = self.n_frame * (ATA_nz + 4 * A_nz + 4 * 3 * self.n_vert)

        # compute x0 and x1
        self.x0.from_numpy(self.cloth_model.verts)
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)
        self.tmp_vec.fill(0.0)
        self.cloth_sim.step(self.tmp_vec, self.tmp_vec)
        self.x1.copy_from(self.tmp_vec)

    @ti.kernel
    def get_frame(self, field: ti.template(), fi: ti.i32, frame_vec: ti.template()):
        for i in frame_vec:
            frame_vec[i] = field[fi * self.n_vert + i]

    @ti.kernel
    def set_frame(self, field: ti.template(), fi: ti.i32, frame_vec: ti.template()):
        for i in frame_vec:
            field[fi * self.n_vert + i] = frame_vec[i]

    @ti.kernel
    def add_to_frame(self, field: ti.template(), fi: ti.i32, frame_vec: ti.template()):
        for i in frame_vec:
            field[fi * self.n_vert + i] += frame_vec[i]

    @ti.kernel
    def set_hessian(self, fi: ti.i32, A: ti.template()):
        for i, j in A:
            self.hess[fi, i, j] = A[i, j]

    @ti.kernel
    def get_hessian(self, fi: ti.i32, A: ti.template()):
        for i, j in A:
            A[i, j] = self.hess[fi, i, j]

    @ti.kernel
    def add_in_block_diagonal(self, fi: ti.i32, fj: ti.i32, a: ti.f32):
        for i in range(self.n_vert):
            self.hess[fi * self.n_vert + i, fj * self.n_vert + i] += a * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def add_in_block_sparse(self, fi: ti.i32, fj: ti.i32, A: ti.template()):
        for i, j in A:
            self.hess[fi * self.n_vert + i, fj * self.n_vert + j] += A[i, j]

    def save_trajectory(self, output_dir, b_save_npy):
        """
        Save a whole trajectory under checkpoint/timestamp/epoch_i/trajectory.ply
        """
        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            writer = ti.PLYWriter(self.n_frame * self.n_vert, self.n_frame * self.cloth_model.n_face, "tri")

            np_traj = np.vstack([self.x0.to_numpy(), self.x1.to_numpy(), self.trajectory.to_numpy()[:-2 * self.n_vert]])
            faces = np.tile(self.cloth_model.faces.flatten(), self.n_frame) + \
                    np.repeat(self.n_vert * np.arange(self.n_frame), 3 * self.cloth_model.n_face)
            np_force = np.roll(self.control_force.to_numpy(), 2 * self.n_vert, axis=0)

            writer.add_vertex_pos(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2])
            writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
            writer.add_faces(faces)
            writer.export(os.path.join(output_dir, "trajectory.ply"))

            if b_save_npy:
                np.save(os.path.join(output_dir, "trajectory.npy"), np_traj)
                np.save(os.path.join(output_dir, "control_force.npy"), np_force)
        else:
            writer = ti.PLYWriter((self.n_frame + 2) * self.n_vert, (self.n_frame + 2) * self.cloth_model.n_face, "tri")

            np_traj = np.vstack([self.x0.to_numpy(), self.x1.to_numpy(), self.trajectory.to_numpy()])
            faces = np.tile(self.cloth_model.faces.flatten(), self.n_frame + 2) + \
                    np.repeat(self.n_vert * np.arange(self.n_frame + 2), 3 * self.cloth_model.n_face)
            np_force = np.vstack([np.zeros((self.n_vert, 3)), self.control_force.to_numpy(), np.zeros((self.n_vert, 3))])

            writer.add_vertex_pos(np_traj[:, 0], np_traj[:, 1], np_traj[:, 2])
            writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
            writer.add_faces(faces)
            writer.export(os.path.join(output_dir, "trajectory.ply"))

            if b_save_npy:
                np.save(os.path.join(output_dir, "trajectory.npy"), np_traj[:-2 * self.n_vert])
                np.save(os.path.join(output_dir, "control_force.npy"), self.control_force.to_numpy())

    def save_frame(self, output_dir):
        """
        Save the model per frame under checkpoint/timestamp/epoch_i/frame_i.ply
        """
        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            for i in range(self.n_frame):
                writer = ti.PLYWriter(self.n_vert, self.cloth_model.n_face, "tri")

                if i == 0:
                    np_verts = self.x0.to_numpy()
                elif i == 1:
                    np_verts = self.x1.to_numpy()
                else:
                    self.get_frame(self.trajectory, i - 2, self.tmp_vec)
                    np_verts = self.tmp_vec.to_numpy()

                self.get_frame(self.control_force, (i - 2) % self.n_frame, self.tmp_vec)
                np_force = self.tmp_vec.to_numpy()

                writer.add_vertex_pos(np_verts[:, 0], np_verts[:, 1], np_verts[:, 2])
                writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
                writer.add_faces(self.cloth_model.faces)
                writer.export_frame(i, os.path.join(output_dir, "frame"))
        else:
            for i in range(self.n_frame + 2):
                writer = ti.PLYWriter(self.n_vert, self.cloth_model.n_face, "tri")

                if i == 0:
                    np_verts = self.x0.to_numpy()
                elif i == 1:
                    np_verts = self.x1.to_numpy()
                else:
                    self.get_frame(self.trajectory, i - 2, self.tmp_vec)
                    np_verts = self.tmp_vec.to_numpy()

                if i == 0 or i == self.n_frame + 1:
                    np_force = np.zeros((self.n_vert, 3))
                else:
                    self.get_frame(self.control_force, i - 1, self.tmp_vec)
                    np_force = self.tmp_vec.to_numpy()

                writer.add_vertex_pos(np_verts[:, 0], np_verts[:, 1], np_verts[:, 2])
                writer.add_vertex_normal(np_force[:, 0], np_force[:, 1], np_force[:, 2])
                writer.add_faces(self.cloth_model.faces)
                writer.export_frame(i, os.path.join(output_dir, "frame"))

    def compute_init_loss(self):
        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            if self.init_med == "static":
                for i in range(self.n_frame):
                    self.set_frame(self.trajectory, i, self.x1)
            elif self.init_med == "solve":
                self.forward(self.control_force)  # 0 force
            elif self.init_med == "load":
                self.trajectory.from_numpy(np.roll(self.cloth_model.init_traj, -2 * self.n_vert, axis=0))
                self.control_force.from_numpy(np.roll(self.cloth_model.init_force, ))

            self.loss, loss_info = self.compute_position_loss(self.trajectory)
            self.loss_list.append(self.loss)
            self.loss_per_frame.append(loss_info)
        else:
            self.loss, loss_info = self.compute_soft_constrain_loss(self.control_force)
            self.loss_list.append([self.loss, self.force_loss, self.constrain_loss])
            self.loss_per_frame.append(loss_info)

    def forward(self, control_force):
        print("[start forward]", end=" ")

        # reset initial states
        self.cloth_sim.x.from_numpy(self.cloth_model.verts)
        self.cloth_sim.v.fill(0.0)
        self.tmp_vec.fill(0.0)
        self.cloth_sim.step(self.tmp_vec, self.tmp_vec) # skip x1

        frame, err = 0, 0
        for i in range(self.n_frame):
            if self.b_verbose: print(f"frame {i}")
            self.get_frame(control_force, i, self.tmp_vec)
            fi, ei = self.cloth_sim.step(self.tmp_vec, self.tmp_vec)  # FIXME: assume ext_f and x_next can be the same vector
            frame += fi
            err += ei
            self.set_frame(self.trajectory, i, self.tmp_vec)

        print("avg.iter: %i, avg.err: %.1e" % (frame / self.n_frame, err / self.n_frame))

        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            self.set_frame(self.trajectory, self.n_frame - 2, self.x0)
            self.set_frame(self.trajectory, self.n_frame - 1, self.x1)

    def compute_soft_constrain_loss(self, control_force):
        """
        L = 1/2 * (M / n_f / h^2)^2 * ||q_{t+1} - q*||^2 + 1/2 * \epsilon * ||p||^2
        """
        loss_per_frame = []
        self.forward(control_force)

        # regularization term
        reduce(self.sum, control_force, control_force)
        self.force_loss = 0.5 * self.epsilon * self.sum[None]
        for i in range(self.n_frame):
            self.get_frame(self.control_force, i, self.tmp_vec)
            reduce(self.sum, self.tmp_vec, self.tmp_vec)
            loss_per_frame.append(0.5 * self.epsilon * self.sum[None])

        # target position term
        self.get_frame(self.trajectory, self.n_frame - 2, self.tmp_vec)
        axpy(-1.0, self.x0, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        self.constrain_loss = 0.5 * (self.mass / self.n_frame / self.dt ** 2) ** 2 * self.sum[None]

        self.get_frame(self.trajectory, self.n_frame - 1, self.tmp_vec)
        axpy(-1.0, self.x1, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        self.constrain_loss += 0.5 * (self.mass / self.n_frame / self.dt ** 2) ** 2 * self.sum[None]

        return self.force_loss + self.constrain_loss, loss_per_frame

    def compute_soft_constrain_ascent_dir(self):
        self.ascent_dir.copy_from(self.adjoint_vec)
        axpy(self.epsilon, self.control_force, self.ascent_dir)
        scale(self.ascent_dir, self.desc_rate)
        self.gradient.copy_from(self.ascent_dir)

    def compute_position_loss(self, trajectory):
        loss = 0.0
        loss_per_frame = []

        for i in range(self.n_frame):
            self.get_frame(trajectory, (i + 1) % self.n_frame, self.ELeq)
            self.get_frame(trajectory, (i - 1) % self.n_frame, self.tmp_vec)
            axpy(1.0, self.tmp_vec, self.ELeq)
            self.get_frame(trajectory, i, self.tmp_vec)
            axpy(-2.0, self.tmp_vec, self.ELeq)
            scale(self.ELeq, self.mass / (self.dt * self.dt))

            if self.opt_integration_med == "implicit":
                self.get_frame(trajectory, (i + 1) % self.n_frame, self.tmp_vec)
            self.cloth_sim.compute_gradient(self.tmp_vec, False)
            axpy(1.0, self.cloth_sim.gradient, self.ELeq)

            axpy(-1.0, self.gravity_force, self.ELeq)

            reduce(self.sum, self.ELeq, self.ELeq)
            loss += 0.5 * self.sum[None]
            loss_per_frame.append(0.5 * self.sum[None])

        return loss, loss_per_frame

    def compute_virtual_force(self):
        for i in range(self.n_frame):
            self.get_frame(self.trajectory, (i + 1) % self.n_frame, self.ELeq)
            self.get_frame(self.trajectory, (i - 1) % self.n_frame, self.tmp_vec)
            axpy(1.0, self.tmp_vec, self.ELeq)
            self.get_frame(self.trajectory, i, self.tmp_vec)
            axpy(-2.0, self.tmp_vec, self.ELeq)
            scale(self.ELeq, self.mass / (self.dt * self.dt))

            if self.opt_integration_med == "implicit":
                self.get_frame(self.trajectory, (i + 1) % self.n_frame, self.tmp_vec)
            self.cloth_sim.compute_gradient(self.tmp_vec, False)
            axpy(1.0, self.cloth_sim.gradient, self.ELeq)

            axpy(-1.0, self.gravity_force, self.ELeq)

            self.set_frame(self.control_force, i, self.ELeq)

    def compute_position_ascent_dir(self):
        if self.linear_solver_type == "cg":
            self.position_solve_cg()
        elif self.linear_solver_type == "direct":
            self.position_solve_direct()

    def position_solve_cg(self):
        print("[CG]: compute gradient and hessian")
        self.extend_x.fill(0.0)
        M_h2 = self.mass / (self.dt * self.dt)
        # np.save("debug/M_h2.npy", [M_h2])
        for i in range(self.n_frame):
            # Euler-Lagragian eqution
            self.get_frame(self.trajectory, (i + 1) % self.n_frame, self.ELeq)
            self.get_frame(self.trajectory, (i - 1) % self.n_frame, self.tmp_vec)
            axpy(1.0, self.tmp_vec, self.ELeq)
            self.get_frame(self.trajectory, i, self.tmp_vec)
            axpy(-2.0, self.tmp_vec, self.ELeq)
            scale(self.ELeq, M_h2)

            self.cloth_sim.compute_gradient(self.tmp_vec, False)
            # save(f"iter_{self.cur_epoch}_x_{i}.npy", self.tmp_vec)
            # save(f"iter_{self.cur_epoch}_grad_{i}.npy", self.cloth_sim.gradient)
            axpy(1.0, self.cloth_sim.gradient, self.ELeq)

            # accumulate to adjacent gradient
            self.cloth_sim.compute_hessian(self.tmp_vec, False)  # self.tmp_vec = x_i
            # save(f"iter_{self.cur_epoch}_hess_{i}.npy", self.cloth_sim.hessian)
            if self.b_matrix_free:
                self.set_hessian(i, self.cloth_sim.hessian)

            self.tmp_vec.copy_from(self.ELeq)
            scale(self.tmp_vec, M_h2)  # self.tmp_vec = M/h^2 * ELeq
            self.add_to_frame(self.extend_x, (i - 1) % self.n_frame, self.tmp_vec)
            self.add_to_frame(self.extend_x, (i + 1) % self.n_frame, self.tmp_vec)
            scale(self.tmp_vec, -2.0)  # self.tmp_vec = -2M/h^2 * ELeq
            self.add_to_frame(self.extend_x, i, self.tmp_vec)

            self.tmp_vec.fill(0.0)
            sparse_mv(self.cloth_sim.hessian, self.ELeq, self.tmp_vec)  # assume \grad^2 U is symmetric
            self.add_to_frame(self.extend_x, i, self.tmp_vec)

            if not self.b_matrix_free:
                self.cloth_sim.add_in_diagonal(-2.0 * M_h2)

                # M/h^2 & M/h^2
                self.add_in_block_diagonal((i - 1) % self.n_frame, (i - 1) % self.n_frame, M_h2 * M_h2)
                self.add_in_block_diagonal((i - 1) % self.n_frame, (i + 1) % self.n_frame, M_h2 * M_h2)
                self.add_in_block_diagonal((i + 1) % self.n_frame, (i - 1) % self.n_frame, M_h2 * M_h2)
                self.add_in_block_diagonal((i + 1) % self.n_frame, (i + 1) % self.n_frame, M_h2 * M_h2)

                # M/h^2 * A
                sparse_set_zero(self.tmp_hess)
                sparse_copy(self.cloth_sim.hessian, self.tmp_hess)
                scale(self.tmp_hess, M_h2)
                self.add_in_block_sparse(i, (i - 1) % self.n_frame, self.tmp_hess)
                self.add_in_block_sparse(i, (i + 1) % self.n_frame, self.tmp_hess)
                self.add_in_block_sparse((i - 1) % self.n_frame, i, self.tmp_hess)
                self.add_in_block_sparse((i + 1) % self.n_frame, i, self.tmp_hess)

                # A^T @ A
                sparse_set_zero(self.tmp_hess)
                sparse_ATA(self.n_vert, self.cloth_sim.hessian, self.cloth_sim.hessian_pointer, self.tmp_hess)
                self.add_in_block_sparse(i, i, self.tmp_hess)

        # save(f"iter_{self.cur_epoch}_full_grad.npy", self.extend_x)
        large_to_small(self.extend_x, self.x_linear_solver.b)
        if not self.b_matrix_free:
            # save(f"iter_{self.cur_epoch}_full_hess.npy", self.hess)
            large_to_small_sparse(self.hess, self.x_linear_solver.A, (self.n_frame - 2) * self.n_vert)

        def ATAx(x, Ax):
            # extent x
            self.extend_x.fill(0.0)
            small_to_large(x, self.extend_x)

            self.extend_Ax.fill(0.0)
            for i in range(self.n_frame):
                # Ax
                self.get_frame(self.extend_x, (i + 1) % self.n_frame, self.x_bar)
                self.get_frame(self.extend_x, (i - 1) % self.n_frame, self.tmp_vec)
                axpy(1.0, self.tmp_vec, self.x_bar)
                self.get_frame(self.extend_x, i, self.tmp_vec)  # self.tmp_vec = \delta x_i
                axpy(-2.0, self.tmp_vec, self.x_bar)
                scale(self.x_bar, self.mass / (self.dt * self.dt))

                self.get_hessian(i, self.cloth_sim.hessian)

                self.ELeq.fill(0.0)
                sparse_mv(self.cloth_sim.hessian, self.tmp_vec, self.ELeq)
                axpy(1.0, self.ELeq, self.x_bar)

                # ATAx
                self.tmp_vec.copy_from(self.x_bar)
                scale(self.tmp_vec, self.mass / (self.dt * self.dt))  # self.tmp_vec = M/h^2 * x_bar
                self.add_to_frame(self.extend_Ax, (i - 1) % self.n_frame, self.tmp_vec)
                self.add_to_frame(self.extend_Ax, (i + 1) % self.n_frame, self.tmp_vec)
                scale(self.tmp_vec, -2.0)  # self.tmp_vec = -2M/h^2 * ELeq
                self.add_to_frame(self.extend_Ax, i, self.tmp_vec)

                self.tmp_vec.fill(0.0)
                sparse_mv(self.cloth_sim.hessian, self.x_bar, self.tmp_vec)
                self.add_to_frame(self.extend_Ax, i, self.tmp_vec)

            # clamp Ax
            large_to_small(self.extend_Ax, Ax)

        if self.b_matrix_free:
            self.x_linear_solver.compute_Ap = ATAx

        # linear solve
        print("[CG]: start")
        self.x_ascent_dir.fill(0.0)
        self.x_linear_solver.conjugate_gradient(self.x_ascent_dir, b_verbose=True)
        self.ascent_dir.fill(0.0)
        small_to_large(self.x_ascent_dir, self.ascent_dir)
        # print("[start direct solver]")
        # np_ascent_dir = np.linalg.solve(np.swapaxes(self.x_linear_solver.A.to_numpy(), 1, 2).reshape([3 * self.n_vert * (self.n_frame - 2), -1]), self.x_linear_solver.b.to_numpy().flatten())
        # self.x_ascent_dir.from_numpy(np_ascent_dir.reshape([-1, 3]))
        # small_to_large(self.x_ascent_dir, self.ascent_dir)

    def position_solve_direct(self):
        @ti.kernel
        def add_in_block_diagonal(A: ti.linalg.sparse_matrix_builder(), fi: ti.i32, fj: ti.i32, a: ti.f32):
            for i in range(3 * self.n_vert):
                A[3 * self.n_vert * fi + i, 3 * self.n_vert * fj + i] += a

        @ti.kernel
        def add_in_block_sparse(A: ti.linalg.sparse_matrix_builder(), fi: ti.i32, fj: ti.i32, A_block: ti.template()):
            for i, j in A_block:
                for ii, jj in ti.static(ti.ndrange(3, 3)):
                    A[3 * self.n_vert * fi + 3 * i + ii, 3 * self.n_vert * fj + 3 * j + jj] += A_block[i, j][ii, jj]

        print("[LLT]: compute gradient and hessian")

        self.extend_x.fill(0.0)
        hess_builder = ti.linalg.SparseMatrixBuilder(3 * (self.n_frame - 2) * self.n_vert, 3 * (self.n_frame - 2) * self.n_vert,
                                                     max_num_triplets=self.sparsity)
        M_h2 = self.mass / (self.dt * self.dt)
        # np.save("debug/M_h2.npy", [M_h2])
        for i in range(self.n_frame):
            # Euler-Lagragian eqution
            self.get_frame(self.trajectory, (i + 1) % self.n_frame, self.ELeq)
            self.get_frame(self.trajectory, (i - 1) % self.n_frame, self.tmp_vec)
            axpy(1.0, self.tmp_vec, self.ELeq)
            self.get_frame(self.trajectory, i, self.tmp_vec)
            axpy(-2.0, self.tmp_vec, self.ELeq)
            scale(self.ELeq, M_h2)

            if self.opt_integration_med == "implicit":
                self.get_frame(self.trajectory, (i + 1) % self.n_frame, self.tmp_vec)
            self.cloth_sim.compute_gradient(self.tmp_vec, False)
            axpy(1.0, self.cloth_sim.gradient, self.ELeq)
            axpy(-1.0, self.gravity_force, self.ELeq)

            # save(f"iter_{self.cur_epoch}_x_{i}.npy", self.tmp_vec)

            self.cloth_sim.compute_hessian(self.tmp_vec, False)  # self.tmp_vec = x_i
            # save(f"iter_{self.cur_epoch}_hess_{i}.npy", self.cloth_sim.hessian)
            if self.opt_integration_med == "symplectic":
                self.cloth_sim.add_in_diagonal(-2.0 * M_h2)
            elif self.opt_integration_med == "implicit":
                self.cloth_sim.add_in_diagonal(M_h2)

            # accumulate gradient
            self.tmp_vec.copy_from(self.ELeq)  # self.tmp_vec = ELeq
            if self.opt_integration_med == "symplectic":
                scale(self.tmp_vec, M_h2)  # self.tmp_vec = M/h^2 * ELeq
                self.add_to_frame(self.extend_x, (i - 1) % self.n_frame, self.tmp_vec)
                self.add_to_frame(self.extend_x, (i + 1) % self.n_frame, self.tmp_vec)

                self.tmp_vec.fill(0.0)
                sparse_mv(self.cloth_sim.hessian, self.ELeq, self.tmp_vec)  # assume \grad^2 U is symmetric
                self.add_to_frame(self.extend_x, i, self.tmp_vec)
            elif self.opt_integration_med == "implicit":
                scale(self.tmp_vec, M_h2)  # self.tmp_vec = M/h^2 * ELeq
                self.add_to_frame(self.extend_x, (i - 1) % self.n_frame, self.tmp_vec)
                scale(self.tmp_vec, -2.0)  # self.tmp_vec = -2.0 * M/h^2 * ELeq
                self.add_to_frame(self.extend_x, i, self.tmp_vec)

                self.tmp_vec.fill(0.0)
                sparse_mv(self.cloth_sim.hessian, self.ELeq, self.tmp_vec)  # assume \grad^2 U is symmetric
                self.add_to_frame(self.extend_x, (i + 1) % self.n_frame, self.tmp_vec)

            # accumulate hessian
            if self.opt_integration_med == "symplectic":
                ## M/h^2 & M/h^2
                if (i - 1) % self.n_frame < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i - 1) % self.n_frame, (i - 1) % self.n_frame, M_h2 * M_h2)
                if (i - 1) % self.n_frame < self.n_frame - 2 and (i + 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i - 1) % self.n_frame, (i + 1) % self.n_frame, M_h2 * M_h2)
                if (i + 1) % self.n_frame < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i + 1) % self.n_frame, (i - 1) % self.n_frame, M_h2 * M_h2)
                if (i + 1) % self.n_frame < self.n_frame - 2 and (i + 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i + 1) % self.n_frame, (i + 1) % self.n_frame, M_h2 * M_h2)

                ## M/h^2 * A
                scale(self.cloth_sim.hessian, M_h2)  # self.cloth_sim.hessian is A-sparse  # self.cloth_sim.hessian is A-sparse
                if i < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, i, (i - 1) % self.n_frame, self.cloth_sim.hessian)
                if i < self.n_frame - 2 and (i + 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, i, (i + 1) % self.n_frame, self.cloth_sim.hessian)
                if (i - 1) % self.n_frame < self.n_frame - 2 and i < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, (i - 1) % self.n_frame, i, self.cloth_sim.hessian)
                if (i + 1) % self.n_frame < self.n_frame - 2 and i < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, (i + 1) % self.n_frame, i, self.cloth_sim.hessian)

                # A^T @ A
                if i < self.n_frame - 2:
                    scale(self.cloth_sim.hessian, 1.0 / M_h2)
                    sparse_set_zero(self.tmp_hess)  # self.tmp_hess is ATA-sparse
                    sparse_ATA(self.n_vert, self.cloth_sim.hessian, self.cloth_sim.hessian_pointer, self.tmp_hess)
                    add_in_block_sparse(hess_builder, i, i, self.tmp_hess)

            elif self.opt_integration_med == "implicit":
                ## M/h^2 & M/h^2
                if (i - 1) % self.n_frame < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i - 1) % self.n_frame, (i - 1) % self.n_frame, M_h2 * M_h2)
                if (i - 1) % self.n_frame < self.n_frame - 2 and i < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, (i - 1) % self.n_frame, i, -2.0 * M_h2 * M_h2)
                if i < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, i, (i - 1) % self.n_frame, -2.0 * M_h2 * M_h2)
                if i < self.n_frame - 2:
                    add_in_block_diagonal(hess_builder, i, i, 4.0 * M_h2 * M_h2)

                ## M/h^2 * A
                scale(self.cloth_sim.hessian, M_h2)  # self.cloth_sim.hessian is A-sparse  # self.cloth_sim.hessian is A-sparse
                if (i - 1) % self.n_frame < self.n_frame - 2 and (i + 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, (i - 1) % self.n_frame, (i + 1) % self.n_frame, self.cloth_sim.hessian)
                if (i + 1) % self.n_frame < self.n_frame - 2 and (i - 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, (i + 1) % self.n_frame, (i - 1) % self.n_frame, self.cloth_sim.hessian)
                    scale(self.cloth_sim.hessian, -2.0)
                if i < self.n_frame - 2 and (i + 1) % self.n_frame < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, i, (i + 1) % self.n_frame, self.cloth_sim.hessian)
                if (i + 1) % self.n_frame < self.n_frame - 2 and i < self.n_frame - 2:
                    add_in_block_sparse(hess_builder, (i + 1) % self.n_frame, i, self.cloth_sim.hessian)

                # A^T @ A
                if (i + 1) % self.n_frame < self.n_frame - 2:
                    scale(self.cloth_sim.hessian, -0.5 / M_h2)
                    sparse_set_zero(self.tmp_hess)  # self.tmp_hess is ATA-sparse
                    sparse_ATA(self.n_vert, self.cloth_sim.hessian, self.cloth_sim.hessian_pointer, self.tmp_hess)
                    add_in_block_sparse(hess_builder, (i + 1) % self.n_frame, (i + 1) % self.n_frame, self.tmp_hess)

        large_to_small(self.extend_x, self.x_ascent_dir)
        # save(f"iter_{self.cur_epoch}_full_grad.npy", self.extend_x)
        self.gradient.fill(0.0)
        small_to_large(self.x_ascent_dir, self.gradient)
        hess = hess_builder.build()

        # print(self.gradient)
        # print(hess)

        print("[LLT]: solve")
        solver = ti.linalg.SparseSolver(solver_type="LLT")
        solver.analyze_pattern(hess)
        solver.factorize(hess)
        dx = solver.solve(self.x_ascent_dir.to_numpy().flatten())
        b_success = solver.info()

        np.save(f"debug/iter_{self.cur_epoch}_dx.npy", dx)

        if not b_success:
            print("[LLT]: solver failed. EXIT.")
            exit(-1)

        self.x_ascent_dir.from_numpy(dx.reshape([-1, 3]))
        self.ascent_dir.fill(0.0)
        small_to_large(self.x_ascent_dir, self.ascent_dir)

    def line_search(self, var, loss, compute_ascent_dir):
        """
        Find a proper step size
        This method will invoke forward simulation several times, so don't need to call forward() anymore
        """
        # compute line search threshold
        compute_ascent_dir()
        reduce(self.sum, self.gradient, self.ascent_dir)
        threshold = -self.ls_gamma * self.sum[None]

        # line-search
        step_size = min(1.0, self.step_size / self.ls_alpha)  # use step size from last epoch as initial guess
        while True:
            self.tentative.copy_from(var)
            axpy(-step_size, self.ascent_dir, self.tentative)

            # update trajectory
            cur_loss, loss_info = loss(self.tentative)

            print("step size: %f  loss: %.1f  threshold: %.1f" % (step_size, cur_loss, self.loss + step_size * threshold))

            if cur_loss < self.loss + step_size * threshold or step_size < 1e-6:
                break
            step_size *= self.ls_alpha

        # commit control force
        self.step_size = step_size
        self.loss = cur_loss
        var.copy_from(self.tentative)

        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            self.loss_list.append(self.loss)
            self.loss_per_frame.append(loss_info)
        else:
            self.loss_list.append([self.loss, self.force_loss, self.constrain_loss])
            self.loss_per_frame.append(loss_info)

        return self.loss < 1e-6

    def compute_constrain(self):
        """ Compute constrain value: C = 1/2 M^2 / (N_f * h^2)^2 ||q_{t+1} - q*||^2 + 1/2 ||q_t - q*||^2"""
        self.get_frame(self.trajectory, self.n_frame - 2, self.tmp_vec)
        axpy(-1.0, self.x0, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        con_pos = self.sum[None]

        self.get_frame(self.trajectory, self.n_frame - 1, self.tmp_vec)
        axpy(-1.0, self.x1, self.tmp_vec)
        reduce(self.sum, self.tmp_vec, self.tmp_vec)
        con_pos += self.sum[None]

        return 0.5 * (self.mass / (self.n_frame * self.dt * self.dt))**2 * con_pos

    def projected_newton(self, b_SAP=False):
        """
        Perform one step of projected Newton: p^{k+1} = \grad C(p)^T * \grad C(p) / ||\grad C(p)||^2 * p - c(p) / ||\grad C(p)||^2 * \grad c(p)
        Perform one step SAP: p^{k+1} = p^{k} - c(p) / ||\grad C(p)||^2 * \grad c(p)
        \grad c(p) = \lambda
        """
        # update control forces
        C = self.compute_constrain()
        reduce(self.sum, self.adjoint_vec, self.adjoint_vec)
        grad_C_norm2 = self.sum[None]

        if b_SAP:
            axpy(-C / grad_C_norm2, self.adjoint_vec, self.control_force)
        else:
            reduce(self.sum, self.adjoint_vec, self.control_force)
            self.control_force.copy_from(self.adjoint_vec)
            scale(self.control_force, (self.sum[None] - C) / grad_C_norm2)

        # compute new loss
        self.loss, loss_info = self.compute_soft_constrain_loss(self.control_force)
        self.loss_list.append([self.loss, self.force_loss, self.constrain_loss])
        self.loss_per_frame.append(loss_info)

        return self.constrain_loss < 1e-6

    def one_iter(self):
        if self.method == self.OptimizeMethod["Gauss-Newton"]:
            print("[start position based optimization (Gauss-Newton)]")
            b_converge = self.line_search(self.trajectory, self.compute_position_loss, self.compute_position_ascent_dir)
        else:
            # compute lambda
            print("[compute lambda]", end=" ")

            if self.sim_med == "implicit":
                iter, err = 0, 0
                scalar = (self.mass / (self.n_frame * self.dt * self.dt)) ** 2
                for i in reversed(range(self.n_frame)):
                    # prepare A
                    self.get_frame(self.trajectory, i, self.tmp_vec)
                    self.cloth_sim.compute_hessian(self.tmp_vec, True)

                    # prepare b
                    if i == self.n_frame - 1:  # b_t = scalar * (q_t - q*)
                        self.b.copy_from(self.tmp_vec)
                        axpy(-1.0, self.x1, self.b)
                        scale(self.b, scalar)
                    elif i == self.n_frame - 2:  # b_{t-1} = scalar * (q_{t-1} - q*) + 2 * M/h^2 * lambda_t
                        # self.b.fill(0.0)
                        self.b.copy_from(self.tmp_vec)
                        axpy(-1.0, self.x0, self.b)
                        scale(self.b, scalar)
                        self.get_frame(self.adjoint_vec, i + 1, self.tmp_vec)
                        axpy(2 * self.mass / (self.dt * self.dt), self.tmp_vec, self.b)
                    else:  # b_i = 2 * M/h^2 * lambda_{i+1} - M/h^2 * lambda_{i+2}
                        self.b.fill(0.0)
                        self.get_frame(self.adjoint_vec, i + 1, self.tmp_vec)
                        axpy(2 * self.mass / (self.dt * self.dt), self.tmp_vec, self.b)
                        self.get_frame(self.adjoint_vec, i + 2, self.tmp_vec)
                        axpy(-self.mass / (self.dt * self.dt), self.tmp_vec, self.b)

                    # linear solve
                    A_builder = ti.linalg.sparse_matrix_builder(3 * self.n_vert, 3 * self.n_vert,
                                                                max_num_triplets=self.cloth_sim.sparsity)
                    field_to_sparse(self.cloth_sim.hessian, A_builder)
                    A = A_builder.build()
                    solver = ti.linalg.SparseSolver(solver_type="LLT")
                    solver.analyze_pattern(A)
                    solver.factorize(A)
                    dx = solver.solve(self.b.to_numpy().flatten())
                    self.tmp_vec.from_numpy(dx.reshape([-1, 3]))

                    # self.get_frame(self.adjoint_vec, i, self.tmp_vec)
                    # iter_i, err_i = self.linear_solver.conjugate_gradient(self.tmp_vec)
                    self.set_frame(self.adjoint_vec, i, self.tmp_vec)
                    # iter += iter_i
                    # err += err_i

                # print("CG avg.iter: %d, avg.err: %.1e" % (iter / self.n_frame, err / self.n_frame))

            elif self.sim_med == "symplectic":
                scalar = self.mass / (self.n_frame * self.dt)**2
                for i in reversed(range(self.n_frame)):  # A_ii = M/h^2
                    self.get_frame(self.trajectory, i, self.tmp_vec)
                    if i == self.n_frame - 1:  # b_t = scalar * (q_t - q*)
                        axpy(-1.0, self.x1, self.tmp_vec)
                        scale(self.tmp_vec, scalar)
                    elif i == self.n_frame - 2:  # b_{t-1} = scalar * (q_{t-1} - q*) + (2 * M/h^2 + hess(q_{t-1})) * lambda_t
                        self.cloth_sim.compute_hessian(self.tmp_vec, False)
                        axpy(-1.0, self.x0, self.tmp_vec)
                        scale(self.tmp_vec, scalar)
                        self.get_frame(self.adjoint_vec, i + 1, self.b)
                        axpy(2.0, self.b, self.tmp_vec)
                        self.x_bar.fill(0.0)
                        sparse_mv(self.cloth_sim.hessian, self.b, self.x_bar)
                    else:  # b_i = (2 * M/h^2 + hess(q_i)) * lambda_{t+1} - M/h^2 * lambda_{i+2}
                        self.cloth_sim.compute_hessian(self.tmp_vec, False)
                        self.get_frame(self.adjoint_vec, i + 1, self.b)
                        self.tmp_vec.fill(0.0)
                        sparse_mv(self.cloth_sim.hessian, self.b, self.tmp_vec)
                        axpy(2.0, self.b, self.tmp_vec)
                        self.get_frame(self.adjoint_vec, i + 2, self.b)
                        axpy(-1.0, self.b, self.tmp_vec)
                    self.set_frame(self.adjoint_vec, i, self.tmp_vec)

            # update control forces
            if self.method == self.OptimizeMethod["gradient"]:
                print("[start gradient descent]")
                b_converge = self.line_search(self.control_force, self.compute_soft_constrain_loss, self.compute_soft_constrain_ascent_dir)
            elif self.method == self.OptimizeMethod["projected Newton"]:
                print("[start projected gradient]")
                b_converge = self.projected_newton(False)
            elif self.method == self.OptimizeMethod["SAP"]:
                print("[start SAP]")
                b_converge = self.projected_newton(True)

        self.cur_epoch += 1

        return b_converge
