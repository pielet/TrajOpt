import taichi as ti
from FieldUtil import *


@ti.data_oriented
class ClothSim:
    SimulationMethod = {
        "XPBD": 0,
        "Newton": 1
    }

    def __init__(self,
                 cloth_model, dt, method, n_iter, err, linear_solver,
                 use_spring, use_stretch, use_bend, use_attach,
                 k_spring=0.0, k_stretch=0.0, k_bend=0.0, k_attach=0.0,
                 density=1.5):  # g/cm^2
        self.cloth_model = cloth_model
        self.n_vert = cloth_model.n_vert
        self.n_face = cloth_model.n_face
        self.method = self.SimulationMethod[method]
        self.dt = dt
        self.n_iter = n_iter
        self.err = err
        self.density = density
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        self.x = ti.Vector.field(3, ti.f32, self.n_vert)
        self.v = ti.Vector.field(3, ti.f32, self.n_vert)

        self.face_idx = ti.Vector.field(3, ti.i32, self.n_face)  # for mass computation
        self.total_area = ti.field(ti.f32, ())

        # construct constraints
        ## mass-spring
        self.use_spring = use_spring
        if use_spring:
            self.k_spring = k_spring
            self.spring_alpha = 1.0 / (k_spring * dt ** 2)  # compliant
            self.n_spring_con = cloth_model.n_edge
            self.spring_idx = ti.Vector.field(2, ti.i32)
            self.spring_l0 = ti.field(ti.f32)
            self.spring_lambda = ti.field(ti.f32)
            ti.root.dense(ti.i, self.n_spring_con).place(self.spring_idx, self.spring_l0, self.spring_lambda)
        ## stretch
        self.use_stretch = use_stretch
        if use_stretch:
            self.k_stretch = k_stretch
            self.stretch_alpha = 1.0 / (k_stretch * dt ** 2)
            self.n_stretch_con = cloth_model.n_face
            self.stretch_idx = ti.Vector.field(3, ti.i32)
            self.stretch_lambda = ti.field(ti.f32)
            ti.root.dense(ti.i, self.n_stretch_con).place(self.stretch_idx, self.stretch_lambda)
        ## bend
        self.use_bend = use_bend
        if use_bend:
            self.k_bend = k_bend
            self.bend_alpha = 1.0 / (k_bend * dt ** 2)
            self.n_bend_con = cloth_model.n_inner_edge
            self.bend_idx = ti.Vector.field(4, ti.i32)
            self.bend_lambda = ti.field(ti.f32)
            ti.root.dense(ti.i, self.n_bend_con).place(self.bend_idx, self.bend_lambda)
        ## attach
        self.use_attach = use_attach
        if use_attach:
            self.k_attach = k_attach
            self.attach_alpha = 1.0 / (k_attach * dt ** 2)
            self.n_attach_con = cloth_model.n_fixed
            self.attach_idx = ti.field(ti.i32)
            self.attach_target = ti.Vector.field(3, ti.f32)
            self.attach_lambda = ti.field(ti.f32)
            ti.root.dense(ti.i, self.n_attach_con).place(self.attach_idx, self.attach_target, self.attach_lambda)

        # Newton's method
        self.y = ti.Vector.field(3, ti.f32, self.n_vert)
        ## linear solver
        self.linear_solver = linear_solver
        self.hessian = linear_solver.A
        self.gradient = linear_solver.b
        self.desc_dir = ti.Vector.field(3, ti.f32, self.n_vert)
        ## line-search
        self.tentetive_x = ti.Vector.field(3, ti.f32, self.n_vert)
        self.energy = ti.field(ti.f32, ())


    @ti.kernel
    def compute_total_area(self):
        self.total_area[None] = 0.0
        for i in range(self.n_face):
            i0, i1, i2 = self.face_idx[i][0], self.face_idx[i][1], self.face_idx[i][2]
            x0, x1, x2 = self.x[i0], self.x[i1], self.x[i2]
            self.total_area[None] += 0.5 * (x1 - x0).cross(x2 - x0).norm()

    @ti.kernel
    def init_spring_con(self):
        for i in range(self.n_spring_con):
            x0 = self.x[self.spring_idx[i][0]]
            x1 = self.x[self.spring_idx[i][1]]
            self.spring_l0[i] = (x1 - x0).norm()

    @ti.kernel
    def init_stretch_con(self):
        pass

    @ti.kernel
    def init_bend_con(self):
        pass

    @ti.kernel
    def init_attach_con(self):
        for i in range(self.n_attach_con):
            self.attach_target[i] = self.x[self.attach_idx[i]]

    def initialize(self):
        self.x.from_numpy(self.cloth_model.verts)
        self.v.fill(0.0)
        self.desc_dir.fill(0.0)

        # compute mass
        self.face_idx.from_numpy(self.cloth_model.faces)
        self.compute_total_area()
        self.mass = self.density * self.total_area[None] / self.n_vert
        self.inv_mass = 1.0 / self.mass

        # init constrain
        if self.use_spring:
            self.spring_idx.from_numpy(self.cloth_model.edges)
            self.init_spring_con()
        if self.use_stretch:
            self.stretch_idx.from_numpy(self.cloth_model.faces)
            self.init_stretch_con()
        if self.use_bend:
            self.bend_idx.from_numpy(self.cloth_model.inner_edges)
            self.init_bend_con()
        if self.use_attach:
            self.attach_idx.from_numpy(self.cloth_model.fixed_idx)
            self.init_attach_con()

    @ti.kernel
    def prologue(self, ext_f: ti.template(), x_next: ti.template()):
        for i in range(self.n_vert):
            self.y[i] = self.x[i] + self.dt * self.v[i] + self.dt ** 2 * (self.inv_mass * ext_f[i] + self.gravity)
            x_next[i] = self.y[i]

    @ti.kernel
    def epilogue(self, x_next: ti.template()):
        for i in range(self.n_vert):
            self.v[i] = (x_next[i] - self.x[i]) / self.dt
            self.x[i] = x_next[i]

    @ti.kernel
    def solve_spring_con(self, x: ti.template()):
        for i in range(self.n_spring_con):
            i0, i1 = self.spring_idx[i]
            x0, x1 = x[i0], x[i1]

            n = safe_normalized(x0 - x1)
            C = (x0 - x1).norm() - self.spring_l0[i]

            dL = -(C + self.spring_alpha * self.spring_lambda[i]) / (2 * self.inv_mass + self.spring_alpha)
            self.spring_lambda[i] += dL
            x[i0] += self.inv_mass * dL * n
            x[i1] += -self.inv_mass * dL * n

    @ti.kernel
    def solve_stretch_con(self, x: ti.template()):
        pass

    @ti.kernel
    def solve_bend_con(self, x: ti.template()):
        pass

    @ti.kernel
    def solve_attach_con(self, x: ti.template()):
        for i in range(self.n_attach_con):
            idx = self.attach_idx[i]
            xi = x[idx]
            xt = self.attach_target[i]

            C = (xi - xt).norm()
            n = safe_normalized(xi - xt)

            dL = -(C + self.attach_alpha * self.attach_lambda[i]) / (2 * self.inv_mass + self.attach_alpha)
            self.spring_lambda[i] += dL
            x[idx] += self.inv_mass * dL * n

    def XPBD(self, ext_f, x_next):
        """
        Run one step of forward simulation under the [ext_f: ti.Vector.field]
        return next position [x: ti.Vector.field]
        """
        # set initial pos
        self.prologue(ext_f, x_next)

        # reset lambda
        if self.use_spring: self.spring_lambda.fill(0.0)
        if self.use_stretch: self.stretch_lambda.fill(0.0)
        if self.use_bend: self.bend_lambda.fill(0.0)
        if self.use_attach: self.attach_lambda.fill(0.0)

        n_frame = 0
        for i in range(self.n_iter):
            n_frame += 1
            if self.use_spring: self.solve_spring_con(x_next)
            if self.use_stretch: self.solve_stretch_con(x_next)
            if self.use_bend: self.solve_bend_con(x_next)
            if self.use_attach: self.solve_attach_con(x_next)

            self.compute_gradient(x_next)
            reduce(self.energy, self.gradient, self.gradient)
            self.energy[None] /= self.n_vert
            if self.energy[None] < self.err:
                break

        # commit pos and vel
        self.epilogue(x_next)

        return n_frame, self.energy[None]

    @ti.kernel
    def compute_spring_energy(self, x: ti.template()):
        for i in range(self.n_spring_con):
            i0, i1 = self.spring_idx[i]
            x0, x1 = x[i0], x[i1]
            self.energy[None] += 0.5 * self.k_spring * ((x0 - x1).norm() - self.spring_l0[i]) ** 2

    @ti.kernel
    def compute_stretch_energy(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_bend_energy(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_attach_energy(self, x: ti.template()):
        for i in range(self.n_attach_con):
            xi = x[self.attach_idx[i]]
            xt = self.attach_target[i]
            self.energy[None] += 0.5 * self.k_attach * (xi - xt).norm_sqr()

    def compute_energy(self, x):
        """ Compute objective of [x: ti.Vector.field]: 1/2 * ||x - y||_M^2 + h^2 * E(x) """
        if self.use_spring: self.compute_spring_energy(x)
        if self.use_stretch: self.compute_stretch_energy(x)
        if self.use_bend: self.compute_bend_energy(x)
        if self.use_attach: self.compute_attach_energy(x)
        inner_energy = self.dt ** 2 * self.energy[None]

        self.tentetive_x.copy_from(x)  # FIXME: assume self.tentative_x is unused
        axpy(-1.0, self.y, self.tentetive_x)
        reduce(self.energy, self.tentetive_x, self.tentetive_x)
        inertia_energy = 0.5 * self.mass * self.energy[None]

        return inner_energy + inertia_energy

    @ti.kernel
    def compute_spring_gradient(self, x: ti.template()):
        for i in range(self.n_spring_con):
            i0, i1 = self.spring_idx[i]
            x0, x1 = x[i0], x[i1]

            local_g = self.k_spring * ((x0 - x1).norm() - self.spring_l0[i]) * safe_normalized(x0 - x1)

            self.gradient[i0] += local_g
            self.gradient[i1] += -local_g

    @ti.kernel
    def compute_stretch_gradient(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_bend_gradient(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_attach_gradient(self, x: ti.template()):
        for i in range(self.n_attach_con):
            idx = self.attach_idx[i]
            xi = x[idx]
            xt = self.attach_target[i]
            self.gradient[idx] += self.k_attach * (xi - xt)

    def compute_gradient(self, x):
        """ Compute gradient of [x: ti.Vector.field]: M * (x - y) + h^2 \grad E(x) """
        self.gradient.fill(0.0)

        if self.use_spring: self.compute_spring_gradient(x)
        if self.use_stretch: self.compute_stretch_gradient(x)
        if self.use_bend: self.compute_bend_gradient(x)
        if self.use_attach: self.compute_attach_gradient(x)

        scale(self.gradient, self.dt ** 2)
        axpy(self.mass, x, self.gradient)
        axpy(-self.mass, self.y, self.gradient)

    @ti.kernel
    def compute_spring_hessian(self, x: ti.template()):
        for i in range(self.n_spring_con):
            i0, i1 = self.spring_idx[i]
            x0, x1 = x[i0], x[i1]  # column vector

            l0 = self.spring_l0[i]
            l = (x0 - x1).norm()

            local_H = self.k_spring * (l0 / l ** 3 * (x0 - x1) @ (x0 - x1).transpose() +
                                       (1.0 - l0 / l) * ti.Matrix.identity(ti.f32, 3))

            # local_H = self.k_spring * l0 / l ** 3 * (x0 - x1) @ (x0 - x1).transpose()
            # if l > l0:
            #     local_H += self.k_spring * (1.0 - l0 / l) * ti.Matrix.identity(ti.f32, 3)

            self.hessian[i0, i0] += local_H
            self.hessian[i0, i1] += -local_H
            self.hessian[i1, i0] += -local_H
            self.hessian[i1, i1] += local_H

    @ti.kernel
    def compute_stretch_hessian(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_bend_hessian(self, x: ti.template()):
        pass

    @ti.kernel
    def compute_attach_hessian(self, x: ti.template()):
        for i in range(self.n_attach_con):
            idx = self.attach_idx[i]
            self.hessian[idx, idx] += self.k_attach * ti.Matrix.identity(ti.f32, 3)

    @ti.kernel
    def add_in_diagonal(self, a: ti.f32):
        for i in range(self.n_vert):
            self.hessian[i, i] += a * ti.Matrix.identity(ti.f32, 3)

    def compute_hessian(self, x):
        """ Compute Hessian matrix of [x: ti.Vector.field]: M + h^2 \grad^2 E(x) """
        self.linear_solver.reset()

        if self.use_spring: self.compute_spring_hessian(x)
        if self.use_stretch: self.compute_stretch_hessian(x)
        if self.use_bend: self.compute_bend_hessian(x)
        if self.use_attach: self.compute_attach_hessian(x)

        scale(self.hessian, self.dt ** 2)
        self.add_in_diagonal(self.mass)

    def line_search(self, x, desc_dir, gradient):
        obj = self.compute_energy(x)
        reduce(self.energy, desc_dir, gradient)
        desc_dir_dot_gradient = self.energy[None]

        step_size = 1.0
        while True:
            self.tentetive_x.copy_from(x)
            axpy(step_size, desc_dir, self.tentetive_x)
            new_obj = self.compute_energy(self.tentetive_x)
            if new_obj < obj + 0.03 * step_size * desc_dir_dot_gradient or step_size < 1e-5:
                break
            step_size *= 0.5

        return step_size

    def Newton(self, ext_f, x_next):
        self.prologue(ext_f, x_next)

        n_frame = 0
        for i in range(self.n_iter):
            n_frame += 1
            self.compute_gradient(x_next)
            self.compute_hessian(x_next)

            self.linear_solver.conjugate_gradient(self.desc_dir)
            scale(self.desc_dir, -1.0)

            step_size = self.line_search(x_next, self.desc_dir, self.gradient)
            reduce(self.energy, self.gradient, self.gradient)
            self.energy[None] /= self.n_vert
            if step_size < 1e-5 or self.energy[None] < self.err:
                break
            axpy(step_size, self.desc_dir, x_next)

        self.epilogue(x_next)

        return n_frame, self.energy[None]

    def step(self, ext_f, x_next):
        if self.method == self.SimulationMethod["XPBD"]:
            return self.XPBD(ext_f, x_next)
        elif self.method == self.SimulationMethod["Newton"]:
            return self.Newton(ext_f, x_next)
