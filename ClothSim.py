import taichi as ti
from FieldUtil import scale, print_field


@ti.data_oriented
class ClothSim:
    def __init__(self,
                 cloth_model, dt, n_iter,
                 use_spring, use_stretch, use_bend,
                 k_spring=0.0, k_stretch=0.0, k_bend=0.0,
                 mass=1.0):
        self.cloth_model = cloth_model
        self.n_vert = cloth_model.n_vert
        self.dt = dt
        self.n_iter = n_iter
        self.mass = mass
        self.inv_mass = 1.0 / mass
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        self.x = ti.Vector.field(3, float, self.n_vert)
        self.v = ti.Vector.field(3, float, self.n_vert)

        # construct constraints
        ## mass-spring
        self.use_spring = use_spring
        if use_spring:
            self.k_spring = k_spring
            self.spring_alpha = 1.0 / (k_spring * dt ** 2)  # compliant
            self.n_spring_con = cloth_model.n_edge
            self.spring_idx = ti.Vector.field(2, int)
            self.spring_l0 = ti.field(float)
            self.spring_lambda = ti.field(float)
            ti.root.dense(ti.i, self.n_spring_con).place(self.spring_idx, self.spring_l0, self.spring_lambda)
        ## stretch
        self.use_stretch = use_stretch
        if use_stretch:
            self.k_stretch = k_stretch
            self.stretch_alpha = 1.0 / (k_stretch * dt ** 2)
            self.n_stretch_con = cloth_model.n_face
            self.stretch_idx = ti.Vector.field(3, int)
            self.stretch_lambda = ti.field(float)
            ti.root.dense(ti.i, self.n_stretch_con).place(self.stretch_idx, self.stretch_lambda)
        ## bend
        self.use_bend = use_bend
        if use_bend:
            self.k_bend = k_bend
            self.bend_alpha = 1.0 / (k_bend * dt ** 2)
            self.n_bend_con = cloth_model.n_inner_edge
            self.bend_idx = ti.Vector.field(4, int)
            self.bend_lambda = ti.field(float)
            ti.root.dense(ti.i, self.n_bend_con).place(self.bend_idx, self.bend_lambda)

        # Hessian matrix
        self.hessian = ti.Matrix.field(3, 3, float)
        ti.root.pointer(ti.ij, self.n_vert).place(self.hessian)

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

    def initialize(self):
        self.x.from_numpy(self.cloth_model.verts)
        self.v.fill(0.0)

        if self.use_spring:
            self.spring_idx.from_numpy(self.cloth_model.edges)
            self.init_spring_con()
        if self.use_stretch:
            self.stretch_idx.from_numpy(self.cloth_model.faces)
            self.init_stretch_con()
        if self.use_bend:
            self.bend_idx.from_numpy(self.cloth_model.inner_edges)
            self.init_bend_con()

    @ti.kernel
    def prologue(self, ext_f: ti.template(), x_next: ti.template()):
        for i in range(self.n_vert):
            x_next[i] = self.x[i] + self.dt * self.v[i] + self.dt ** 2 * (self.inv_mass * ext_f[i] + self.gravity)

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

            n = (x0 - x1).normalized()
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

    def XPBD_step(self, ext_f, x_next):
        """
        Run one step of forward simulation under the [ext_f: ti.Vector.field]
        return next position [x: ti.Vector.field]
        """
        # set initial pos
        self.prologue(ext_f, x_next)
        # print("[XPBD start]:")
        # print_field(self.x)

        # reset lambda
        if self.use_spring: self.spring_lambda.fill(0.0)
        if self.use_stretch: self.stretch_lambda.fill(0.0)
        if self.use_bend: self.bend_lambda.fill(0.0)

        for i in range(self.n_iter):
            if self.use_spring: self.solve_spring_con(x_next)
            if self.use_stretch: self.solve_stretch_con(x_next)
            if self.use_bend: self.solve_bend_con(x_next)

        # commit pos and vel
        self.epilogue(x_next)
        # print("[XPBD end]: ")
        # print_field(self.x)

    @ti.kernel
    def compute_spring_hessian(self, x: ti.template()):
        for i in range(self.n_spring_con):
            i0, i1 = self.spring_idx[i]
            x0, x1 = x[i0], x[i1]  # column vector

            l0 = self.spring_l0[i]
            l = (x0 - x1).norm()

            local_H = self.k_spring * ((1.0 - l0 / l) * ti.Matrix.identity(float, 3)
                                       + l0 / l ** 3 * (x0 - x1) @ (x0 - x1).transpose())

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
    def reset_hessian(self):
        for i, j in self.hessian:
            self.hessian[i, j] = ti.Matrix.zero(float, 3, 3)

    @ti.kernel
    def add_in_diagonal(self, a: ti.f32):
        for i in range(self.n_vert):
            self.hessian[i, i] += a * ti.Matrix.identity(float, 3)

    def compute_hessian(self, x):
        """
        Compute Hessian matrix of [x: ti.Vector.field]: M + h^2 \grad^2 E(x)
        """
        self.reset_hessian()

        if self.use_spring: self.compute_spring_hessian(x)
        if self.use_stretch: self.compute_stretch_hessian(x)
        if self.use_bend: self.compute_bend_hessian(x)

        scale(self.hessian, self.dt ** 2)
        self.add_in_diagonal(self.mass)
