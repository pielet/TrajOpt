import taichi as ti
from FieldUtil import *


@ti.data_oriented
class LinearSolver:
    CG_PRECOND_METHED = {
        "None": 0,
        "Jacobi": 1
    }

    def __init__(self, n, cg_precond="Jacobi", cg_iter=100, cg_err=1e-6):
        self.n = n

        self.cg_precond = self.CG_PRECOND_METHED[cg_precond]
        self.cg_iter = cg_iter
        self.cg_err = cg_err

        # A (sparse)
        self.A = ti.Matrix.field(3, 3, ti.f32)
        self.A_pointer = ti.root.pointer(ti.ij, self.n)
        self.A_pointer.place(self.A)
        self.b_empty = True
        self.compute_Ap = lambda x, Ax: sparse_mv(self.A, x, Ax)

        # CG iteration variables
        self.alpha = ti.field(ti.f32, shape=())
        self.beta = ti.field(ti.f32, shape=())
        self.rTz = ti.field(ti.f32, shape=())
        self.sum = ti.field(ti.f32, shape=())
        self.res = ti.field(ti.f32, shape=())

        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.inv_A_diag = ti.Vector.field(3, ti.f32, self.n)
        self.b = ti.Vector.field(3, ti.f32, self.n)
        self.r = ti.Vector.field(3, ti.f32, self.n)
        self.z = ti.Vector.field(3, ti.f32, self.n)
        self.p = ti.Vector.field(3, ti.f32, self.n)
        self.Ap = ti.Vector.field(3, ti.f32, self.n)

    def reset(self):
        if self.b_empty:
            self.b_empty = False
        else:
            self.__reset()

    @ti.kernel
    def __reset(self):
        for i, j in self.A:
            self.A[i, j] = ti.Matrix.zero(ti.f32, 3, 3)

    @ti.kernel
    def update_preconditioner(self):
        for i in range(self.n):
            for j in ti.static(range(3)):
                self.inv_A_diag[i][j] = 1.0 / self.A[i, i][j, j]

    def conjugate_gradient(self, x, b_verbose=False):
        """
        Solve Ax = b
        A and b must be compute before
        [x: ti.Vector.field] is both input (initial guess) and output
        """

        # r = b - Ax (x's initial value is lambda from last epoch)
        self.r.copy_from(self.b)
        self.Ap.fill(0.0)
        self.compute_Ap(x, self.Ap)
        axpy(-1.0, self.Ap, self.r)

        reduce(self.sum, self.b, self.b)
        if b_verbose: print(f"  CG init: b_norm {self.sum[None]}")
        threshold = self.sum[None] * self.cg_err  # |b| scaled threshold

        # z and p
        if self.cg_precond == self.CG_PRECOND_METHED["Jacobi"]:
            self.update_preconditioner()
            # print_field(self.inv_A_diag)
            element_wist_mul(self.inv_A_diag, self.r, self.z)
        else:
            self.z.copy_from(self.r)
        self.p.copy_from(self.z)

        # rTz
        reduce(self.rTz, self.r, self.z)
        # print("CG iter -1: %.1ef" % self.rTz[None])

        reduce(self.res, self.r, self.r)
        if self.res[None] < threshold:
            return 0, self.res[None]

        if b_verbose:
            print(f"  CG init: res {self.res[None]}")

        n_iter = 0
        for i in range(self.cg_iter):
            n_iter += 1
            self.Ap.fill(0.0)
            self.compute_Ap(self.p, self.Ap)

            # alpha
            reduce(self.sum, self.p, self.Ap)
            self.alpha[None] = self.rTz[None] / self.sum[None]

            # update x and r(z)
            axpy(self.alpha[None], self.p, x)
            axpy(-self.alpha[None], self.Ap, self.r)

            reduce(self.res, self.r, self.r)
            if b_verbose: print(f"  CG iter {i}: res {self.res[None]}")
            if self.res[None] < threshold:
                if b_verbose: print(f"  CG converge at {i}: res {self.res[None]}")
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
            if b_verbose: print(f"  CG doesn't converge: res {self.res[None]}")

        return n_iter, self.res[None]
