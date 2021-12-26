import taichi as ti
import numpy as np
import os

@ti.kernel
def scale(x: ti.template(), a: ti.f32):
    for I in ti.grouped(x):
        x[I] *= a

@ti.kernel
def axpy(a: ti.f32, x: ti.template(), y: ti.template()):
    """
    y += a * x
    """
    for I in ti.grouped(y):
        y[I] += a * x[I]

@ti.kernel
def reduce(res: ti.template(), x: ti.template(), y: ti.template()):
    res[None] = 0.0
    for I in ti.grouped(x):
        res[None] += x[I].dot(y[I])

@ti.kernel
def element_wist_mul(x: ti.template(), y: ti.template(), z: ti.template()):
    for I in ti.grouped(z):
        z[I] = x[I] * y[I]

@ti.kernel
def print_field(x: ti.template()):
    for I in ti.grouped(x):
        print(I, x[I], end="\n")

@ti.func
def safe_normalized(vec):
    return vec / max(vec.norm(), 1e-12)

@ti.kernel
def sparse_set_zero(A: ti.template()):
    for i, j in A:
        A[i, j] = ti.Matrix.zero(ti.f32, 3, 3)

@ti.kernel
def sparse_copy(A: ti.template(), B: ti.template()):
    """ copy A to B based on A's sparsity"""
    for i, j in A:
        B[i, j] = A[i, j]

@ti.kernel
def sparse_mv(A: ti.template(), x: ti.template(), Ax: ti.template()):
    for i, j in A:
        Ax[i] += A[i, j] @ x[j]

@ti.kernel
def sparse_ATA(n: ti.i32, A: ti.template(), A_pointer: ti.template(), ATA: ti.template()):
    for i, j in ti.ndrange(n, n):
        sum = ti.Matrix.zero(ti.f32, 3, 3)
        b_active = False
        for k in range(n):
            if ti.is_active(A_pointer, [k, i]) and ti.is_active(A_pointer, [k, j]):
                sum += A[k, i] @ A[k, j]
                b_active = True
        if b_active:
            ATA[i, j] = sum

@ti.kernel
def large_to_small_sparse(A: ti.template(), B: ti.template(), n: ti.i32):
    for i, j in A:
        if i < n and j < n:
            B[i, j] = A[i, j]

@ti.kernel
def small_to_large(A: ti.template(), B: ti.template()):
    for I in ti.grouped(A):
        B[I] = A[I]

@ti.kernel
def large_to_small(A: ti.template(), B: ti.template()):
    for I in ti.grouped(B):
        B[I] = A[I]

@ti.kernel
def sparsity(A: ti.template(), A_pointer: ti.template()) -> ti.i32:
    nonzero = 0
    for i, j in A:
        if ti.is_active(A_pointer, [i, j]):
            nonzero += 9
    return nonzero

@ti.kernel
def field_to_sparse(field: ti.template(), sparse: ti.linalg.sparse_matrix_builder()):
    for i, j in field:
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            sparse[3 * i + ii, 3 * j + jj] += field[i, j][ii, jj]

def save(file_name, A):
    np.save(os.path.join("debug", file_name), A.to_numpy())
