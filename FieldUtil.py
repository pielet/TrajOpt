import taichi as ti

@ti.kernel
def scale(x: ti.template(), a: ti.f32):
    for I in ti.grouped(x):
        x[I] *= a

@ti.kernel
def axpy(a: ti.f32, x: ti.template(), y: ti.template()):
    for I in ti.grouped(y):
        y[I] += a * x[I]

@ti.kernel
def reduce(res: ti.template(), x: ti.template(), y:ti.template()):
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
        print(x[I], end="")
    print('\n')
