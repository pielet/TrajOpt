import numpy as np

n_frame = 10
n_vert = 4
iter_i = 1

M_h2 = np.load("M_h2.npy")[0]

integration = "symplectic"

def load_tensor(file_name):
    return np.swapaxes(np.load(file_name), 1, 2).reshape([3 * n_vert, 3 * n_vert])

x = []
hess = []
for i in range(n_frame):
    x.append(np.load(f"iter_{iter_i}_x_{i}.npy"))
    hess.append(load_tensor(f"iter_{iter_i}_hess_{i}.npy"))

# full_hess = np.swapaxes(np.load("iter_0_full_hess.npy"), 1, 2).reshape([3 * n_frame * n_vert, -1])

np_full_hess = np.zeros([3 * n_frame * n_vert, 3 * n_frame * n_vert])
local_hess = []
local_sq_hess = []
for i in range(n_frame):
    if integration == "symplectic":
        local_hess.append(np.concatenate(
            [M_h2 * np.eye(3 * n_vert), -2.0 * M_h2 * np.eye(3 * n_vert) + hess[i], M_h2 * np.eye(3 * n_vert)], axis=1))
    elif integration == "implicit":
        local_hess.append(np.concatenate(
            [M_h2 * np.eye(3 * n_vert), -2.0 * M_h2 * np.eye(3 * n_vert), M_h2 * np.eye(3 * n_vert) + hess[i]], axis=1))
    idx = np.roll(np.arange(3 * n_frame * n_vert), -3 * n_vert * (i - 1))[:3 * 3 * n_vert]
    local_sq_hess.append(local_hess[-1].T @ local_hess[-1])
    np_full_hess[np.ix_(idx, idx)] += local_sq_hess[-1]

# res = np_full_hess - full_hess
# norm_res = res / np_full_hess
#
# print("err:", norm_res[~np.isnan(norm_res)].max())
# print("eigen value:\n", np.linalg.eigvals(full_hess[:3 * (n_frame - 2) * n_vert, :3 * (n_frame - 2) * n_vert]))
# print("eigen value:\n", np.linalg.eigvals(np_full_hess[:3 * (n_frame - 2) * n_vert, :3 * (n_frame - 2) * n_vert]))

full_grad = np.load(f"iter_{iter_i}_full_grad.npy").flatten()
dx = np.load(f"iter_{iter_i}_dx.npy")
np_dx = np.linalg.solve(np_full_hess[:3 * (n_frame - 2) * n_vert, :3 * (n_frame - 2) * n_vert], full_grad[:3 * (n_frame - 2) * n_vert])
res = (np_dx - dx) / np_dx

pass