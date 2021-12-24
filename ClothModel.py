import os
import json
import numpy as np
import taichi as ti

def load_obj(file_name):
    verts = []
    faces = []
    with open(file_name) as f:
        for line in f:
            if line.startswith('v'):
                verts.append([float(x) for x in line.split()[1:]])
            elif line.startswith('f'):
                faces.append([int(x) - 1 for x in line.split()[1:]])
    return np.array(verts), np.array(faces)

class ClothModel:
    def __init__(self, input_json):
        with open(input_json, 'r') as f:
            input = json.load(f)

        self.verts, self.faces = load_obj(input["init"])
        self.target_verts, _ = load_obj(input['target'])
        self.fixed_idx = np.array(input["fixed"])
        self.init_traj = np.load(input['init_traj'])
        self.init_force = np.load(input['force'])

        self.n_vert = self.verts.shape[0]
        self.n_face = self.faces.shape[0]
        self.n_fixed = self.fixed_idx.shape[0]

        # construct edge list
        edge_list = []
        tmp_vert_list = []
        for f_i in range(self.n_face):
            f_idx = sorted(self.faces[f_i])
            for i in range(3):
                i0, i1, i2 = f_idx[i], f_idx[(i + 1) % 3], f_idx[(i + 2) % 3]
                if i0 > i1: i0, i1 = i1, i0

                if not [i0, i1] in edge_list:
                    edge_list.append([i0, i1])
                    tmp_vert_list.append([i2])
                else:
                    tmp_vert_list[edge_list.index([i0, i1])].append(i2)

        self.n_edge = len(edge_list)
        self.edges = np.array(edge_list)

        inner_edge_list = []
        for edge, tmp_vert in zip(edge_list, tmp_vert_list):
            if len(tmp_vert) > 1:
                inner_edge_list.append([*edge, *tmp_vert])
        self.n_inner_edge = len(inner_edge_list)
        self.inner_edges = np.array(inner_edge_list)

        self.face_field = ti.field(ti.i32, 3 * self.n_face)

    def initialize(self):
        self.face_field.from_numpy(self.faces.flatten())

