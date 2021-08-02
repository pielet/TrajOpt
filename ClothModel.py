import os
import json
import numpy as np
import taichi as ti
import tina


class ClothModel:
    def __init__(self, input_json):
        with open(input_json, 'r') as f:
            input = json.load(f)

        self.verts, self.faces = tina.readobj(input["init"], simple=True)
        self.target_verts, _ = tina.readobj(input["target"], simple=True)  # assume target and initial model have the same topology
        self.fixed_idx = np.array(input["fixed"])

        self.mesh_model = tina.ConnectiveMesh()
        self.wire_model = tina.MeshToWire(self.mesh_model)

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

    def bind_scene(self, scene):
        scene.add_object(self.mesh_model, tina.Classic())
        scene.add_object(self.wire_model)
        self.mesh_model.set_vertices(self.verts)
        self.mesh_model.set_faces(self.faces)

    def update_scene(self, verts):
        self.mesh_model.set_vertices(verts)

