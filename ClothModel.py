import os
import numpy as np
import taichi as ti
import tina


class ClothModel:
    def __init__(self, input_model, target_model, output_dir):
        self.verts, self.faces = tina.readobj(input_model, simple=True)
        self.target_verts, _ = tina.readobj(target_model, simple=True)  # assume target and initial model have the same topology

        self.mesh = tina.ConnectiveMesh()

        self.n_vert = self.verts.shape[0]
        self.n_face = self.faces.shape[0]

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

        self.output_dir = output_dir

    def bind_scene(self, scene):
        scene.add_object(self.mesh, tina.Classic())
        self.mesh.set_vertices(self.verts)
        self.mesh.set_faces(self.faces)

    def update_scene(self, verts):
        self.mesh.set_vertices(verts)

    def save_ply(self, epoch, frame):
        """
        Save self.verts into .ply file in folder checkpoints/timestamp
        file name format: epoch_[epoch]/frame_[frame]
        """
        epoch_dir = os.path.join(self.output_dir, "epoch_%i" % epoch)

        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        ply_path = os.path.join(epoch_dir, "frame_%i.ply" % frame)

        # fill in data
        writer = ti.PLYWriter(self.n_vert, self.n_face, "tri")
        writer.add_vertex_pos(self.verts[:, 0], self.verts[:, 1], self.verts[:, 2])
        writer.add_faces(self.faces.flatten())

        writer.export_ascii(ply_path)

