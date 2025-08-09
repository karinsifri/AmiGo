import numpy as np
import trimesh as tm


def vertex_normals(mesh: tm.Trimesh) -> np.ndarray:
    scaled_f_norm = mesh.face_normals * mesh.area_faces[:, None]
    v_norm = np.zeros_like(mesh.vertices)
    v_norm[mesh.faces[:, 0]] += scaled_f_norm
    v_norm[mesh.faces[:, 1]] += scaled_f_norm
    v_norm[mesh.faces[:, 2]] += scaled_f_norm
    scale = np.linalg.norm(v_norm, axis=1)
    return v_norm[scale >= 1e-10] / scale[scale >= 1e-10, None]


if __name__ == '__main__':
    m = tm.load_mesh("../meshes/pear_dent.off")
    my_vn = vertex_normals(m)
