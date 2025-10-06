import numpy as np
import pytest
import trimesh as tm

from preprocess.scale import set_surface_area_to_one


@pytest.mark.parametrize("input_mesh, expected_mesh", [
    (tm.primitives.Sphere(0.1).to_mesh(), tm.primitives.Sphere(np.sqrt(1/(4.0 * np.pi))).to_mesh()),
    (tm.primitives.Sphere(1).to_mesh(), tm.primitives.Sphere(np.sqrt(1/(4.0 * np.pi))).to_mesh()),
    (tm.primitives.Sphere(5).to_mesh(), tm.primitives.Sphere(np.sqrt(1/(4.0 * np.pi))).to_mesh()),
    (tm.primitives.Sphere(20).to_mesh(), tm.primitives.Sphere(np.sqrt(1/(4.0 * np.pi))).to_mesh()),
    (tm.primitives.Box(np.ones(3) * 0.1).to_mesh(), tm.primitives.Box(np.sqrt(np.ones(3)/6)).to_mesh()),
    (tm.primitives.Box(np.ones(3) * 1).to_mesh(), tm.primitives.Box(np.sqrt(np.ones(3)/6)).to_mesh()),
    (tm.primitives.Box(np.ones(3) * 5).to_mesh(), tm.primitives.Box(np.sqrt(np.ones(3)/6)).to_mesh()),
    (tm.primitives.Box(np.ones(3) * 20).to_mesh(), tm.primitives.Box(np.sqrt(np.ones(3)/6)).to_mesh()),
])
def test_set_surface_area_to_one(input_mesh: tm.Trimesh, expected_mesh: tm.Trimesh):
    """ Test the set_surface_area_to_one function for simple, primitive meshes """
    result_mesh = set_surface_area_to_one(input_mesh)

    # Check if surface area is 1
    assert np.isclose(result_mesh.area, 1)

    # check if the mesh is the same as the one with the surface area of 1
    assert (np.allclose(result_mesh.vertices, expected_mesh.vertices, atol=1e-3) and
            np.array_equal(result_mesh.faces, expected_mesh.faces))
