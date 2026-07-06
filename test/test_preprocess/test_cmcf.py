import numpy as np
import pytest
import trimesh as tm

from src.consts import EPSILON
from src.preprocess.cmcf import smooth_craters
from src.shapeop import shape_operator_ftf


@pytest.fixture
def convex_sphere():
    """Unit icosphere — all faces convex (H > 0, K > 0), no crater regions."""
    return tm.creation.icosphere(subdivisions=3)


@pytest.fixture
def inverted_sphere():
    """Unit icosphere with reversed face winding — all faces are craters (H < 0, K > 0)."""
    sphere = tm.creation.icosphere(subdivisions=1)
    return tm.Trimesh(sphere.vertices.copy(), sphere.faces[:, ::-1])


@pytest.fixture
def sphere_with_dimple():
    """Icosphere with a Gaussian indent at the north pole, creating a localised crater region."""
    sphere = tm.creation.icosphere(subdivisions=2)
    verts = sphere.vertices.copy().astype(float)
    normals = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    dists_to_north = np.linalg.norm(verts - np.array([0.0, 0.0, 1.0]), axis=1)
    indent = -0.5 * np.exp(-dists_to_north ** 2 / 0.2)
    verts += indent[:, None] * normals
    return tm.Trimesh(verts, sphere.faces)


def test_smooth_craters_output_is_trimesh(convex_sphere):
    """smooth_craters must return a Trimesh regardless of whether any smoothing was applied."""
    assert isinstance(smooth_craters(convex_sphere), tm.Trimesh)


def test_smooth_craters_topology_unchanged(convex_sphere):
    """Vertex count, face count, and face connectivity must be identical after smoothing."""
    result = smooth_craters(convex_sphere)
    assert result.vertices.shape == convex_sphere.vertices.shape
    assert np.array_equal(result.faces, convex_sphere.faces)


def test_smooth_craters_vertices_finite(convex_sphere):
    """The implicit linear solve must not produce NaN or Inf vertex coordinates."""
    assert np.all(np.isfinite(smooth_craters(convex_sphere).vertices))


def test_convex_sphere_has_no_craters(convex_sphere):
    """Fixture guard: convex sphere must have no faces satisfying K >= 0 and H <= 0."""
    _, _, kminf, kmaxf = shape_operator_ftf(convex_sphere)
    H = 0.5 * (kminf + kmaxf)
    K = kminf * kmaxf
    assert not np.any((K >= -EPSILON) & (H <= EPSILON))


def test_smooth_craters_no_craters_vertices_unchanged(convex_sphere):
    """When no crater faces are detected the loop breaks immediately — vertices must not move."""
    result = smooth_craters(convex_sphere)
    assert np.array_equal(result.vertices, convex_sphere.vertices)


def test_inverted_sphere_is_entirely_craters(inverted_sphere):
    """Fixture guard: every face of the inverted sphere must have H < 0 and K > 0.
    An inverted sphere is a fully concave mesh — useful to verify crater detection,
    but not passed to smooth_craters (MCF collapses it to a point in 250 iterations)."""
    _, _, kminf, kmaxf = shape_operator_ftf(inverted_sphere)
    H = 0.5 * (kminf + kmaxf)
    K = kminf * kmaxf
    assert np.all(H < -EPSILON)
    assert np.all(K > EPSILON)


def test_smooth_craters_dimple_vertices_change(sphere_with_dimple):
    """Smoothing a mesh that contains craters must move at least some vertices."""
    result = smooth_craters(sphere_with_dimple)
    assert not np.allclose(result.vertices, sphere_with_dimple.vertices)


def test_smooth_craters_dimple_topology_unchanged(sphere_with_dimple):
    """The solver only moves vertex positions — it must never change face connectivity."""
    result = smooth_craters(sphere_with_dimple)
    assert np.array_equal(result.faces, sphere_with_dimple.faces)


def test_smooth_craters_dimple_vertices_finite(sphere_with_dimple):
    """Smoothing a localised crater must not cause numerical blow-up in any vertex coordinate."""
    assert np.all(np.isfinite(smooth_craters(sphere_with_dimple).vertices))


def test_smooth_craters_crater_faces_decrease_after_smoothing(sphere_with_dimple):
    """After smoothing the dimple, the number of faces satisfying K >= 0, H <= 0
    must be strictly smaller — the algorithm must have made progress."""
    _, _, kminf_before, kmaxf_before = shape_operator_ftf(sphere_with_dimple)
    H_before = 0.5 * (kminf_before + kmaxf_before)
    K_before = kminf_before * kmaxf_before
    craters_before = int(np.sum((K_before >= -EPSILON) & (H_before <= EPSILON)))

    result = smooth_craters(sphere_with_dimple)

    _, _, kminf_after, kmaxf_after = shape_operator_ftf(result)
    H_after = 0.5 * (kminf_after + kmaxf_after)
    K_after = kminf_after * kmaxf_after
    craters_after = int(np.sum((K_after >= -EPSILON) & (H_after <= EPSILON)))

    assert craters_after < craters_before


def test_dimple_has_crater_faces(sphere_with_dimple):
    """Fixture guard: the indented region must contain at least one crater face."""
    _, _, kminf, kmaxf = shape_operator_ftf(sphere_with_dimple)
    H = 0.5 * (kminf + kmaxf)
    K = kminf * kmaxf
    assert np.any((K >= -EPSILON) & (H <= EPSILON))


def test_smooth_craters_crater_region_moves_more_than_non_crater_region(sphere_with_dimple):
    """Vertices near the north-pole crater must displace more than south-pole vertices,
    because the Gaussian weight drops to nearly zero at geodesic distance >> sigma."""
    verts_before = sphere_with_dimple.vertices.copy()
    result = smooth_craters(sphere_with_dimple)
    displacement = np.linalg.norm(result.vertices - verts_before, axis=1)

    near_crater = verts_before[:, 2] > 0.5
    far_from_crater = verts_before[:, 2] < -0.5

    assert near_crater.any() and far_from_crater.any()
    assert displacement[near_crater].mean() > displacement[far_from_crater].mean()
