import numpy as np
import pytest
import trimesh as tm

from src.shapeop import (
    CurvatureResult,
    _edge_vectors,
    edge_basis,
    shape_operator_ftf,
    shapeop,
    vertex_normals,
)


@pytest.fixture
def unit_sphere():
    """ Fine icosphere approximating a unit sphere (radius = 1). """
    return tm.creation.icosphere(subdivisions=3)


@pytest.fixture
def unit_cylinder():
    """ Cylinder of radius 1 and height 4 with 64 circumferential sections. """
    return tm.creation.cylinder(radius=1.0, height=4.0, sections=64)


def test_vertex_normals_unit_length(unit_sphere):
    """ Every returned normal must be a unit vector. """
    norms = np.linalg.norm(vertex_normals(unit_sphere), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_vertex_normals_sphere_direction(unit_sphere):
    """ On a unit sphere centred at the origin, area-weighted normals should
    point radially outward, i.e. match the normalised vertex positions. """
    expected = unit_sphere.vertices / np.linalg.norm(unit_sphere.vertices, axis=1, keepdims=True)
    assert np.allclose(vertex_normals(unit_sphere), expected, atol=0.02)


def test_edge_vectors_sum_to_zero(unit_sphere):
    """ The three edge vectors of every triangle sum to zero (they form a
    closed loop). """
    e0, e1, e2 = _edge_vectors(unit_sphere)
    assert np.allclose(e0 + e1 + e2, 0.0, atol=1e-12)


def test_edge_vectors_opposite_vertex(unit_sphere):
    """ e_i must be the edge opposite vertex i, i.e. it connects the other
    two vertices and is therefore orthogonal to no particular vertex — but
    it must equal the difference of those two vertices. """
    v = unit_sphere.vertices
    f = unit_sphere.faces
    e0, e1, e2 = _edge_vectors(unit_sphere)
    assert np.allclose(e0, v[f[:, 2]] - v[f[:, 1]], atol=1e-12)
    assert np.allclose(e1, v[f[:, 0]] - v[f[:, 2]], atol=1e-12)
    assert np.allclose(e2, v[f[:, 1]] - v[f[:, 0]], atol=1e-12)


def test_shape_operator_returns_named_tuple(unit_sphere):
    """ Result must be a CurvatureResult NamedTuple with the expected fields. """
    result = shape_operator_ftf(unit_sphere)
    assert isinstance(result, CurvatureResult)
    assert hasattr(result, 'dminf')
    assert hasattr(result, 'dmaxf')
    assert hasattr(result, 'kminf')
    assert hasattr(result, 'kmaxf')


def test_shape_operator_output_shapes(unit_sphere):
    """ Output arrays must have the right shapes. """
    nf = len(unit_sphere.faces)
    result = shape_operator_ftf(unit_sphere)
    assert result.dminf.shape == (nf, 3)
    assert result.dmaxf.shape == (nf, 3)
    assert result.kminf.shape == (nf,)
    assert result.kmaxf.shape == (nf,)


def test_shape_operator_curvature_ordering(unit_sphere):
    """ kminf <= kmaxf must hold for every face. """
    result = shape_operator_ftf(unit_sphere)
    assert np.all(result.kminf <= result.kmaxf)


def test_shape_operator_directions_unit_length(unit_sphere):
    """ Principal direction vectors must be unit length. """
    result = shape_operator_ftf(unit_sphere)
    assert np.allclose(np.linalg.norm(result.dminf, axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(result.dmaxf, axis=1), 1.0, atol=1e-10)


def test_shape_operator_directions_tangent(unit_sphere):
    """ Principal directions must lie in the tangent plane of each face,
    i.e. be orthogonal to the face normal. """
    result = shape_operator_ftf(unit_sphere)
    n = unit_sphere.face_normals
    assert np.allclose(np.einsum('fi,fi->f', result.dminf, n), 0.0, atol=1e-10)
    assert np.allclose(np.einsum('fi,fi->f', result.dmaxf, n), 0.0, atol=1e-10)


def test_shape_operator_directions_orthogonal(unit_sphere):
    """ The two principal directions must be orthogonal to each other. """
    result = shape_operator_ftf(unit_sphere)
    dot = np.einsum('fi,fi->f', result.dminf, result.dmaxf)
    assert np.allclose(dot, 0.0, atol=1e-10)


def test_shape_operator_sphere_curvatures(unit_sphere):
    """ On a unit sphere both principal curvatures equal 1 everywhere.
    The FTF estimator is approximate, so we check the mean with loose tolerance. """
    result = shape_operator_ftf(unit_sphere)
    assert np.allclose(result.kminf.mean(), 1.0, atol=0.05)
    assert np.allclose(result.kmaxf.mean(), 1.0, atol=0.05)


def test_shape_operator_cylinder_curvatures(unit_cylinder):
    """ On a unit cylinder the axial curvature is 0 and the circumferential
    curvature is 1/r = 1. Side faces (normals not aligned with z) are used
    to exclude the flat caps. """
    result = shape_operator_ftf(unit_cylinder)
    n = unit_cylinder.face_normals
    side_faces = np.abs(n[:, 2]) < 0.1  # exclude caps whose normal ~ [0,0,±1]
    # FTF systematically over-estimates kmin on a cylinder because cap vertices
    # are shared with side vertices, introducing axial curvature contamination.
    assert np.allclose(result.kminf[side_faces].mean(), 0.0, atol=0.1)
    assert np.allclose(result.kmaxf[side_faces].mean(), 1.0, atol=0.05)


def test_edge_basis_shape(unit_sphere):
    """ Output must be a (2·nf, 3·nf) sparse matrix. """
    nf = len(unit_sphere.faces)
    eb = edge_basis(unit_sphere)
    assert eb.shape == (2 * nf, 3 * nf)


def test_edge_basis_e0_projects_to_first_axis(unit_sphere):
    """ e0 is aligned with the first basis vector, so projecting it into the
    2-D tangent basis must yield (|e0|, 0) for every face. """
    e0, _, _ = _edge_vectors(unit_sphere)
    eb = edge_basis(unit_sphere, e0)
    nf = len(unit_sphere.faces)
    coords = (eb @ e0.flatten(order='F')).reshape(nf, 2, order='F')  # (nf, 2)
    assert np.allclose(coords[:, 0], np.linalg.norm(e0, axis=1), atol=1e-10)
    assert np.allclose(coords[:, 1], 0.0, atol=1e-10)


def test_shapeop_shape(unit_sphere):
    """ Output must be a (2·nf, 2·nf) sparse matrix. """
    nf = len(unit_sphere.faces)
    SO = shapeop(unit_sphere)
    assert SO.shape == (2 * nf, 2 * nf)


def test_shapeop_symmetric(unit_sphere):
    """ V^T D V is symmetric for any diagonal D and any V. """
    SO = shapeop(unit_sphere)
    diff = SO - SO.T
    assert np.allclose(diff.data, 0.0, atol=1e-10)
