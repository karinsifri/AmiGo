import numpy as np
import pytest
import trimesh as tm

from src.branching import _get_vertex_rank, find_saddle_points, get_vertex_sign_changes

MAJOR_RADIUS = 2.0
MINOR_RADIUS = 0.7


@pytest.fixture
def fan():
    """ A single interior vertex (index 0) surrounded by a closed ring link of 12 vertices.

    The link is a full loop, so the sign-change count at the center is exactly controllable by choosing the field
    values on the ring.
    """
    num_ring = 12
    angles = np.linspace(0, 2 * np.pi, num_ring, endpoint=False)
    vertices = np.vstack([[0, 0, 0], np.c_[np.cos(angles), np.sin(angles), np.zeros(num_ring)]])
    faces = np.array([[0, 1 + i, 1 + (i + 1) % num_ring] for i in range(num_ring)])

    return tm.Trimesh(vertices=vertices, faces=faces, process=False), angles


@pytest.fixture
def unit_sphere():
    """ Fine icosphere approximating a unit sphere (radius = 1). """
    return tm.creation.icosphere(subdivisions=3)


@pytest.fixture
def torus():
    """ A torus lying in the xy-plane. """
    return tm.creation.torus(major_radius=MAJOR_RADIUS, minor_radius=MINOR_RADIUS)


def test_get_vertex_rank_orders_by_field():
    """ The rank of each vertex is its position in the field's sorted order. """
    np.testing.assert_array_equal(_get_vertex_rank(np.array([5.0, 1.0, 9.0, 3.0])), [2, 0, 3, 1])


def test_get_vertex_rank_breaks_ties_by_index():
    """ Vertices sharing a field value are ordered by their index, lowest index first. """
    np.testing.assert_array_equal(_get_vertex_rank(np.array([5.0, 1.0, 5.0, 3.0])), [2, 0, 3, 1])


def test_get_vertex_rank_is_a_permutation():
    """ The ranks are a strict total order - every rank in 0..v-1 appears exactly once, even for a constant field. """
    np.testing.assert_array_equal(np.sort(_get_vertex_rank(np.zeros(50))), np.arange(50))


@pytest.mark.parametrize("frequency, expected_count", [
    (1, 2),  # one crossing on each side of the link - a regular point
    (2, 4),  # a simple saddle
    (3, 6),  # a monkey saddle
])
def test_sign_changes_count_crossings_around_the_link(fan, frequency, expected_count):
    """ A field oscillating k times around the link produces exactly 2k sign changes at the center. """
    mesh, angles = fan
    field = np.concatenate([[0.0], np.cos(frequency * angles)])

    assert get_vertex_sign_changes(mesh, field)[0] == expected_count


def test_sign_changes_are_zero_at_an_extremum(fan):
    """ A vertex whose whole link lies above it is a minimum, and has no sign changes. """
    mesh, angles = fan
    field = np.concatenate([[0.0], np.ones_like(angles)])

    assert get_vertex_sign_changes(mesh, field)[0] == 0


def test_sign_changes_are_even_on_a_closed_mesh(unit_sphere):
    """ Every link of a closed mesh is a full loop, so no vertex can have an odd number of sign changes. """
    counts = get_vertex_sign_changes(unit_sphere, unit_sphere.vertices[:, 2])

    assert np.all(counts % 2 == 0)


def test_sign_changes_unaffected_by_exact_ties():
    """ A uv-sphere has rings of identical height, so the height field is full of exact ties.

    The ranking must resolve them into a strict order; comparing the raw values would produce zero-signed differences
    and report saddles all over the sphere.
    """
    mesh = tm.creation.uv_sphere(radius=1.0, count=[24, 24])
    counts = get_vertex_sign_changes(mesh, mesh.vertices[:, 2])

    assert np.sum(counts == 0) == 2, "expected exactly the two poles to be extrema"
    assert np.all(counts <= 2), "a sphere's height field has no saddles"


@pytest.mark.parametrize("frequency, is_saddle", [
    (1, False),  # a regular point
    (2, True),  # a simple saddle
    (3, True),  # a monkey saddle
])
def test_find_saddle_points_classifies_the_centre(fan, frequency, is_saddle):
    """ Only vertices with more than two sign changes are reported as saddles. """
    mesh, angles = fan
    field = np.concatenate([[0.0], np.cos(frequency * angles)])

    assert bool(find_saddle_points(mesh, field)[0]) is is_saddle


def test_find_saddle_points_on_a_sphere(unit_sphere):
    """ The height field of a sphere has only a minimum and a maximum, so no vertex is a saddle. """
    assert not np.any(find_saddle_points(unit_sphere, unit_sphere.vertices[:, 2]))


def test_find_saddle_points_on_a_torus(torus):
    """ Standing a torus up gives the textbook Morse function: one maximum, one minimum and two saddles.

    The saddles sit on the inner rim of the hole, at a distance of (major - minor) radius from the axis.
    """
    saddles = find_saddle_points(torus, torus.vertices[:, 0])

    assert np.sum(saddles) == 2
    np.testing.assert_allclose(np.abs(torus.vertices[saddles][:, 0]), MAJOR_RADIUS - MINOR_RADIUS, atol=1e-6)


@pytest.mark.parametrize("field", [
    np.zeros(5),
    np.zeros(700),
    np.zeros((642, 3)),
    np.zeros((642, 1)),
])
def test_find_saddle_points_rejects_a_mismatched_field(unit_sphere, field):
    """ A field that does not hold exactly one value per vertex is rejected. """
    with pytest.raises(ValueError, match="Expected one scalar value per vertex"):
        find_saddle_points(unit_sphere, field)
