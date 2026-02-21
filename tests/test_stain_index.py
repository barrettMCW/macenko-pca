# SPDX-FileCopyrightText: 2024-present barrettMCW <mjbarrett@mcw.edu>
#
# SPDX-License-Identifier: MIT
"""Tests for stain_color_map, find_stain_index, and HistomicsTK cross-validation.

The ``TestStainColorMap`` class validates the well-known reference vectors.
``TestFindStainIndex`` covers the pure-Python alignment logic on synthetic
matrices. ``TestFindStainIndexHistomicsTKParity`` re-implements the
HistomicsTK reference code inline and asserts that our results match to
high precision. ``TestFindStainIndexWithDeconvolution`` exercises
``find_stain_index`` on real stain matrices produced by our Rust-backed
deconvolution pipeline.
"""

import numpy as np
import pytest

from macenko_pca.deconvolution import (
    find_stain_index,
    rgb_separate_stains_macenko_pca,
    stain_color_map,
)

# ---------------------------------------------------------------------------
# HistomicsTK reference implementation (inlined for parity testing)
# ---------------------------------------------------------------------------

# These are the exact vectors from HistomicsTK's stain_color_map.
_HTK_STAIN_COLOR_MAP = {
    "hematoxylin": [0.65, 0.70, 0.29],
    "eosin": [0.07, 0.99, 0.11],
    "dab": [0.27, 0.57, 0.78],
    "null": [0.0, 0.0, 0.0],
}


def _htk_normalize(a):
    """HistomicsTK-style normalize: 1-D → unit vector, 2-D → unit columns."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        norm = np.linalg.norm(a)
        return a / norm if norm != 0.0 else np.zeros_like(a)
    norms = np.linalg.norm(a, axis=0, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return a / norms


def _htk_find_stain_index(reference, w):
    """Reference implementation matching HistomicsTK.find_stain_index."""
    dot_products = np.dot(
        _htk_normalize(np.array(reference)),
        _htk_normalize(np.array(w)),
    )
    return int(np.argmax(np.abs(dot_products)))


# ---------------------------------------------------------------------------
# stain_color_map
# ---------------------------------------------------------------------------


class TestStainColorMap:
    """Tests for the stain_color_map dictionary."""

    def test_has_expected_keys(self):
        """Map should contain the four well-known stain keys."""
        assert set(stain_color_map.keys()) == {
            "hematoxylin",
            "eosin",
            "dab",
            "null",
        }

    def test_each_value_is_length_3(self):
        """Every reference vector must have exactly 3 components."""
        for name, vec in stain_color_map.items():
            assert len(vec) == 3, f"{name} has length {len(vec)}"

    def test_values_match_histomicstk(self):
        """Our vectors must exactly match the HistomicsTK originals."""
        for name, htk_vec in _HTK_STAIN_COLOR_MAP.items():
            np.testing.assert_array_equal(
                stain_color_map[name],
                htk_vec,
                err_msg=f"Mismatch for stain '{name}'",
            )

    def test_null_is_zero(self):
        """The null stain must be the zero vector."""
        assert stain_color_map["null"] == [0.0, 0.0, 0.0]

    def test_non_null_have_positive_norm(self):
        """Non-null stain vectors should have a positive L2 norm."""
        for name, vec in stain_color_map.items():
            if name == "null":
                continue
            assert np.linalg.norm(vec) > 0.0, f"{name} has zero norm"

    def test_hematoxylin_values(self):
        """Verify exact hematoxylin components."""
        assert stain_color_map["hematoxylin"] == [0.65, 0.70, 0.29]

    def test_eosin_values(self):
        """Verify exact eosin components."""
        assert stain_color_map["eosin"] == [0.07, 0.99, 0.11]

    def test_dab_values(self):
        """Verify exact DAB components."""
        assert stain_color_map["dab"] == [0.27, 0.57, 0.78]


# ---------------------------------------------------------------------------
# find_stain_index — basic behaviour
# ---------------------------------------------------------------------------


class TestFindStainIndex:
    """Tests for the find_stain_index function on synthetic data."""

    def test_identity_matrix_hematoxylin(self):
        """Hematoxylin reference has largest component along axis 1 (0.70)."""
        w = np.eye(3)
        idx = find_stain_index(stain_color_map["hematoxylin"], w)
        # hematoxylin = [0.65, 0.70, 0.29] → strongest along col 1
        assert idx == 1

    def test_identity_matrix_eosin(self):
        """Eosin reference has largest component along axis 1 (0.99)."""
        w = np.eye(3)
        idx = find_stain_index(stain_color_map["eosin"], w)
        assert idx == 1

    def test_identity_matrix_dab(self):
        """DAB reference has largest component along axis 2 (0.78)."""
        w = np.eye(3)
        idx = find_stain_index(stain_color_map["dab"], w)
        assert idx == 2

    def test_exact_match_returns_correct_column(self):
        """When a column of w is exactly the reference, that column wins."""
        ref = np.array([0.65, 0.70, 0.29])
        w = np.column_stack(
            [
                [0.07, 0.99, 0.11],  # eosin
                [0.27, 0.57, 0.78],  # dab
                [0.65, 0.70, 0.29],  # hematoxylin
            ]
        )
        assert find_stain_index(ref, w) == 2

    def test_scaled_vector_still_matches(self):
        """Scaling a vector should not change alignment (angle-based)."""
        ref = np.array([0.65, 0.70, 0.29])
        w = np.column_stack(
            [
                np.array([0.07, 0.99, 0.11]) * 100.0,
                np.array([0.65, 0.70, 0.29]) * 0.001,
                np.array([0.27, 0.57, 0.78]) * 50.0,
            ]
        )
        assert find_stain_index(ref, w) == 1

    def test_antiparallel_vector_still_matches(self):
        """Negated vector should still be the best match (abs dot product)."""
        ref = np.array([0.65, 0.70, 0.29])
        w = np.column_stack(
            [
                [0.07, 0.99, 0.11],
                [-0.65, -0.70, -0.29],  # negated hematoxylin
                [0.27, 0.57, 0.78],
            ]
        )
        assert find_stain_index(ref, w) == 1

    def test_returns_int(self):
        """Return type must be a plain Python int."""
        w = np.eye(3)
        result = find_stain_index([1.0, 0.0, 0.0], w)
        assert isinstance(result, int)

    def test_axis_aligned_reference(self):
        """A reference along axis 0 should pick column 0 of the identity."""
        w = np.eye(3)
        assert find_stain_index([1.0, 0.0, 0.0], w) == 0
        assert find_stain_index([0.0, 1.0, 0.0], w) == 1
        assert find_stain_index([0.0, 0.0, 1.0], w) == 2

    def test_two_column_matrix(self):
        """Works with a 3x2 stain matrix (only two stains)."""
        ref = np.array([0.65, 0.70, 0.29])
        w = np.column_stack(
            [
                [0.07, 0.99, 0.11],
                [0.65, 0.70, 0.29],
            ]
        )
        assert find_stain_index(ref, w) == 1

    def test_list_inputs(self):
        """Function should accept plain Python lists, not just arrays."""
        idx = find_stain_index([1.0, 0.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert idx == 0

    def test_noisy_reference_still_finds_closest(self):
        """A slightly perturbed reference should still match the right column."""
        rng = np.random.default_rng(123)
        ref = np.array([0.65, 0.70, 0.29]) + rng.normal(0, 0.02, 3)
        w = np.column_stack(
            [
                [0.27, 0.57, 0.78],
                [0.07, 0.99, 0.11],
                [0.65, 0.70, 0.29],
            ]
        )
        assert find_stain_index(ref, w) == 2

    def test_permuted_he_matrix(self):
        """H&E columns in either order should be identifiable."""
        h = np.array(stain_color_map["hematoxylin"])
        e = np.array(stain_color_map["eosin"])
        complement = np.cross(h, e)
        complement /= np.linalg.norm(complement)

        # Order: [E, complement, H]
        w = np.column_stack([e, complement, h])
        assert find_stain_index(stain_color_map["hematoxylin"], w) == 2
        assert find_stain_index(stain_color_map["eosin"], w) == 0

        # Order: [complement, H, E]
        w2 = np.column_stack([complement, h, e])
        assert find_stain_index(stain_color_map["hematoxylin"], w2) == 1
        assert find_stain_index(stain_color_map["eosin"], w2) == 2


# ---------------------------------------------------------------------------
# Parity with HistomicsTK reference implementation
# ---------------------------------------------------------------------------


class TestFindStainIndexHistomicsTKParity:
    """Cross-validate our find_stain_index against the HistomicsTK logic.

    These tests run both implementations on the same inputs and assert the
    results are identical. The reference implementation is inlined at the
    top of this file.
    """

    @pytest.fixture
    def random_stain_matrices(self):
        """Generate a batch of random 3x3 stain-like matrices."""
        rng = np.random.default_rng(99)
        matrices = []
        for _ in range(50):
            m = rng.random((3, 3))
            # Normalise columns to mimic real stain matrices
            m /= np.linalg.norm(m, axis=0, keepdims=True)
            matrices.append(m)
        return matrices

    def test_identity_parity(self):
        """Both implementations agree on identity matrix for all stains."""
        w = np.eye(3)
        for name, vec in stain_color_map.items():
            if name == "null":
                continue
            ours = find_stain_index(vec, w)
            theirs = _htk_find_stain_index(vec, w)
            assert ours == theirs, f"Mismatch for {name} on identity"

    def test_known_he_matrix_parity(self):
        """Both agree on a known H&E stain matrix."""
        h = np.array([0.65, 0.70, 0.29])
        e = np.array([0.07, 0.99, 0.11])
        complement = np.cross(h, e)
        complement /= np.linalg.norm(complement)
        w = np.column_stack([h, e, complement])

        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name}"

    def test_random_matrices_parity(self, random_stain_matrices):
        """Both implementations agree across many random stain matrices."""
        for i, w in enumerate(random_stain_matrices):
            for name in ("hematoxylin", "eosin", "dab"):
                ref = stain_color_map[name]
                ours = find_stain_index(ref, w)
                theirs = _htk_find_stain_index(ref, w)
                assert ours == theirs, f"Mismatch for {name} on random matrix {i}"

    def test_scaled_columns_parity(self):
        """Both agree when columns are scaled arbitrarily."""
        rng = np.random.default_rng(77)
        w_base = rng.random((3, 3))
        scales = np.array([0.001, 100.0, 5.0])
        w = w_base * scales[np.newaxis, :]

        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name} (scaled)"

    def test_negated_columns_parity(self):
        """Both agree when some columns are negated."""
        h = np.array([0.65, 0.70, 0.29])
        e = np.array([0.07, 0.99, 0.11])
        d = np.array([0.27, 0.57, 0.78])
        w = np.column_stack([-h, e, -d])

        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name} (negated)"

    def test_non_square_parity(self):
        """Both agree on non-square (3xN) matrices."""
        rng = np.random.default_rng(55)
        for ncols in (2, 4, 5, 7):
            w = rng.random((3, ncols))
            for name in ("hematoxylin", "eosin", "dab"):
                ref = stain_color_map[name]
                ours = find_stain_index(ref, w)
                theirs = _htk_find_stain_index(ref, w)
                assert ours == theirs, f"Mismatch for {name} with {ncols} columns"


# ---------------------------------------------------------------------------
# find_stain_index with deconvolution pipeline output
# ---------------------------------------------------------------------------


class TestFindStainIndexWithDeconvolution:
    """Exercise find_stain_index on stain matrices from our Rust pipeline.

    These tests estimate a stain matrix via ``rgb_separate_stains_macenko_pca``
    and then verify that ``find_stain_index`` returns consistent results
    between our implementation and the HistomicsTK reference logic.
    """

    @pytest.fixture
    def he_like_image(self):
        """Synthesise a simple image with two dominant colour directions.

        Mixes H&E-like stain directions with some noise so that PCA-based
        estimation yields a meaningful stain matrix.
        """
        rng = np.random.default_rng(42)
        h = np.array([0.65, 0.70, 0.29])
        e = np.array([0.07, 0.99, 0.11])

        rows, cols = 128, 128
        n = rows * cols
        # Random concentration weights in [0, 1]
        c_h = rng.uniform(0.1, 1.0, n)
        c_e = rng.uniform(0.1, 1.0, n)

        # Optical density = c_h * h + c_e * e + small noise
        od = np.outer(c_h, h) + np.outer(c_e, e)
        od += rng.normal(0, 0.02, od.shape)
        od = np.clip(od, 0, None)

        # OD → RGB: rgb = 256 * exp(-od)
        rgb = 256.0 * np.exp(-od)
        rgb = np.clip(rgb, 1.0, 255.0).reshape(rows, cols, 3)
        return rgb

    def test_stain_index_parity_with_htk_on_estimated_matrix_f64(self, he_like_image):
        """Our find_stain_index matches HistomicsTK on a pipeline-estimated matrix."""
        w = rgb_separate_stains_macenko_pca(he_like_image)

        for name in ("hematoxylin", "eosin"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name} on f64 estimated matrix"

    def test_stain_index_parity_with_htk_on_estimated_matrix_f32(self, he_like_image):
        """Parity holds when the pipeline runs in f32."""
        im_f32 = he_like_image.astype(np.float32)
        w = rgb_separate_stains_macenko_pca(im_f32)

        for name in ("hematoxylin", "eosin"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name} on f32 estimated matrix"

    def test_h_and_e_map_to_different_columns(self, he_like_image):
        """Hematoxylin and eosin must resolve to different column indices."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        h_idx = find_stain_index(stain_color_map["hematoxylin"], w)
        e_idx = find_stain_index(stain_color_map["eosin"], w)
        assert h_idx != e_idx, (
            f"Both mapped to column {h_idx}; expected distinct columns"
        )

    def test_h_and_e_cover_first_two_columns(self, he_like_image):
        """H and E together should map to columns 0 and 1 (the PCA stains)."""
        w = rgb_separate_stains_macenko_pca(he_like_image)
        h_idx = find_stain_index(stain_color_map["hematoxylin"], w)
        e_idx = find_stain_index(stain_color_map["eosin"], w)
        # Column 2 is the cross-product complement — neither H nor E should
        # map there when the image truly contains those stains.
        assert {h_idx, e_idx} == {0, 1}, f"Expected {{0, 1}}, got {{{h_idx}, {e_idx}}}"

    def test_deterministic_index_assignment(self, he_like_image):
        """Running twice on the same image should yield the same indices."""
        w1 = rgb_separate_stains_macenko_pca(he_like_image)
        w2 = rgb_separate_stains_macenko_pca(he_like_image)
        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            assert find_stain_index(ref, w1) == find_stain_index(ref, w2), (
                f"Non-deterministic index for {name}"
            )

    def test_f32_and_f64_agree_on_index(self, he_like_image):
        """f32 and f64 pipelines should produce the same stain index mapping."""
        w_f64 = rgb_separate_stains_macenko_pca(he_like_image)
        w_f32 = rgb_separate_stains_macenko_pca(he_like_image.astype(np.float32))
        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            idx_f64 = find_stain_index(ref, w_f64)
            idx_f32 = find_stain_index(ref, w_f32)
            assert idx_f64 == idx_f32, (
                f"f32/f64 disagree for {name}: {idx_f32} vs {idx_f64}"
            )

    def test_stain_vectors_close_to_reference(self, he_like_image):
        """Estimated stain vectors should be close to the known references.

        This is a soft check: the cosine similarity between the estimated
        stain column and its matched reference should be high (> 0.8).
        """
        w = rgb_separate_stains_macenko_pca(he_like_image)
        for name in ("hematoxylin", "eosin"):
            ref = np.array(stain_color_map[name], dtype=np.float64)
            idx = find_stain_index(ref, w)
            col = w[:, idx].astype(np.float64)
            cosine_sim = abs(
                np.dot(ref, col) / (np.linalg.norm(ref) * np.linalg.norm(col))
            )
            assert cosine_sim > 0.8, (
                f"Cosine similarity for {name} is only {cosine_sim:.4f}"
            )

    def test_random_fixture_images(self, sample_rgb_image):
        """find_stain_index should not error on the shared conftest image."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            idx = find_stain_index(ref, w)
            assert 0 <= idx < w.shape[1]

    def test_random_fixture_images_f32(self, sample_rgb_image_f32):
        """find_stain_index should not error on the f32 conftest image."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image_f32)
        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            idx = find_stain_index(ref, w)
            assert 0 <= idx < w.shape[1]

    def test_htk_parity_on_random_fixture(self, sample_rgb_image):
        """Parity with HistomicsTK logic on the shared conftest image."""
        w = rgb_separate_stains_macenko_pca(sample_rgb_image)
        for name in ("hematoxylin", "eosin", "dab"):
            ref = stain_color_map[name]
            ours = find_stain_index(ref, w)
            theirs = _htk_find_stain_index(ref, w)
            assert ours == theirs, f"Mismatch for {name} on conftest image"
