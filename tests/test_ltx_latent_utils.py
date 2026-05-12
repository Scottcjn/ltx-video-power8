import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from ltx_latent_utils import map_key, pack_latents, unpack_latents


def _product(values):
    total = 1
    for value in values:
        total *= value
    return total


def _flat_index(coord, shape):
    index = 0
    for axis, size in zip(coord, shape):
        index = index * size + axis
    return index


def _iter_coords(shape):
    if not shape:
        yield ()
        return
    for axis in range(shape[0]):
        for rest in _iter_coords(shape[1:]):
            yield (axis, *rest)


class FakeTensor:
    """Tiny tensor double with the reshape/permute API used by the helpers."""

    def __init__(self, values, shape):
        self.values = list(values)
        self.shape = tuple(shape)
        if len(self.values) != _product(self.shape):
            raise ValueError("value count does not match shape")

    def reshape(self, *shape):
        if _product(shape) != len(self.values):
            raise ValueError("cannot reshape values to requested shape")
        return FakeTensor(self.values, shape)

    def permute(self, *dims):
        if sorted(dims) != list(range(len(self.shape))):
            raise ValueError("dims must be a permutation of tensor axes")

        new_shape = tuple(self.shape[dim] for dim in dims)
        new_values = []
        for new_coord in _iter_coords(new_shape):
            old_coord = [0] * len(self.shape)
            for new_axis, old_axis in enumerate(dims):
                old_coord[old_axis] = new_coord[new_axis]
            new_values.append(self.values[_flat_index(old_coord, self.shape)])
        return FakeTensor(new_values, new_shape)


def test_pack_and_unpack_round_trip_with_spatial_and_temporal_patches():
    latents = FakeTensor(range(64), (1, 2, 4, 4, 2))

    packed = pack_latents(latents, patch_size=2, patch_size_t=2)
    restored = unpack_latents(
        packed,
        num_frames=4,
        height=4,
        width=2,
        patch_size=2,
        patch_size_t=2,
        out_channels=2,
    )

    assert packed.shape == (1, 4, 16)
    assert restored.shape == latents.shape
    assert restored.values == latents.values


def test_pack_and_unpack_preserve_independent_batches():
    latents = FakeTensor(range(32), (2, 2, 2, 2, 2))

    packed = pack_latents(latents)
    restored = unpack_latents(packed, num_frames=2, height=2, width=2, out_channels=2)

    assert packed.shape == (2, 8, 2)
    assert restored.values[:16] == latents.values[:16]
    assert restored.values[16:] == latents.values[16:]


@pytest.mark.parametrize(
    ("checkpoint_key", "expected"),
    [
        ("model.diffusion_model.patchify_proj.weight", "proj_in.weight"),
        ("model.diffusion_model.block.0.q_norm.weight", "block.0.norm_q.weight"),
        ("model.diffusion_model.block.0.k_norm.weight", "block.0.norm_k.weight"),
        (
            "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight",
            "time_embed.emb.timestep_embedder.linear_1.weight",
        ),
        (
            "model.diffusion_model.adaln_single.linear.weight",
            "time_embed.linear.weight",
        ),
    ],
)
def test_map_key_normalizes_checkpoint_names(checkpoint_key, expected):
    assert map_key(checkpoint_key) == expected
