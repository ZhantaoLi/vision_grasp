import pytest

from vision_grasp import camera_node


def test_build_sim_blocks_returns_named_block_tuples():
    blocks = camera_node.build_sim_blocks(
        block_names=["red_block", "green_block"],
        block_xs=[0.30, 0.00],
        block_ys=[0.15, -0.18],
        block_sizes=[0.035, 0.035],
        block_color_bs=[0, 0],
        block_color_gs=[0, 180],
        block_color_rs=[200, 0],
    )

    assert blocks == [
        ("red_block", 0.30, 0.15, 0.035, (0, 0, 200)),
        ("green_block", 0.00, -0.18, 0.035, (0, 180, 0)),
    ]


def test_build_sim_blocks_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        camera_node.build_sim_blocks(
            block_names=["red_block"],
            block_xs=[0.30, 0.00],
            block_ys=[0.15],
            block_sizes=[0.035],
            block_color_bs=[0],
            block_color_gs=[0],
            block_color_rs=[200],
        )
