from math import exp, log, floor
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from dataset.hash_embedding import hash


class VariationLoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self,
                 min_resolution,
                 max_resolution,
                 log2_hashmap_size,
                 n_levels=16):
        super().__init__()
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.n_levels = n_levels

    def forward(self, embeddings, level):
        b = exp((log(self.max_resolution) - log(self.min_resolution)) / (self.n_levels - 1))
        resolution = paddle.to_tensor(floor(self.min_resolution * b ** level))

        min_cube_size = self.min_resolution - 1
        max_cube_size = 50  # can be tuned
        if min_cube_size > max_cube_size:
            assert 'error'

        cube_size = paddle.floor(paddle.clip(resolution / 10.0, min_cube_size, max_cube_size))
        cube_size = paddle.to_tensor(cube_size, dtype='int32')

        # Sample cuboid
        min_vertex = paddle.randint(0, resolution - cube_size, (3,))
        idx = min_vertex + paddle.stack([paddle.arange(cube_size + 1) for _ in range(3)], axis=-1)
        cube_indices = paddle.stack(paddle.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2]), axis=-1)

        hashed_indices = hash(cube_indices, self.log2_hashmap_size)
        cube_embeddings = embeddings(hashed_indices)

        tv_x = paddle.pow(cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2).sum()
        tv_y = paddle.pow(cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2).sum()
        tv_z = paddle.pow(cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2).sum()

        return (tv_x + tv_y + tv_z) / cube_size
