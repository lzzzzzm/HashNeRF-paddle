import paddle
import paddle.nn as nn
import numpy as np

def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(
                shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor

def uniform_init(param, a, b):
    """
    Modified tensor inspace using uniform_
    Args:
        param (paddle.Tensor): paddle Tensor
        a (float|int): min value.
        b (float|int): max value.
    Return:
        tensor
    """
    return _no_grad_uniform_(param, a, b)

def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = paddle.zeros_like(coords, dtype='int64')[..., 0]
    coords = paddle.to_tensor(coords, dtype='int64')
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return paddle.to_tensor((1<<log2_hashmap_size)-1, dtype='int64') & xor_result


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    BOX_OFFSETS = paddle.to_tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])
    box_min, box_max = bounding_box

    # use numpy to calc keep_mask
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    box_max_x = paddle.expand(box_max[0], shape=x.shape)
    box_max_y = paddle.expand(box_max[1], shape=y.shape)
    box_max_z = paddle.expand(box_max[2], shape=z.shape)
    stack_x = paddle.stack([x, box_max_x])
    stack_y = paddle.stack([y, box_max_y])
    stack_z = paddle.stack([z, box_max_z])
    x_keep_mask = x == paddle.min(stack_x, axis=0)
    y_keep_mask = y == paddle.min(stack_y, axis=0)
    z_keep_mask = z == paddle.min(stack_z, axis=0)
    keep_mask = paddle.stack([x_keep_mask, y_keep_mask, z_keep_mask], axis=1)
    # keep_mask = xyz == paddle.max(paddle.min(xyz, box_max), box_min)
    if not paddle.all(xyz <= box_max) or not paddle.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = paddle.clip(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = paddle.floor((xyz - box_min) / grid_size)
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = voxel_min_vertex + paddle.to_tensor([1.0, 1.0, 1.0]) * grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

class HashEmbedding(nn.Layer):

    def __init__(self,
                 bounding_box,
                 n_levels=16,
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 finest_resolution=512
                 ):
        super(HashEmbedding, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        self.out_dim = self.n_levels * self.n_features_per_level
        self.b = paddle.exp((paddle.log(paddle.to_tensor(self.finest_resolution, dtype='float32')) - paddle.log(paddle.to_tensor(self.base_resolution, dtype='float32'))) / (self.n_levels - 1))

        self.embeddings = nn.LayerList([nn.Embedding(2 ** self.log2_hashmap_size, \
                                                      self.n_features_per_level) for i in range(self.n_levels)])

        # custom uniform initialization
        for i in range(self.n_levels):
           uniform_init(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds:    B x 8 x 2
        '''
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 4] * weights[:, 0][:, None]
        c01 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 5] * weights[:, 0][:, None]
        c10 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 6] * weights[:, 0][:, None]
        c11 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, None]) + voxel_embedds[:, 7] * weights[:, 0][:, None]

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, x):
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = paddle.floor(self.base_resolution * self.b ** i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = get_voxel_vertices( \
                x, self.bounding_box, \
                resolution, self.log2_hashmap_size)

            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded = self.trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(axis=-1) == keep_mask.shape[-1]
        return paddle.concat(x_embedded_all, axis=-1), keep_mask

