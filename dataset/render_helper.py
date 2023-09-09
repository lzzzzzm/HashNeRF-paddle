import numpy as np

import paddle


def paddle_gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            x_arange = paddle.arange(x.shape[k], dtype=index.dtype)
            x_arange = x_arange.reshape(reshape_shape)
            dim_index = paddle.expand(x_arange, index_shape).flatten()
            nd_index.append(dim_index)
    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0]).astype("int64")
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / paddle.sum(weights, axis=-1, keepdim=True)
    cdf = paddle.cumsum(pdf, axis=-1)
    cdf = paddle.concat([paddle.zeros_like(cdf[..., :1]), cdf], axis=-1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = paddle.linspace(0., 1., num=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = paddle.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = paddle.to_tensor(u)

    # Invert CDF
    # u = u.contiguous() torch需要这步，但paddle不用，因为torch执行transpose等操作时，并不会创建新的、转置后的tensor，两个tensor内存共享
    inds = paddle.searchsorted(cdf, u, right=True)
    below = paddle.maximum(paddle.zeros_like(inds - 1), inds - 1)
    above = paddle.minimum((cdf.shape[-1] - 1) * paddle.ones_like(inds), inds)
    inds_g = paddle.stack([below, above], axis=-1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    # matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_shape = cdf.unsqueeze(2).expand(matched_shape)
    cdf_g = paddle_gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
    bins_g = paddle_gather(bins.unsqueeze(2).expand(matched_shape), 3, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = paddle.where(denom < 1e-5, paddle.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_rays(H, W, K, c2w):
    """
     use camera to world transform matrix to get 3D world rays from HxW picture
    """
    i, j = paddle.meshgrid(paddle.linspace(0, W - 1, W),
                           paddle.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = paddle.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -paddle.ones_like(i)], axis=-1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = paddle.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                        axis=-1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_from_dir(dir, c2w):
    rays_d = dir @ c2w[:3, :3].T # (H, W, 3)
    rays_d = rays_d / paddle.norm(rays_d, axis=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, -1].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.reshape((-1, 3))
    rays_o = rays_o.reshape((-1, 3))

    return rays_o, rays_d


def output2img(H, W, img_list):
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    rgb = paddle.concat(img_list, axis=0)
    output_img = paddle.reshape(rgb, (H, W, 3))
    output_img = to8b(output_img.cpu().numpy())
    return output_img