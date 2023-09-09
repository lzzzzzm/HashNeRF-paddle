import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.distribution import Categorical


def raw2outputs(raw,
                z_vals,
                rays_d,
                raw_noise_std=0,
                batch_input=False,
                white_bkgd=True,
                pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - paddle.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = paddle.concat([dists, paddle.to_tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * paddle.linalg.norm(rays_d[..., None, :], axis=-1)

    rgb = F.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = paddle.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = paddle.to_tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = []
    for batch in range(alpha.shape[0]):
        weight = alpha[batch] * paddle.cumprod(paddle.concat([paddle.ones((alpha[batch].shape[0], 1)), 1. - alpha[batch] + 1e-10], axis=-1),
                                     dim=-1)[:, :-1]
        weights.append(weight)
    weights = paddle.stack(weights)
    # weights = alpha * paddle.cumprod(paddle.concat([paddle.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], axis=-1),
    #                                  dim=-1)[:, :-1]
    rgb_map = paddle.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = paddle.sum(weights * z_vals, -1)
    disp_map = 1. / paddle.maximum(1e-10 * paddle.ones_like(depth_map), depth_map / paddle.sum(weights, -1))
    acc_map = paddle.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    entropy = Categorical(logits=paddle.concat([weights, 1.0 - paddle.sum(weights, axis=-1, keepdim=True) + 1e-6], axis=-1)).entropy()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss

def batchify_rays(model, rays_flat, chunk=1024*32):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = model(rays_flat[i:i+chunk])
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : paddle.concat(all_ret[k], axis=0) for k in all_ret}
    return all_ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return paddle.concat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], axis=0)

    return ret