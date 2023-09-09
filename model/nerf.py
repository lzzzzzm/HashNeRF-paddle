import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .utils import batchify, raw2outputs

def nerf_render(
                coarse_nerf,
                fine_nerf,
                dataset,
                inputs,
                pts,
                viewdirs,
                z_vals,
                rays_o,
                rays_d,
                chunk=1024*32,
                fine_sample=64,
                return_coarse=False,
                return_sparsity_loss=False):

    # TODO only support batchsize=1
    # inputs = inputs.squeeze(0)
    # pts = pts.squeeze(0)
    # z_vals = z_vals.squeeze(0)
    # rays_o = rays_o.squeeze(0)
    # rays_d = rays_d.squeeze(0)
    # viewdirs = viewdirs.squeeze(0)

    # coarse infer
    coarse_outputs_flat = batchify(coarse_nerf, chunk=chunk)(inputs)
    # TODO support batchsize=1
    coarse_outputs = paddle.reshape(coarse_outputs_flat, list(pts.shape[:-1]) + [coarse_outputs_flat.shape[-1]]).squeeze(0)
    coarse_rgb_map, disp_map, acc_map, weights, depth_map, coarse_sparsity_loss = raw2outputs(coarse_outputs, z_vals, rays_d)

    # fine infer
    fine_inputs, pts, z_vals, rays_d = dataset.get_fine_input(weights, viewdirs, z_vals, rays_o, rays_d, N_samples=fine_sample, batch_input=True)
    fine_outputs_flat = batchify(fine_nerf, chunk=chunk)(fine_inputs)
    fine_outputs = paddle.reshape(fine_outputs_flat, list(pts.shape[:-1]) + [fine_outputs_flat.shape[-1]]).squeeze(0)
    fine_rgb_map, disp_map, acc_map, weights, depth_map, fine_sparsity_loss = raw2outputs(fine_outputs, z_vals, rays_d)

    if return_coarse:
        if return_sparsity_loss:
            return coarse_rgb_map, fine_rgb_map, coarse_sparsity_loss, fine_sparsity_loss
        else:
            return coarse_rgb_map, fine_rgb_map
    else:
        if return_sparsity_loss:
            return fine_rgb_map, fine_sparsity_loss
        else:
            return fine_rgb_map



class NeRF(nn.Layer):
    def __init__(self,
                D=8,
                W=256,
                input_ch=3,
                input_ch_views=3,
                output_ch=4,
                skips=[4],
                chunk=1024 * 16,
                sample_number=64,
                use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.chunk = chunk
        self.use_viewdirs = use_viewdirs
        self.sample_number = sample_number

        self.pts_linears = nn.LayerList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range( D -1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.LayerList([nn.Linear(input_ch_views + W, W// 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = paddle.split(x, num_or_sections=[self.input_ch, self.input_ch_views], axis=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = paddle.concat([input_pts, h], axis=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = paddle.concat([feature, input_views], axis=-1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = paddle.concat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

