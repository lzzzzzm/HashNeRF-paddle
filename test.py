import os
import imageio
import numpy as np

import paddle

from dataset import BlenderDataset, sample_pdf, output2img
from model import NeRF, HashNeRF, nerf_render
from utils import load_pretrained_model_from_path, load_pretrained_model_from_state_dict, logger, config_parse

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def test(args):
    # make save dir
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

    # load model
    state_dict = paddle.load(args.load_from)

    coarse_state_dict = state_dict['coarse_nerf_state_dict']
    fine_state_dict = state_dict['fine_nerf_state_dict']
    if args.model_type == 'nerf':
        coarse_nerf = NeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=args.use_viewdirs)
        fine_nerf = NeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=args.use_viewdirs)
    else:
        coarse_nerf = HashNeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views)
        fine_nerf = HashNeRF(input_ch=args.input_ch, input_ch_views=args.input_ch)

    load_pretrained_model_from_state_dict(coarse_nerf, coarse_state_dict)
    load_pretrained_model_from_state_dict(fine_nerf, fine_state_dict)

    # load dataset
    dataset = BlenderDataset(
        data_root=args.dataset_root,
        mode='test',
        model_type=args.model_type,
        half_res=args.half_res,
        log_name=args.log_out,
        pts_input_ch=args.pts_input_ch, view_input_ch=args.view_input_ch,
        pts_embedding_freqs=args.pts_embedding_freqs, view_embeding_freqs=args.view_embeding_freqs,
        render_phi=args.render_phi, render_radius=args.render_radius)
    H, W = dataset.H, dataset.W

    chunk = args.chunk
    all_coarse_rgb = []
    all_fine_rgb = []
    with paddle.no_grad():
        for index, (image, poses, inputs, pts, viewdirs, z_vals, rays_o, rays_d) in logger.enumerate(dataset,
                                                                                                     msg='nerf-test'):
            coarse_rgb_maps = []
            fine_rgb_maps = []
            j_index = 0
            # use chunk to infer

            for i in range(0, pts.shape[0], chunk):
                j = j_index * chunk * dataset.sample_number
                j_index = j_index + 1
                inputs_chunk = chunk * dataset.sample_number
                # batchsize=1
                chunk_inputs = inputs[j:j + inputs_chunk].unsqueeze(0)
                chunk_viewdirs = viewdirs[i: i + chunk].unsqueeze(0)
                chunk_z_vals = z_vals[i:i + chunk].unsqueeze(0)
                chunk_rays_o = rays_o[i:i + chunk].unsqueeze(0)
                chunk_rays_d = rays_d[i:i + chunk].unsqueeze(0)
                chunk_pts = pts[i:i + chunk].unsqueeze(0)
                coarse_rgb_map, fine_rgb_map = nerf_render(
                    coarse_nerf=coarse_nerf,
                    fine_nerf=fine_nerf,
                    dataset=dataset,
                    inputs=chunk_inputs,
                    pts=chunk_pts,
                    viewdirs=chunk_viewdirs,
                    z_vals=chunk_z_vals,
                    rays_o=chunk_rays_o,
                    rays_d=chunk_rays_d,
                    fine_sample=64,
                    return_coarse=True,
                )

                coarse_rgb_maps.append(coarse_rgb_map)
                fine_rgb_maps.append(fine_rgb_map)

            # save img
            coarse_img = output2img(W, H, coarse_rgb_maps)
            fine_img = output2img(W, H, fine_rgb_maps)

            all_coarse_rgb.append(coarse_img)
            all_fine_rgb.append(fine_img)

    # save video
    logger.info('Infer Done, save Video in {} dir'.format(args.save_dir))
    all_coarse_rgb = np.stack(all_coarse_rgb, 0)
    all_fine_rgb = np.stack(all_fine_rgb, 0)

    imageio.mimwrite(os.path.join(args.save_dir, 'coarse_video.mp4'), all_coarse_rgb, fps=15, quality=8)
    imageio.mimwrite(os.path.join(args.save_dir, 'fine_video.mp4'), all_fine_rgb, fps=15, quality=8)


if __name__ == '__main__':
    args = config_parse()
    test(args)
