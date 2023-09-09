import os
import numpy as np

import paddle
from paddle.io import DataLoader


from dataset import BlenderDataset, sample_pdf
from model import NeRF, HashNeRF, nerf_render
from utils import logger, config_parse
from losses import MSELoss, VariationLoss
from metrics import MSEPSNR

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def train(args):
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

    # define model
    if args.model_type == 'nerf':
        coarse_nerf = NeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=True)
        fine_nerf = NeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views, use_viewdirs=True)
    if args.model_type == 'hashnerf':
        coarse_nerf = HashNeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views)
        fine_nerf = HashNeRF(input_ch=args.input_ch, input_ch_views=args.input_ch_views)
    else:
        assert 'Not support {} model type'.format(args.model_type)

    # load dataset
    train_dataset = BlenderDataset(
        data_root=args.dataset_root,
        mode='train',
        model_type=args.model_type,
        half_res=args.half_res,
        log_name=args.log_out,
        pts_input_ch=args.pts_input_ch, view_input_ch=args.view_input_ch,
        pts_embedding_freqs=args.pts_embedding_freqs, view_embeding_freqs=args.view_embeding_freqs,
        render_phi=args.render_phi, render_radius=args.render_radius,
        train_select_number=1024)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # define train params
    loss_fn = MSELoss()
    if args.model_type == 'hashnerf':
        loss_var = VariationLoss(
            min_resolution=16,
            max_resolution=512,
            log2_hashmap_size=19,
            n_levels=16
        )
    metric = MSEPSNR()
    grad_vars = list(coarse_nerf.parameters())
    grad_vars += list(fine_nerf.parameters())
    if args.model_type == 'hashnerf':
        grad_vars += list(train_dataset.pts_positional_embedding.parameters())

    optimizer = paddle.optimizer.Adam(parameters=grad_vars, learning_rate=args.lrate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-07)

    H, W = train_dataset.H, train_dataset.W

    save_index = 0

    chunk = args.chunk
    all_coarse_rgb = []
    all_fine_rgb = []
    cur_iters = 0
    while cur_iters < args.train_iters:

        for index, (image, poses, inputs, pts, viewdirs, z_vals, rays_o, rays_d) in enumerate(train_dataloader):

            cur_iters = cur_iters + 1

            coarse_rgb_map, fine_rgb_map, coarse_sparsity_loss, fine_sparsity_loss = nerf_render(
                coarse_nerf=coarse_nerf,
                fine_nerf=fine_nerf,
                dataset=train_dataset,
                inputs=inputs,
                pts=pts,
                viewdirs=viewdirs,
                z_vals=z_vals,
                rays_o=rays_o,
                rays_d=rays_d,
                fine_sample=128,
                return_coarse=True,
                return_sparsity_loss=True
            )

            optimizer.clear_grad()
            # calc loss
            coarse_loss = loss_fn(coarse_rgb_map, image[:, :, :3])
            fine_loss = loss_fn(fine_rgb_map, image[:, :, :3])
            total_loss = coarse_loss + fine_loss
            total_loss = total_loss + args.sparse_loss_weight*(paddle.sum(coarse_sparsity_loss) + paddle.sum(fine_sparsity_loss))
            if args.model_type == 'hashnerf':
                embeddings = train_dataset.pts_positional_embedding.embeddings
                TV_loss = 0
                for i in range(16):
                    TV_loss += loss_var(embeddings[i], level=i)

            total_loss = total_loss + args.tv_loss_weight*TV_loss
            if cur_iters > 1000:
                args.tv_loss_weight = 0.0

            # step
            total_loss.backward()
            optimizer.step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (cur_iters / decay_steps))
            optimizer.set_lr(new_lrate)

            if cur_iters % 1000 == 0:
                psnr = metric(total_loss)
                logger.info('Iter: {} Train Loss:{}, PSNR:{}'.format(cur_iters, float(total_loss), float(psnr)))

            if cur_iters % 10000 == 0:
                save_path = os.path.join('save/{:06d}.pdparams'.format(cur_iters))
                paddle.save(
                    {
                        'train_iter': cur_iters,
                        'coarse_nerf_state_dict': coarse_nerf.state_dict(),
                        'fine_nerf_state_dict': fine_nerf.state_dict(),
                    },
                    save_path
                )


if __name__ == '__main__':
    args = config_parse()
    train(args)
