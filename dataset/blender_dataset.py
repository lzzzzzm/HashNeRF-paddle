import os
import paddle
import numpy as np
import imageio
import json
import cv2
import logging

from paddle.io import Dataset

from .positional_embedding import PositionalEmbedding
from .hash_embedding import HashEmbedding
from .sh_embedding import SHEmbedding
from .render_helper import get_rays, sample_pdf
from utils import logger
from .dataset_utils import *


class BlenderDataset(Dataset):
    def __init__(self,
                 data_root,
                 mode,
                 testskip=1,
                 use_viewdirs=True,
                 half_res=True,
                 use_embedding=True,
                 pts_input_ch=3,
                 view_input_ch=None,
                 pts_embedding_freqs=10,
                 view_embeding_freqs=None,
                 sample_number=64,
                 train_select_number=1024,
                 render_phi=-40,
                 render_radius=4.0,
                 log_name='exp.log',
                 model_type='nerf'):
        self.data_root = data_root
        self.mode = mode
        self.testskip = testskip
        self.half_res = half_res
        self.use_viewdirs = use_viewdirs
        self.use_embedding = use_embedding
        self.train_select_number = train_select_number
        self.model_type= model_type

        self.sample_number = sample_number
        self.near = 2.
        self.far = 6.

        with open(os.path.join(data_root, 'transforms_{}.json'.format(self.mode)), 'r') as fp:
            self.metas = json.load(fp)

        if mode == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        imgs = []
        poses = []
        for frame in self.metas['frames'][::skip]:
            fname = os.path.join(data_root, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        # 默认读入图片尺寸一致
        self.H, self.W = imgs[0].shape[:2]
        camera_angle_x = float(self.metas['camera_angle_x'])
        self.focal = .5 * self.W / np.tan(.5 * camera_angle_x)  # 获取焦距

        if half_res:
            self.H = self.H // 2
            self.W = self.W // 2
            self.focal = self.focal / 2.
            imgs_half_res = np.zeros((imgs.shape[0], self.H, self.W, 4))
            for i, img in enumerate(imgs):
                imgs_half_res[i] = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            imgs = imgs_half_res
        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])

        if model_type == 'nerf':
            self.pts_positional_embedding = PositionalEmbedding(
                input_dims=pts_input_ch,
                include_input=True,
                max_freq_log2=pts_embedding_freqs - 1,
                num_freqs=pts_embedding_freqs,
                log_sampling=True
            )
            self.dir_positional_embedding = PositionalEmbedding(
                input_dims=view_input_ch,
                include_input=True,
                max_freq_log2=view_embeding_freqs - 1,
                num_freqs=view_embeding_freqs,
                log_sampling=True
            )
        if model_type == 'hashnerf':
            bounding_box = get_bbox3d_for_blenderobj(self.metas, self.H, self.W, near = self.near, far = self.far)
            self.pts_positional_embedding = HashEmbedding(
                bounding_box=bounding_box,
                log2_hashmap_size=19,
                finest_resolution=512,
            )
            self.dir_positional_embedding = SHEmbedding()

        self.imgs = imgs
        self.poses = poses
        self.render_poses = paddle.stack(
            [pose_spherical(angle, render_phi, render_radius) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], axis=0)
        # logger.info('Blender {} dataset loading'.format(mode))
        # logger.info('{} dataset has number of {} samples'.format(mode, len(imgs)))

    def embedding_inputs(self, pts_inputs, viewdirs_inputs=None, batch_input=False):
        if not batch_input:
            pts_inputs = pts_inputs.unsqueeze(0)
            viewdirs_inputs = viewdirs_inputs.unsqueeze(0)

        outputs = []
        for batch in range(pts_inputs.shape[0]):
            pts = pts_inputs[batch]

            pts_flat = paddle.reshape(pts, [-1, pts.shape[-1]])

            if self.model_type == 'nerf':
                embedded_pts = self.pts_positional_embedding(pts_flat)
            else:
                embedded_pts, keep_mask = self.pts_positional_embedding(pts_flat)
            if self.use_viewdirs:
                input_dirs = viewdirs_inputs[batch][:, None].expand(pts.shape)
                input_dirs_flat = paddle.reshape(input_dirs, [-1, input_dirs.shape[-1]])
                embedded_viewdirs = self.dir_positional_embedding(input_dirs_flat)
                output = paddle.concat([embedded_pts, embedded_viewdirs], -1)
            else:
                output = embedded_pts
            outputs.append(output)
        outputs = paddle.stack(outputs).squeeze(0)
        return outputs

    def get_fine_input(self, weights, viewdirs, z_vals, rays_o, rays_d, N_samples, batch_input=False):
        """
            use coarse weights to get fine sample points
        """
        inputsz_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(inputsz_vals_mid, weights[..., 1:-1], N_samples=N_samples, det=True)
        z_samples = z_samples.detach()
        z_vals = paddle.sort(paddle.concat([z_vals, z_samples], -1), axis=-1)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]
        if self.use_embedding:
            fine_inputs = self.embedding_inputs(pts, viewdirs, batch_input=batch_input)
        else:
            # TODO
            assert 'embedding'

        return fine_inputs, pts, z_vals, rays_d

    def raysd2viewdirs(self, rays_d):
        viewdirs = rays_d
        viewdirs = viewdirs / paddle.linalg.norm(viewdirs, axis=-1, keepdim=True)
        viewdirs = paddle.reshape(viewdirs, [-1, 3]).astype('float32')
        return viewdirs

    def rays2pts(self, rays_o, rays_d, viewdirs=None):
        rays_o = paddle.reshape(rays_o, [-1, 3]).astype('float32')
        rays_d = paddle.reshape(rays_d, [-1, 3]).astype('float32')

        near, far = self.near * paddle.ones_like(rays_d[..., :1]), self.far * paddle.ones_like(rays_d[..., :1])
        # rays to pts
        rays = paddle.concat([rays_o, rays_d, near, far], axis=-1)
        if self.use_viewdirs:
            rays = paddle.concat([rays, viewdirs], axis=-1)
        t_vals = paddle.linspace(0., 1., num=self.sample_number)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        N_rays = rays.shape[0]
        z_vals = z_vals.expand([N_rays, self.sample_number])
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return pts, z_vals, rays_o, rays_d

    def train_select_coords(self, rays_o, rays_d, image):
        """
            when training, select samples to train
        """
        coords = paddle.stack(
            paddle.meshgrid(paddle.linspace(0, self.H - 1, self.H), paddle.linspace(0, self.W - 1, self.W)),
            -1)  # (H, W, 2)
        coords = paddle.reshape(coords, [-1, 2])
        select_inds = np.random.choice(coords.shape[0], size=[self.train_select_number], replace=False)  # (N_rand,)
        select_coords = coords[select_inds].astype('int64')  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = paddle.stack([rays_o, rays_d], axis=0)
        image = image[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        viewdirs = self.raysd2viewdirs(rays_d)

        pts, z_vals, rays_o, rays_d = self.rays2pts(rays_o, rays_d, viewdirs)

        return image, pts, viewdirs, z_vals, rays_o, rays_d

    def __getitem__(self, idx):
        image = paddle.to_tensor(self.imgs[idx])
        if self.mode == 'test':
            poses = self.render_poses[idx]
            rays_o, rays_d = get_rays(self.H, self.W, self.K, poses)
        else:
            poses = paddle.to_tensor(self.poses[idx])
            rays_o, rays_d = get_rays(self.H, self.W, self.K, poses)

        if self.use_viewdirs:
            viewdirs = self.raysd2viewdirs(rays_d)
        else:
            viewdirs = None

        if self.mode == 'train':
            image, pts, viewdirs, z_vals, rays_o, rays_d = self.train_select_coords(rays_o, rays_d, image)
        else:
            pts, z_vals, rays_o, rays_d = self.rays2pts(rays_o, rays_d, viewdirs)

        if self.use_embedding:
            output = self.embedding_inputs(pts, viewdirs)
        else:
            # TODO
            assert 'embedding'

        return image, poses, output, pts, viewdirs, z_vals, rays_o, rays_d

    def __len__(self):
        return len(self.poses)

    @property
    def is_train_mode(self) -> bool:
        return 'train' in self.mode


if __name__ == '__main__':
    blender_dataset = BlenderDataset(data_root='../data/nerf_example_data/nerf_synthetic/lego', mode='train',
                                     half_res=True, model_type='hashnerf')
    for img, pos, input in blender_dataset:
        print(img.shape)
        print(pos.shape)
        break
