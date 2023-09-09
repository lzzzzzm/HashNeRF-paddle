import paddle
import numpy as np
from .render_helper import get_rays_from_dir

trans_t = lambda t: paddle.to_tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).astype('float32')

rot_phi = lambda phi: paddle.to_tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).astype('float32')

rot_theta = lambda th: paddle.to_tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).astype('float32')


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = paddle.to_tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])).astype('float32') @ c2w
    return c2w

def create_meshgrid(H, W, normalized_coordinates=False):
    xs = paddle.linspace(0, W - 1, W)
    ys = paddle.linspace(0, H - 1, H)
    if normalized_coordinates:
        xs = (xs / (W - 1) - 0.5) * 2
        ys = (ys / (H - 1) - 0.5) * 2

    base_grid = paddle.stack(paddle.meshgrid([xs, ys], indexing="ij"), axis=-1)  # WxHx2
    return paddle.transpose(base_grid, (0, 1, 2)).unsqueeze(0)

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = paddle.unbind(grid, axis=-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        paddle.stack([(i-W/2)/focal, -(j-H/2)/focal, -paddle.ones_like(i)], -1) # (H, W, 3)

    dir_bounds = directions.reshape((-1, 3))
    # print("Directions ", directions[0,0,:], directions[H-1,0,:], directions[0,W-1,:], directions[H-1, W-1, :])
    # print("Directions ", dir_bounds[0], dir_bounds[W-1], dir_bounds[H*W-W], dir_bounds[H*W-1])

    return directions

def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = paddle.to_tensor(frame["transform_matrix"], dtype='float32')
        rays_o, rays_d = get_rays_from_dir(directions, c2w)

        def find_min_max(pt):
            for i in range(3):
                if (min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if (max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W - 1, H * W - W, H * W - 1]:
            min_point = rays_o[i] + near * rays_d[i]
            max_point = rays_o[i] + far * rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)
    x = paddle.to_tensor(min_bound) - paddle.to_tensor([1.0, 1.0, 1.0])
    bounding_box = [paddle.to_tensor(min_bound) - paddle.to_tensor([1.0, 1.0, 1.0]), paddle.to_tensor(max_bound) + paddle.to_tensor([1.0, 1.0, 1.0])]
    bounding_box[0] = bounding_box[0][:, 0]
    bounding_box[1] = bounding_box[1][:, 0]
    return bounding_box
