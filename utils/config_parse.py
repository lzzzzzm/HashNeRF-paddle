import configargparse

def config_parse():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', default='configs/nerf_lego.yml', is_config_file=True, help='config file path')
    # dataset define
    parser.add_argument("--dataset_type", type=str, help='dataset type')
    parser.add_argument("--dataset_root", type=str, help='dataset_root')
    parser.add_argument("--half_res", type=bool, help='half_res')  # half res to test
    parser.add_argument("--render_phi", type=float, help='render_phi')
    parser.add_argument("--render_radius", type=float, help='render_radius')

    parser.add_argument("--pts_input_ch", type=int, help='pts_input_ch')
    parser.add_argument("--view_input_ch", type=int, help='view_input_ch')
    parser.add_argument("--pts_embedding_freqs", type=int, help='pts_embedding_freqs')
    parser.add_argument("--view_embeding_freqs", type=int, help='view_embeding_freqs')

    # train params
    parser.add_argument("--lrate", type=float, help='lrate')
    parser.add_argument("--lrate_decay", type=float, help='lrate_decay')
    parser.add_argument("--batch_size", type=int, help='batch_size')
    parser.add_argument("--train_iters", type=int, help='train_iters')
    parser.add_argument("--N_rand", type=int, help='N_rand')
    parser.add_argument("--sparse_loss_weight", type=float, help='1e-10')
    parser.add_argument("--tv_loss_weight", type=float, help='1e-6')
    parser.add_argument("--return_sparsity_loss", type=bool, help='return_sparsity_loss')

    # model params define
    parser.add_argument("--model_type", type=str, help='model_type')
    parser.add_argument("--input_ch", type=int, help='dataset input_ch')
    parser.add_argument("--input_ch_views", type=int, help='input_ch_views input_ch')
    parser.add_argument("--use_viewdirs", type=bool, help='use_viewviders')

    # pretrained weight define
    parser.add_argument("--load_from", type=str, help='load_from')

    # eval or test
    parser.add_argument("--eval", action='store_true')

    # log out name
    parser.add_argument("--log_out", default='test_out.log', type=str, help='load_from')

    # save_dir
    parser.add_argument("--save_dir", default=None, type=str, help='save_dir')

    # infer params
    parser.add_argument("--num_workers", default=2, type=int, help='num_workers')
    parser.add_argument("--chunk", default=1024 * 32, type=int, help='chunk')

    args = parser.parse_args()
    return args