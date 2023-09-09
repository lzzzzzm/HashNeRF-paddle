import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class HashNeRF(nn.Layer):
    def __init__(self,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 input_ch=3,
                 input_ch_views=3
                 ):
        """
        """
        super(HashNeRF, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.num_layers_color = num_layers_color
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.hidden_dim_color = hidden_dim_color
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias_attr=False))
        self.sigma_net = nn.LayerList(sigma_net)


        # color network
        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias_attr=False))

        self.color_net = nn.LayerList(color_net)

    def forward(self, x):
        input_pts, input_views = paddle.split(x, num_or_sections=[self.input_ch, self.input_ch_views], axis=-1)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = paddle.concat([input_views, geo_feat], axis=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h)

        # color = torch.sigmoid(h)
        color = h
        outputs = paddle.concat([color, paddle.unsqueeze(sigma, -1)], -1)

        return outputs