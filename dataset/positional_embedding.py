import paddle
import paddle.nn as nn

class PositionalEmbedding(nn.Layer):

    def __init__(self,
                 input_dims,
                 include_input,
                 max_freq_log2,
                 num_freqs,
                 log_sampling,
                 periodic_fns=None
                 ):
        super(PositionalEmbedding, self).__init__()
        if periodic_fns is None:
            periodic_fns = [paddle.sin, paddle.cos]
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2. ** paddle.linspace(0., max_freq, num=N_freqs)
        else:
            freq_bands = paddle.linspace(2. ** 0., 2. ** max_freq, num=N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    def forward(self, inputs):

        return paddle.concat([fn(inputs) for fn in self.embed_fns], axis=-1)