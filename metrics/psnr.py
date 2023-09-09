import paddle
import paddle.nn as nn

class MSEPSNR(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, mse):
        psnr = -10. * paddle.log(mse) /paddle.log(paddle.to_tensor([10.]))

        return psnr