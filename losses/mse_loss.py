import paddle
import paddle.nn as nn

class MSELoss(nn.Layer):
    """Focal loss class
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        loss = paddle.mean((prediction - target) ** 2)

        return loss

if __name__ == '__main__':
    loss_fn = MSELoss()
    pred = paddle.rand(shape=(1024, 3))
    target = paddle.rand(shape=(1, 1024, 4))
    loss = loss_fn(pred, target[:, :, :3])
