# 基于Paddle的Instant-NGP复现

本项目中有关nerf部分的代码参考至：https://github.com/kongdebug/nerf-paddle.git。

本项目有关instant-ngp的部分参考至其pytorch实现方式：https://github.com/yashbhalgat/HashNeRF-pytorch

数据集获取：https://aistudio.baidu.com/datasetdetail/136989

其中数据集和模型的实现，均进行了一定程度的重写，以个人认为比较好理解和好阅读的方式进行了改写，其中本项目并没有针对精度和速度方面进行复现，但大体思路上的实现没有太大问题。

## 代码结构

* 在configs中保存了相关模型和数据集的参数细节，而在utils中的config_parse中进行参数的读取。
* 在dataset中存放了数据集读取的实现细节，本项目仅对nerf中lego数据集进行了复现，如果需要使用其他数据集，可以参考其中的实现。
* losses中存放nerf使用的mse损失以及hashnerf中提出的variation损失
* metrics中则存放有关psnr的计算实现
* model中存放nerf和hashnerf的实现细节，以及模型后处理部分的实现

# 训练代码

```
python train.py --config configs/nerf_lego.yml
```

# 测试代码

```
python test.py --config configs/nerf_lego.yml --save_dir save --load_from save/040000.pdparams
```

