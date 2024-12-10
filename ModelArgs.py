# 使用dataclass装饰器自动生成初始化方法和类的字符串表示方法
from dataclasses import dataclass
import math
from typing import Union
@dataclass
class ModelArgs:
    # @dataclass 会自动为这个类生成初始化方法和代表类的字符串形式的方法
    d_model: int # 定义模型的隐藏层维度
    n_layer: int  # 定义模型的层数
    vocab_size: int  # 定义词汇表的大小
    d_state: int = 16  # 定义状态空间的维度，默认为16
    expand: int = 2  # 定义扩展因子，默认为2
    dt_rank: Union[int, str] = 'auto'  # 定义输入依赖步长Δ的秩，'auto'表示自动设置
    d_conv: int = 4  # 定义卷积核的维度，默认为4
    pad_vocab_size_multiple: int = 8  # 定义词汇表大小的最小公倍数，默认为8
    conv_bias: bool = True  # 定义卷积层是否使用偏置项
    bias: bool = False  # 定义其他层（如线性层）是否使用偏置项

    def __post_init__(self):
        # 在__init__后自动被调用，用于执行初始化之后的额外设置或验证
        # 计算内部维度，即扩展后的维度
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':  # 如果dt_rank未指定，则自动计算设置
            # 根据隐藏层维度自动计算Δ的秩
            self.dt_rank = math.ceil(self.d_model / 16)
        # 确保vocab_size是pad_vocab_size_multiple的倍数
        # 如果不是，调整为最近的倍数
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
