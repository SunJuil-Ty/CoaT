"""
CoaT architecture.
Modified from timm/models/vision_transformer.py
"""
import collections.abc
import numpy as np
from itertools import repeat

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore.numpy import empty
from mindspore import Tensor
from mindspore import ms_function

#from ..data.constants import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from .registry import register_model
from .utils import load_pretrained



IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


__all__ = [
    "coat_tiny",
    "coat_mini",
    "coat_small",
    "coat_lite_tiny",
    "coat_lite_mini",
    "coat_lite_small",
    "coat_lite_medium"
]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'coat_tiny': _cfg(url=''),
    'coat_mini': _cfg(url=''),
    'coat_small': _cfg(url=''),
    'coat_lite_tiny': _cfg(url=''),
    'coat_lite_mini': _cfg(url=''),
    'coat_lite_small': _cfg(url=''),
    'coat_lite_medium':_cfg(url="")
}


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        return x


def drop_path(x: Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True) -> Tensor:
    """ DropPath (Stochastic Depth) regularization layers """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = ops.bernoulli(empty(shape), p=keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor = ops.div(random_tensor, keep_prob)
    return x * random_tensor


class DropPath(nn.Cell):
    """ DropPath (Stochastic Depth) regularization layers """
    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = nn.GELU(approximate=False)
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvRelPosEnc(nn.Cell):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.CellList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(in_channels=cur_head_split * Ch,
                                 out_channels=cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size, padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 group=cur_head_split * Ch,
                                 pad_mode='pad',
                                 has_bias=True,
                                 weight_init="TruncatedNormal"
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]

    def splt(self, img):
        length = len(self.channel_splits)

        img_list = [img[:, :self.channel_splits[0], :, :]]
        for n in range(1, length - 1):
            img_list.append(img[:, sum(self.channel_splits[:n]):sum(self.channel_splits[:n + 1]), :, :])
        img_list.append(img[:, sum(self.channel_splits[:length - 1]):, :, :])

        return img_list

    def construct(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]

        v_img = v_img.transpose((0, 1, 3, 2)).reshape((B, h * Ch, H, W))
        v_img_list = self.splt(v_img)
        #conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img_list = []
        for i, conv in enumerate(self.conv_list):
            conv_v_img_list.append(conv(v_img_list[i]))
        conv_v_img = ops.concat(conv_v_img_list, axis=1)
        conv_v_img = conv_v_img.reshape((B, h, Ch, H * W)).transpose((0, 1, 3, 2))

        EV_hat_img = q_img * conv_v_img
        zero = ops.Zeros()((B, h, 1, Ch), q.dtype)
        EV_hat = ops.concat((zero, EV_hat_img), axis=2)

        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Cell):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)

        self.crpe = shared_crpe

    def construct(self, x, size):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_softmax = ops.Softmax(axis=2)(k)
        factor_att = ops.BatchMatMul(transpose_b=True)(q, k_softmax)
        factor_att = ops.matmul(factor_att, v)

        crpe = self.crpe(q, v, size=size)

        x = self.scale * factor_att + crpe
        x = x.transpose((0, 2, 1, 3)).reshape((B, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ConvPosEnc(nn.Cell):
    """ Convolutional Position Encoding.
        Note: This module is similar to the conditional position encoding in CPVT.
    """

    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(in_channels=dim,
                              out_channels=dim,
                              kernel_size=k,
                              stride=1,
                              padding=k // 2,
                              group=dim,
                              pad_mode='pad',
                              has_bias=True,
                              weight_init="TruncatedNormal"
                              )

    def construct(self, x, size):
        B, N, C = x.shape
        H, W = size

        cls_token, img_tokens = x[:, :1], x[:, 1:]

        feat = img_tokens.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.proj(feat) + feat

        x = ops.reshape(x, (B, C, H * W))
        x = ops.transpose(x, (0, 2, 1))

        x = ops.concat((cls_token, x), axis=1)

        return x


class SerialBlock(nn.Cell):
    """
    Serial block class.
        Note: In this implementation, each serial block only contains a conv-attention and a FFN (MLP) module.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 shared_cpe=None,
                 shared_crpe=None):
        super().__init__()

        self.cpe = shared_cpe

        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim,
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attn_drop=attn_drop,
                                                      proj_drop=drop,
                                                      shared_crpe=shared_crpe
                                                      )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def construct(self, x, size):
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)

        cur = self.norm2(x)
        cur = self.mlp(cur)
        x = x + self.drop_path(cur)

        return x


class ParallelBlock(nn.Cell):
    """ Parallel block class. """

    def __init__(self,
                 dims,
                 num_heads,
                 mlp_ratios=[],
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 shared_crpes=None):
        super().__init__()

        self.norm12 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm13 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm14 = nn.LayerNorm((dims[3],), epsilon=1e-6)
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[1]
                                                       )
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[2]
                                                       )
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[3]
                                                       )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm22 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm23 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm24 = nn.LayerNorm((dims[3],), epsilon=1e-6)
        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, drop=drop)

    def upsample(self, x, output_size, size):
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        B, N, C = x.shape
        H, W = size

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = img_tokens.transpose([0, 2, 1]).reshape([B, C, H, W])
        img_tokens = ops.ResizeBilinear(output_size)(img_tokens)
        # img_tokens = ops.interpolate(img_tokens,
        #                              scales=scales,
        #                              mode='bilinear'
        #                              )
        img_tokens = img_tokens.reshape([B, C, -1]).transpose([0, 2, 1])

        out = ops.concat((cls_token, img_tokens), axis=1)

        return out

    def construct(self, x1, x2, x3, x4, sizes):
        _, (H2, W2), (H3, W3), (H4, W4) = sizes

        cur2 = self.norm12(x2)
        cur3 = self.norm13(x3)
        cur4 = self.norm14(x4)
        cur2 = self.factoratt_crpe2(cur2, size=(H2, W2))
        cur3 = self.factoratt_crpe3(cur3, size=(H3, W3))
        cur4 = self.factoratt_crpe4(cur4, size=(H4, W4))
        upsample3_2 = self.upsample(cur3, output_size=(H2,W2), size=(H3, W3))
        upsample4_3 = self.upsample(cur4, output_size=(H3,W3), size=(H4, W4))
        upsample4_2 = self.upsample(cur4, output_size=(H2,W2), size=(H4, W4))
        downsample2_3 = self.downsample(cur2, output_size=(H3,W3), size=(H2, W2))
        downsample3_4 = self.downsample(cur3, output_size=(H4,W4), size=(H3, W3))
        downsample2_4 = self.downsample(cur2, output_size=(H4,W4), size=(H2, W2))
        cur2 = cur2 + upsample3_2 + upsample4_2
        cur3 = cur3 + upsample4_3 + downsample2_3
        cur4 = cur4 + downsample3_4 + downsample2_4
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        cur2 = self.norm22(x2)
        cur3 = self.norm23(x3)
        cur4 = self.norm24(x4)
        cur2 = self.mlp2(cur2)
        cur3 = self.mlp3(cur3)
        cur4 = self.mlp4(cur4)
        x2 = x2 + self.drop_path(cur2)
        x3 = x3 + self.drop_path(cur3)
        x4 = x4 + self.drop_path(cur4)

        return x1, x2, x3, x4


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding """

    def __init__(self, image_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]

        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              pad_mode='pad',
                              has_bias=True,
                              weight_init="TruncatedNormal"
                              )

        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5)

    def construct(self, x):
        B = x.shape[0]

        x = ops.reshape(self.proj(x), (B, self.embed_dim, -1))
        x = ops.transpose(x, (0, 2, 1))
        x = self.norm(x)

        return x


class CoaT(nn.Cell):
    """ CoaT class. """

    def __init__(self,
                 image_size = 224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[0, 0, 0, 0],
                 serial_depths=[0, 0, 0, 0],
                 parallel_depth=0,
                 num_heads=0,
                 mlp_ratios=[0, 0, 0, 0],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 return_interm_layers=False,
                 out_features=None,
                 crpe_window={3: 2, 5: 3, 7: 3},
                 **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed(image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(image_size=image_size // (2**2), patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(image_size=image_size // (2**3), patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(image_size=image_size // (2**4), patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.cls_token1 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[0]), mindspore.float32))
        self.cls_token2 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[1]), mindspore.float32))
        self.cls_token3 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[2]), mindspore.float32))
        self.cls_token4 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[3]), mindspore.float32))

        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)

        dpr = drop_path_rate

        self.serial_blocks1 = nn.CellList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        self.serial_blocks2 = nn.CellList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        self.serial_blocks3 = nn.CellList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        self.serial_blocks4 = nn.CellList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )

        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.CellList([
                ParallelBlock(dims=embed_dims,
                              num_heads=num_heads,
                              mlp_ratios=mlp_ratios,
                              qkv_bias=qkv_bias,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr,
                              shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4]
                              )
                for _ in range(parallel_depth)]
            )
        else:
            self.parallel_blocks = None

        if not self.return_interm_layers:
            if self.parallel_blocks is not None:
                self.norm2 = nn.LayerNorm((embed_dims[1],), epsilon=1e-6)
                self.norm3 = nn.LayerNorm((embed_dims[2],), epsilon=1e-6)
            else:
                self.norm2 = None
                self.norm3 = None

            self.norm4 = nn.LayerNorm((embed_dims[3],), epsilon=1e-6)

            if self.parallel_depth > 0:
                self.aggregate = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, has_bias=True)    #may problem
                self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()
            else:
                self.aggregate = None
                self.head = nn.Dense(embed_dims[3], num_classes) if num_classes > 0 else Identity()

        self.cls_token1 = mindspore.Parameter(
            init.initializer(init.TruncatedNormal(sigma=.02), (1, 1, embed_dims[0]), mindspore.float32))
        self.cls_token2 = mindspore.Parameter(
            init.initializer(init.TruncatedNormal(sigma=.02), (1, 1, embed_dims[1]), mindspore.float32))
        self.cls_token3 = mindspore.Parameter(
            init.initializer(init.TruncatedNormal(sigma=.02), (1, 1, embed_dims[2]), mindspore.float32))
        self.cls_token4 = mindspore.Parameter(
            init.initializer(init.TruncatedNormal(sigma=.02), (1, 1, embed_dims[3]), mindspore.float32))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.Constant(1), cell.gamma.shape))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def insert_cls(self, x, cls_token):
        # print("x:", x.shape)
        # print("cls_token:", cls_token.shape)
        # print("target:", (x.shape[0], -1, -1))
        #
        y = Tensor(np.ones((x.shape[0], cls_token.shape[1], cls_token.shape[2])))
        cls_tokens = cls_token.expand_as(y)
        # print("cls_tokens:", cls_tokens.shape)
        # print("----------------")
        x = ops.concat((cls_tokens, x), axis=1)
        return x

    def remove_cls(self, x):
        return x[:, 1:, :]

    def forward_features(self, x0):
        B = x0.shape[0]

        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.patches_resolution
        x1 = self.insert_cls(x1, self.cls_token1)
        #cls_token1 = ops.broadcast_to(self.cls_token1, (x1.shape[0], 1, -1))
        #x1 = ops.concat((cls_token1, x1), axis=1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = x1[:, 1:, :].reshape([B, H1, W1, -1]).transpose([0, 3, 1, 2])

        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.patches_resolution
        x2 = self.insert_cls(x2, self.cls_token2)
        #cls_token2 = ops.broadcast_to(self.cls_token2, (x2.shape[0], 1, -1))
        #x2 = ops.concat((cls_token2, x2), axis=1)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = x2[:, 1:, :].reshape([B, H2, W2, -1]).transpose([0, 3, 1, 2])

        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.patches_resolution
        x3 = self.insert_cls(x3, self.cls_token3)
        #cls_token3 = ops.broadcast_to(self.cls_token3, (x3.shape[0], 1, -1))
        #x3 = ops.concat((cls_token3, x3), axis=1)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = x3[:, 1:, :].reshape([B, H3, W3, -1]).transpose([0, 3, 1, 2])

        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.patches_resolution
        x4 = self.insert_cls(x4, self.cls_token4)
        #cls_token4 = ops.broadcast_to(self.cls_token4, (x4.shape[0], 1, -1))
        #x4 = ops.concat((cls_token4, x4), axis=1)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = x4[:, 1:, :].reshape([B, H4, W4, -1]).transpose([0, 3, 1, 2])

        if self.parallel_depth == 0:
            if self.return_interm_layers:
                feat_out = {}
                if 'x1_nocls' in self.out_features:
                    feat_out['x1_nocls'] = x1_nocls
                if 'x2_nocls' in self.out_features:
                    feat_out['x2_nocls'] = x2_nocls
                if 'x3_nocls' in self.out_features:
                    feat_out['x3_nocls'] = x3_nocls
                if 'x4_nocls' in self.out_features:
                    feat_out['x4_nocls'] = x4_nocls
                return feat_out
            else:
                x4 = self.norm4(x4)
                x4_cls = x4[:, 0]
                return x4_cls

        for blk in self.parallel_blocks:
            x2 = self.cpe2(x2, (H2, W2))
            x3 = self.cpe3(x3, (H3, W3))
            x4 = self.cpe4(x4, (H4, W4))
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = x1[:, 1:, :].reshape([B, H1, W1, -1]).transpose([0, 3, 1, 2])
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = x2[:, 1:, :].reshape([B, H2, W2, -1]).transpose([0, 3, 1, 2])
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = x3[:, 1:, :].reshape([B, H3, W3, -1]).transpose([0, 3, 1, 2])
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = x4[:, 1:, :].reshape([B, H4, W4, -1]).transpose([0, 3, 1, 2])
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            merged_cls = ops.concat((x2_cls, x3_cls, x4_cls), axis=1)
            merged_cls = self.aggregate(merged_cls).squeeze(axis=1)
            return merged_cls

    def construct(self, x):
        if self.return_interm_layers:
            return self.forward_features(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


@register_model
def coat_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_mini']
    model = CoaT(patch_size=4, embed_dims=[152, 152, 152, 152], serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_mini']
    model = CoaT(patch_size=4, embed_dims=[152, 216, 216, 216], serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_small']
    model = CoaT(patch_size=4, embed_dims=[152, 320, 320, 320], serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_tiny']
    model = CoaT(patch_size=4, embed_dims=[64, 128, 256, 320], serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_mini']
    model = CoaT(patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_small']
    model = CoaT(patch_size=4, embed_dims=[64, 128, 320, 512], serial_depths=[3, 4, 6, 3], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_medium(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_medium']
    model = CoaT(patch_size=4, embed_dims=[128, 256, 320, 512], serial_depths=[3, 6, 10, 8], parallel_depth=0,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model
