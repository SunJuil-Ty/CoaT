"""
CoaT architecture.
Modified from timm/models/vision_transformer.py
"""
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as init
from mindspore.numpy import ones, split
from mindspore import Tensor
from mindspore import ms_function

from registry import register_model
# from .utils import load_pretrained


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
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
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


class DropPath(nn.Cell):
    """ DropPath (Stochastic Depth) regularization layers """
    def __init__(self,
                 drop_prob: float = 0.,
                 scale_by_keep: bool = True) -> None:
        super().__init__()
        self.keep_prob = 1. - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1. or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class Mlp(nn.Cell):
    """Feed-forward network (FFN, a.k.a. MLP) class."""
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
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
        """
        Initialization.
            Ch: Channels per head.
            h: Number of heads.
            window: Window size(s) in convolutional relative positional encoding. It can have two forms:
                    1. An integer of window size, which assigns all attention heads with the same window size in ConvRelPosEnc.
                    2. A dict mapping window size to #attention head splits (e.g. {window size 1: #attention head split 1, window size 2: #attention head split 2})
                        It will apply different window size to the attention head splits.
        """
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
                                 weight_init="HeUniform",
                                 bias_init="Uniform"
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.idx1 = self.channel_splits[0]
        self.idx2 = self.channel_splits[0] + self.channel_splits[1]

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.zeros = ops.Zeros()
        self.cast = ops.Cast()

    def construct(self, q, v, size):

        B, h, N, Ch = q.shape
        H, W = size

        # Convolutional relative position encoding.
        q_img = q[:, :, 1:, :]
        v_img = v[:, :, 1:, :]

        v_img = self.transpose(v_img, (0, 1, 3, 2))
        v_img = self.reshape(v_img, (B, h * Ch, H, W))

        v_img_list = split(x=v_img, indices_or_sections=[self.idx1, self.idx2], axis=1)
        conv_v_img_list = []
        i = 0
        for conv in self.conv_list:
            conv_v_img_list.append(conv(v_img_list[i]))
            i = i + 1

        conv_v_img = ops.concat(conv_v_img_list, axis=1)

        conv_v_img = self.reshape(conv_v_img, (B, h, Ch, H * W))
        conv_v_img = self.transpose(conv_v_img, (0, 1, 3, 2))

        EV_hat_img = q_img * conv_v_img
        zero = self.zeros((B, h, 1, Ch), mindspore.float32)
        #zero = ops.Zeros()((B, h, 1, Ch), q.dtype)
        EV_hat_img = self.cast(EV_hat_img, mindspore.float32)
        EV_hat = ops.concat((zero, EV_hat_img), axis=2)

        return EV_hat


class FactorAtt_ConvRelPosEnc(nn.Cell):
    """Factorized attention with convolutional relative position encoding class."""

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 shared_crpe=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(keep_prob=1 - proj_drop)

        # Shared convolutional relative position encoding.
        self.crpe = shared_crpe

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.softmax = nn.Softmax(axis=2)
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.attn_matmul_v = ops.BatchMatMul()
        #self.batch_matmul = ops.BatchMatMul()

    def construct(self, x, size):
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)
        # q, k, v = qkv[0], qkv[1], qkv[2]

        # Factorized attention.
        k_softmax = self.softmax(k)
        factor_att = self.q_matmul_k(q, k_softmax)
        factor_att = self.attn_matmul_v(factor_att, v)
        # k_softmax = ops.Softmax(axis=2)(k)
        # factor_att = ops.BatchMatMul(transpose_b=True)(q, k_softmax)
        # factor_att = ops.matmul(factor_att, v)

        # Convolutional relative position encoding.
        crpe = self.crpe(q, v, size=size)

        # Merge and reshape.
        x = self.scale * factor_att + crpe
        # x = ops.mul(self.scale, factor_att)
        # x = ops.add(x, crpe)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (B, N, C))

        # Output projection.
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
                              weight_init="HeUniform",
                              bias_init="Uniform"
                              )

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.cast = ops.Cast()

    def construct(self, x, size):
        B, N, C = x.shape
        H, W = size

        # Extract CLS token and image tokens.
        cls_token, img_tokens = x[:, :1], x[:, 1:]

        # Depthwise convolution.
        feat = self.transpose(img_tokens, (0, 2, 1))
        feat = self.reshape(feat, (B, C, H, W))
        # x = ops.add(self.proj(feat), feat)
        x = self.proj(feat) + feat
        x = self.reshape(x, (B, C, H * W))
        x = self.transpose(x, (0, 2, 1))

        # Combine with CLS token.
        x = self.cast(x, mindspore.float32)
        cls_token = self.cast(cls_token, mindspore.float32)
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
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 shared_cpe=None,
                 shared_crpe=None):
        super().__init__()

        self.cpe = shared_cpe

        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.factoratt_crpe = FactorAtt_ConvRelPosEnc(dim,
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      qk_scale=qk_scale,
                                                      attn_drop=attn_drop,
                                                      proj_drop=drop,
                                                      shared_crpe=shared_crpe
                                                      )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, size):
        # Conv-Attention.
        # x = x + self.drop_path(self.factoratt_crpe(self.norm1(self.cpe(x, size)), size))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.cpe(x, size)
        cur = self.norm1(x)
        cur = self.factoratt_crpe(cur, size)
        x = x + self.drop_path(cur)

        # MLP.
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
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 shared_cpes=None,
                 shared_crpes=None):
        super().__init__()

        self.cpes = shared_cpes

        self.norm12 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm13 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm14 = nn.LayerNorm((dims[3],), epsilon=1e-6)
        self.factoratt_crpe2 = FactorAtt_ConvRelPosEnc(dims[1],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       qk_scale=qk_scale,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[1]
                                                       )
        self.factoratt_crpe3 = FactorAtt_ConvRelPosEnc(dims[2],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       qk_scale=qk_scale,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[2]
                                                       )
        self.factoratt_crpe4 = FactorAtt_ConvRelPosEnc(dims[3],
                                                       num_heads=num_heads,
                                                       qkv_bias=qkv_bias,
                                                       qk_scale=qk_scale,
                                                       attn_drop=attn_drop,
                                                       proj_drop=drop,
                                                       shared_crpe=shared_crpes[3]
                                                       )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm22 = nn.LayerNorm((dims[1],), epsilon=1e-6)
        self.norm23 = nn.LayerNorm((dims[2],), epsilon=1e-6)
        self.norm24 = nn.LayerNorm((dims[3],), epsilon=1e-6)

        mlp_hidden_dim = int(dims[1] * mlp_ratios[1])
        self.mlp2 = self.mlp3 = self.mlp4 = Mlp(in_features=dims[1], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def upsample(self, x, output_size, size):
        """ Feature map up-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def downsample(self, x, output_size, size):
        """ Feature map down-sampling. """
        return self.interpolate(x, output_size=output_size, size=size)

    def interpolate(self, x, output_size, size):
        """ Feature map interpolation. """
        B, N, C = x.shape
        H, W = size

        cls_token = x[:, :1, :]
        img_tokens = x[:, 1:, :]

        img_tokens = ops.transpose(img_tokens, (0, 2, 1))
        img_tokens = ops.reshape(img_tokens, (B, C, H, W))
        #img_tokens = ops.ResizeBilinear(output_size)(img_tokens)
        img_tokens = ops.interpolate(img_tokens,
                                     sizes=output_size,
                                     mode='bilinear'
                                     )
        img_tokens = ops.reshape(img_tokens, (B, C, -1))
        img_tokens = ops.transpose(img_tokens, (0, 2, 1))

        cls_token = ops.Cast()(cls_token, mindspore.float32)
        img_tokens = ops.Cast()(img_tokens, mindspore.float32)
        out = ops.concat((cls_token, img_tokens), axis=1)

        return out

    def construct(self, x1, x2, x3, x4, sizes):
        _, (H2, W2), (H3, W3), (H4, W4) = sizes

        # Conv-Attention.
        x2 = self.cpes[1](x2, size=(H2, W2))  # Note: x1 is ignored.
        x3 = self.cpes[2](x3, size=(H3, W3))
        x4 = self.cpes[3](x4, size=(H4, W4))

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

        # MLP.
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

    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
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
                              weight_init="HeUniform",
                              bias_init="Uniform"
                              )

        self.norm = nn.LayerNorm((embed_dim,), epsilon=1e-5)

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        B = x.shape[0]

        x = self.reshape(self.proj(x), (B, self.embed_dim, -1))
        x = self.transpose(x, (0, 2, 1))
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
                 return_interm_layers=False,
                 out_features=None,
                 crpe_window={3: 2, 5: 3, 7: 3},
                 **kwargs):
        super().__init__()
        self.return_interm_layers = return_interm_layers
        self.out_features = out_features
        self.num_classes = num_classes

        # Patch embeddings.
        self.patch_embed1 = PatchEmbed(image_size=image_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(image_size=image_size // (2**2), patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(image_size=image_size // (2**3), patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(image_size=image_size // (2**4), patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Class tokens.
        self.cls_token1 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[0]), mindspore.float32))
        self.cls_token2 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[1]), mindspore.float32))
        self.cls_token3 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[2]), mindspore.float32))
        self.cls_token4 = mindspore.Parameter(ops.Zeros()((1, 1, embed_dims[3]), mindspore.float32))

        # Convolutional position encodings.
        self.cpe1 = ConvPosEnc(dim=embed_dims[0], k=3)
        self.cpe2 = ConvPosEnc(dim=embed_dims[1], k=3)
        self.cpe3 = ConvPosEnc(dim=embed_dims[2], k=3)
        self.cpe4 = ConvPosEnc(dim=embed_dims[3], k=3)

        # Convolutional relative position encodings.
        self.crpe1 = ConvRelPosEnc(Ch=embed_dims[0] // num_heads, h=num_heads, window=crpe_window)
        self.crpe2 = ConvRelPosEnc(Ch=embed_dims[1] // num_heads, h=num_heads, window=crpe_window)
        self.crpe3 = ConvRelPosEnc(Ch=embed_dims[2] // num_heads, h=num_heads, window=crpe_window)
        self.crpe4 = ConvRelPosEnc(Ch=embed_dims[3] // num_heads, h=num_heads, window=crpe_window)

        # Enable stochastic depth.
        dpr = drop_path_rate

        # Serial blocks 1.
        self.serial_blocks1 = nn.CellList([
            SerialBlock(
                dim=embed_dims[0], num_heads=num_heads, mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe1, shared_crpe=self.crpe1
            )
            for _ in range(serial_depths[0])]
        )

        # Serial blocks 2.
        self.serial_blocks2 = nn.CellList([
            SerialBlock(
                dim=embed_dims[1], num_heads=num_heads, mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe2, shared_crpe=self.crpe2
            )
            for _ in range(serial_depths[1])]
        )

        # Serial blocks 3.
        self.serial_blocks3 = nn.CellList([
            SerialBlock(
                dim=embed_dims[2], num_heads=num_heads, mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe3, shared_crpe=self.crpe3
            )
            for _ in range(serial_depths[2])]
        )

        # Serial blocks 4.
        self.serial_blocks4 = nn.CellList([
            SerialBlock(
                dim=embed_dims[3], num_heads=num_heads, mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr,
                shared_cpe=self.cpe4, shared_crpe=self.crpe4
            )
            for _ in range(serial_depths[3])]
        )

        # Parallel blocks.
        self.parallel_depth = parallel_depth
        if self.parallel_depth > 0:
            self.parallel_blocks = nn.CellList([
                ParallelBlock(dims=embed_dims,
                              num_heads=num_heads,
                              mlp_ratios=mlp_ratios,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              drop=drop_rate,
                              attn_drop=attn_drop_rate,
                              drop_path=dpr,
                              shared_cpes=[self.cpe1, self.cpe2, self.cpe3, self.cpe4],
                              shared_crpes=[self.crpe1, self.crpe2, self.crpe3, self.crpe4]
                              )
                for _ in range(parallel_depth)]
            )

        # Classification head(s).
        if not self.return_interm_layers:
            self.norm1 = nn.LayerNorm((embed_dims[0],), epsilon=1e-6)
            self.norm2 = nn.LayerNorm((embed_dims[1],), epsilon=1e-6)
            self.norm3 = nn.LayerNorm((embed_dims[2],), epsilon=1e-6)
            self.norm4 = nn.LayerNorm((embed_dims[3],), epsilon=1e-6)

            # CoaT series: Aggregate features of last three scales for classification.
            if self.parallel_depth > 0:
                self.aggregate = nn.Conv1d(in_channels=3,
                                           out_channels=1,
                                           kernel_size=1,
                                           has_bias=True,
                                           weight_init="HeUniform",
                                           bias_init="Uniform"
                                           )    #may problem
                self.head = nn.Dense(embed_dims[3], num_classes)
            # CoaT-Lite series: Use feature of last scale for classification.
            else:
                self.head = nn.Dense(embed_dims[3], num_classes)

        self.cls_token1.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token1.data.shape))
        self.cls_token2.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token2.data.shape))
        self.cls_token3.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token3.data.shape))
        self.cls_token4.set_data(init.initializer(init.TruncatedNormal(sigma=.02), self.cls_token4.data.shape))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(init.TruncatedNormal(sigma=.02), cell.weight.data.shape))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(init.Constant(0), cell.bias.shape))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(init.initializer(init.Constant(1.0), cell.gamma.shape))
                cell.beta.set_data(init.initializer(init.Constant(0), cell.beta.shape))

    def insert_cls(self, x, cls_token):

        t0 = x.shape[0]
        t1 = cls_token.shape[1]
        t2 = cls_token.shape[2]
        y = Tensor(np.ones((t0, t1, t2)))
        cls_tokens = cls_token.expand_as(y)

        x = ops.Cast()(x, mindspore.float32)
        cls_tokens = ops.Cast()(cls_tokens, mindspore.float32)
        x = ops.concat((cls_tokens, x), axis=1)
        return x

    def remove_cls(self, x):
        return x[:, 1:, :]

    def forward_features(self, x0):
        B = x0.shape[0]

        x1 = self.patch_embed1(x0)
        H1, W1 = self.patch_embed1.patches_resolution
        x1 = self.insert_cls(x1, self.cls_token1)
        # cls_token1 = ops.broadcast_to(self.cls_token1, (x1.shape[0], 1, -1))
        # x1 = ops.concat((cls_token1, x1), axis=1)
        for blk in self.serial_blocks1:
            x1 = blk(x1, size=(H1, W1))
        x1_nocls = self.remove_cls(x1)
        x1_nocls = ops.reshape(x1_nocls, (B, H1, W1, -1))
        x1_nocls = ops.transpose(x1_nocls, (0, 3, 1, 2))

        x2 = self.patch_embed2(x1_nocls)
        H2, W2 = self.patch_embed2.patches_resolution
        x2 = self.insert_cls(x2, self.cls_token2)
        # cls_token2 = ops.broadcast_to(self.cls_token2, (x2.shape[0], 1, -1))
        # x2 = ops.concat((cls_token2, x2), axis=1)
        for blk in self.serial_blocks2:
            x2 = blk(x2, size=(H2, W2))
        x2_nocls = self.remove_cls(x2)
        x2_nocls = ops.reshape(x2_nocls, (B, H2, W2, -1))
        x2_nocls = ops.transpose(x2_nocls, (0, 3, 1, 2))

        x3 = self.patch_embed3(x2_nocls)
        H3, W3 = self.patch_embed3.patches_resolution
        x3 = self.insert_cls(x3, self.cls_token3)
        # cls_token3 = ops.broadcast_to(self.cls_token3, (x3.shape[0], 1, -1))
        # x3 = ops.concat((cls_token3, x3), axis=1)
        for blk in self.serial_blocks3:
            x3 = blk(x3, size=(H3, W3))
        x3_nocls = self.remove_cls(x3)
        x3_nocls = ops.reshape(x3_nocls, (B, H3, W3, -1))
        x3_nocls = ops.transpose(x3_nocls, (0, 3, 1, 2))

        x4 = self.patch_embed4(x3_nocls)
        H4, W4 = self.patch_embed4.patches_resolution
        x4 = self.insert_cls(x4, self.cls_token4)
        # cls_token4 = ops.broadcast_to(self.cls_token4, (x4.shape[0], 1, -1))
        # x4 = ops.concat((cls_token4, x4), axis=1)
        for blk in self.serial_blocks4:
            x4 = blk(x4, size=(H4, W4))
        x4_nocls = self.remove_cls(x4)
        x4_nocls = ops.reshape(x4_nocls, (B, H4, W4, -1))
        x4_nocls = ops.transpose(x4_nocls, (0, 3, 1, 2))

        # Only serial blocks: Early return.
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
            x1, x2, x3, x4 = blk(x1, x2, x3, x4, sizes=[(H1, W1), (H2, W2), (H3, W3), (H4, W4)])

        if self.return_interm_layers:
            feat_out = {}
            if 'x1_nocls' in self.out_features:
                x1_nocls = self.remove_cls(x1)
                x1_nocls = x1_nocls.reshape((B, H1, W1, -1)).transpose((0, 3, 1, 2))
                feat_out['x1_nocls'] = x1_nocls
            if 'x2_nocls' in self.out_features:
                x2_nocls = self.remove_cls(x2)
                x2_nocls = x2_nocls.reshape((B, H2, W2, -1)).transpose((0, 3, 1, 2))
                feat_out['x2_nocls'] = x2_nocls
            if 'x3_nocls' in self.out_features:
                x3_nocls = self.remove_cls(x3)
                x3_nocls = x3_nocls.reshape((B, H3, W3, -1)).transpose((0, 3, 1, 2))
                feat_out['x3_nocls'] = x3_nocls
            if 'x4_nocls' in self.out_features:
                x4_nocls = self.remove_cls(x4)
                x4_nocls = x4_nocls.reshape((B, H4, W4, -1)).transpose((0, 3, 1, 2))
                feat_out['x4_nocls'] = x4_nocls
            return feat_out
        else:
            x2 = self.norm2(x2)
            x3 = self.norm3(x3)
            x4 = self.norm4(x4)
            x2_cls = x2[:, :1]
            x3_cls = x3[:, :1]
            x4_cls = x4[:, :1]
            x2_cls = ops.Cast()(x2_cls, mindspore.float32)
            x3_cls = ops.Cast()(x3_cls, mindspore.float32)
            x4_cls = ops.Cast()(x4_cls, mindspore.float32)
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
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 152, 152, 152],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_mini']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 216, 216, 216],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_small']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[152, 320, 320, 320],
                 serial_depths=[2, 2, 2, 2], parallel_depth=6,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_tiny(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_tiny']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 256, 320],
                 serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_mini(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_mini']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 320, 512],
                 serial_depths=[2, 2, 2, 2], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_small(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_small']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[64, 128, 320, 512],
                 serial_depths=[3, 4, 6, 3], parallel_depth=0,
                 num_heads=8, mlp_ratios=[8, 8, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model


@register_model
def coat_lite_medium(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    default_cfg = default_cfgs['coat_lite_medium']
    model = CoaT(in_channels=in_channels, num_classes=num_classes,
                 patch_size=4, embed_dims=[128, 256, 320, 512],
                 serial_depths=[3, 6, 10, 8], parallel_depth=0,
                 num_heads=8, mlp_ratios=[4, 4, 4, 4], **kwargs)
    model.default_cfg = default_cfg

    # if pretrained:
    #     load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_channels)
    return model

if __name__ == '__main__':
    import numpy as np
    import mindspore
    from mindspore import Tensor

    model = coat_lite_tiny()
    mindspore.save_checkpoint(model, "D:\openi(HW)\coat_lite_tiny.ckpt")
    print(model)
    dummy_input = Tensor(np.random.rand(8, 3, 224, 224), dtype=mindspore.float32)
    y = model(dummy_input)
    print(y.shape)
    # from mindspore import load_checkpoint, load_param_into_net
    #
    # params_dict = load_checkpoint("D:\\openi(HW)\\pth2ckpt_map\\coat_lite_tiny.ckpt")
    # load_param_into_net(model, params_dict)