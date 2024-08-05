import torch
import torch.nn as nn
import numpy as np
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, emb_size=768, img_size=224, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.img_size = img_size
        self.channels = channels

        # 计算沿宽度和高度方向的patch数量
        self.grid_size = img_size // patch_size
        # 计算总的patch数量
        self.num_patches = self.grid_size * self.grid_size
        # 创建一个线性层对patches进行嵌入，使用卷积层作为线性层可以高效地并行处理
        self.projection = nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [batch_size, channels, img_size, img_size]
        # 通过卷积层的操作将图像转换成一系列的patch embeddings
        x = self.projection(x)
        # 改变x的形状为 [batch_size, emb_size, num_patches]
        x = x.flatten(2)
        # 将patch的位置调换，以将其设置为序列的维度 [batch_size, num_patches, emb_size]
        x = x.transpose(1, 2)
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.num_patches = num_patches
        # 创建可学习的位置参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, emb_size))

    def forward(self, x):
        # 输入x形状为(batch_size, num_patches, emb_size)
        # 将位置编码添加到输入x中
        return x + self.position_embeddings


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入x的大小: [batch_size, sequence_length, dim]
        x = self.norm(x)  # LayerNorm不改变张量大小

        # self.to_qkv的线性层将维度从dim映射到inner_dim * 3
        # qkv的大小: [batch_size, sequence_length, inner_dim * 3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v的大小: [batch_size, sequence_length, inner_dim]，它们将被整理(rearrange)

        # 进行rearrange后，q, k, v的大小: [batch_size, heads, sequence_length, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 矩阵q和k转置乘积后dots的大小: [batch_size, heads, sequence_length, sequence_length]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 计算Softmax并不改变张量大小
        # attn的大小: [batch_size, heads, sequence_length, sequence_length]
        attn = self.attend(dots)
        attn = self.dropout(attn)  # Dropout也不改变张量大小

        # out乘积后的大小: [batch_size, heads, sequence_length, dim_head]
        out = torch.matmul(attn, v)

        # out经过rearrange后的大小: [batch_size, sequence_length, heads * dim_head (即inner_dim)]
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 输出的大小经过线性层（或者Identity）后可能为[batch_size, sequence_length, dim]，这取决于是否进行project_out
        return self.to_out(out)


class FeedForward(nn.Module):
    '''
    FeedForward Module

        Args:
            dim: int, the dimension of input features
            hidden_dim: int, the dimension of hidden layer
            drop: float, the dropout rate

        Shape:
            - Input: (B, N, C)
            - Output: (B, N, C)
    '''

    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class CMA(nn.Module):
    '''
    Cross-Modal Attention Module

        输入为两个模态的特征图，通过多头注意力机制实现两个模态之间的交互，

        两个模态的输入分别为Fv: (B, C, H, W)和Fa: (B, C, H, W)。

        首先对两个特征分别进行patch embedding,
        然后进行线性映射得到Q, K, V，；其中Fv->V，K；Fa->Q
        接着计算Q和K的点积得到attention map，再与V相乘得到交互后的特征。
        最后将交互后的特征与原始特征相加得到最终的输出。

        Args:
            input_channels: int, the dimension of input features
            num_heads: int, the number of heads in multi-head attention
            input_size: int, the size of input features
            patch_size: int, the size of patches


        Shape:
            - Fv: (B, C, H, W)
            - Fa: (B, C, H, W)
            - Output: (B, C, H, W)

    '''

    def __init__(self,
                 input_channels=96,
                 emb_size=96,
                 num_heads=4,
                 dim_head=64,
                 input_size=4,
                 patch_size=4,
                 mlp_channels=96,
                 att_drop=0.,
                 ffn_drop=0., ):
        super().__init__()
        num_patches = (input_size // patch_size) ** 2
        self.num_patches = num_patches
        self.emb_size = emb_size

        self.fv_patch_layer = nn.Sequential(
            PatchEmbedding(patch_size=patch_size, emb_size=emb_size, img_size=input_size, channels=input_channels),
            LearnablePositionalEncoding(num_patches=num_patches, emb_size=emb_size)
        )
        self.fa_patch_layer = nn.Sequential(
            PatchEmbedding(patch_size=patch_size, emb_size=emb_size, img_size=input_size, channels=input_channels),
            LearnablePositionalEncoding(num_patches=num_patches, emb_size=emb_size)
        )

        self.fv_patch_norm = nn.LayerNorm(emb_size)
        self.fa_patch_norm = nn.LayerNorm(emb_size)

        # 多头在注意力机制所需部分
        self.inner_dim = dim_head * num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        project_out = not (num_heads == 1 and dim_head == emb_size)

        self.att_norm = nn.LayerNorm(emb_size)
        self.attend = nn.Softmax(dim=-1)
        self.att_dropout = nn.Dropout(att_drop)

        self.kv_proj = nn.Linear(emb_size, self.inner_dim * 2)
        self.q_proj = nn.Linear(emb_size, self.inner_dim)
        self.att_out = nn.Sequential(
            nn.Linear(self.inner_dim, emb_size),
            nn.Dropout(att_drop)
        ) if project_out else nn.Identity()

        self.ffn = FeedForward(emb_size, mlp_channels, ffn_drop)

    def forward(self, Fv, Fa):
        # Fv: (B, C, H, W)
        # Fa: (B, C, H, W)
        Fv_ = Fv
        Fa_ = Fa
        # patch embedding
        Fv = self.fv_patch_layer(Fv)  # (B, num_patches, emb_size)
        Fa = self.fa_patch_layer(Fa)  # (B, num_patches, emb_size)
        # patch norm
        Fv = self.fv_patch_norm(Fv)  # (B, num_patches, emb_size)
        Fa = self.fa_patch_norm(Fa)  # (B, num_patches, emb_size)
        # linear projection
        kv = self.kv_proj(Fv)  # (B, num_patches, inner_dim * 2)
        q = self.q_proj(Fa)  # (B, num_patches, inner_dim)
        k = kv[:, :, :self.inner_dim]  # (B, num_patches, inner_dim)
        v = kv[:, :, self.inner_dim:]  # (B, num_patches, inner_dim)
        # k,q,v 从 (B, num_patches, inner_dim) 转化为 (b, num_heads, num_patches, dim_head)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        att = self.attend(dots)
        att = self.att_dropout(att)

        att = torch.matmul(att, v)
        att = rearrange(att, 'b h n d -> b n (h d)')
        att = self.att_out(att)

        # residual connection
        att = att + Fv
        # feed forward network (FFN) residual connection
        att = att + self.ffn(att)  # (B, num_patches, emb_size)
        # reshape to (B, C, H, W)
        att = rearrange(att, 'b (h w) d -> b d h w',
                        h=int(np.sqrt(self.num_patches)))  # (B, emb_size, sqrt(num_patches), sqrt(num_patches))
        return att
