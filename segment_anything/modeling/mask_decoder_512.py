# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import List, Tuple, Type

from .common import LayerNorm2d
from torch import Tensor
import math
import numpy as np
from .utils import noise_list
from .mult_attention import CIF_block


class MaskDecoder_512(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 768,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # HQ-SAM parameters
        self.hf_token = nn.Embedding(1, transformer_dim) # HQ-Ouptput-Token
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) # corresponding new MLP layer for HQ-Ouptput-Token
        # self.num_mask_tokens = self.num_mask_tokens + 1
        
        # three conv fusion layers for obtaining HQ-Feature
        self.compress_vit_feat = nn.Sequential(
                                        nn.Conv2d(vit_dim, transformer_dim, kernel_size=1))
        # self.compress_vit_feat = nn.Sequential(
        #                                 nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
        #                                 LayerNorm2d(transformer_dim),
        #                                 nn.GELU(), 
        #                                 nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )
        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        self.Att = CIF_block(256)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings1: torch.Tensor,
        interm_embeddings2: torch.Tensor,
        interm_embeddings3: torch.Tensor,
        mode = 'train',
        gt=None,
        img_size = 224
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # print("==========")
        vit_features1 = interm_embeddings1[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.compress_vit_feat(vit_features1)
        # print("vit_features:",vit_features.shape)  #torch.Size([8, 768, 32, 32])
        # print("image_embeddings:",image_embeddings.shape)  #torch.Size([8, 256, 32, 32])
        # hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        # print("hq_features:",hq_features.shape)  #hq_features: torch.Size([8, 32, 128, 128])
        # print("==========")
        vit_features2 = interm_embeddings2[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features2 = self.compress_vit_feat(vit_features2)
        vit_features3 = interm_embeddings3[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features3 = self.compress_vit_feat(vit_features3)
        masks, iou_pred, attn_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
            hq_features2=hq_features2,
            hq_features3=hq_features3,
            mode = mode,
            gt=gt,
            img_size=img_size
        )

        # Prepare output
        return masks, iou_pred, attn_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
        hq_features2: torch.Tensor,
        hq_features3: torch.Tensor,
        mode='train',
        gt=None,
        img_size=224
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)  #torch.Size([11, 256])
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  #torch.Size([1, 11, 256])
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  #torch.Size([1, 11, 256])

        ### 结果很好
        src=self.Att(image_embeddings,hq_features)
        ###
        # ###
        # src=self.Att(image_embeddings,hq_features3)   #结果很差
        # src=self.Att(image_embeddings,hq_features2)
        # src=self.Att(image_embeddings,hq_features)
        # ###
        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # src = src + dense_prompt_embeddings
        src = image_embeddings + dense_prompt_embeddings  #torch.Size([8, 256, 32, 32])
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        # ###
        # src=src+hq_features+hq_features2+hq_features3  #结果很差
        # ##
       
        # Run the transformer
        hs, src, att_out= self.transformer(src, pos_src, tokens)  #hs:torch.Size([8, 11, 256])  src:torch.Size([8, 1024, 256])
        iou_token_out = hs[:, 0, :]  #torch.Size([8, 256])
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]   #torch.Size([8, 9, 256]) 
        msk_feat = torch.matmul(mask_tokens_out,src.transpose(1, 2))   # torch.Size([8, 9, 1024])

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)   #torch.Size([8, 256, 32, 32])
        src=self.Att(src,hq_features)  #ASIF+ASIF
        upscaled_embedding = self.output_upscaling(src)  #torch.Size([8, 32, 128, 128])
        # upscaled_embedding_sam = self.output_upscaling(src)  #torch.Size([8, 32, 128, 128])
        ## upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features.repeat(b,1,1,1)
        # upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features2+hq_features3


        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # for i in range(self.num_mask_tokens):
        #     if i < self.num_mask_tokens - 1:
        #         hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        #     else:
        #         hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]


        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings
        # masks_sam = (hyper_in[:,:self.num_mask_tokens-1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        # masks_sam_hq = (hyper_in[:,self.num_mask_tokens-1:] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        # masks = torch.cat([masks_sam,masks_sam_hq],dim=1)
        # return masks, iou_pred, msk_feat, upscaled_embedding_hq
        return masks, iou_pred, att_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder2_512(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer2: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer2 = transformer2

        self.num_multimask_outputs = num_multimask_outputs

        # self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.skip_connect = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 8),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 16),
            activation(),
        )
        self.output_hypernetworks_mlps2 = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 5)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        self.med_sel = nn.Sequential(
            nn.Linear(self.num_mask_tokens,self.num_mask_tokens),
            nn.ReLU()
        )

        self.self_attn = Attention(
            1024, num_heads=8
        )
        self.norm1 = nn.LayerNorm(1024)
        self.self_attn2 = Attention(
            1024, num_heads=8
        )
        self.norm2 = nn.LayerNorm(1024)
        self.mlp = MLPBlock(1024, 2048)
        self.med_sel = nn.Sequential(
            nn.Linear(self.num_mask_tokens,1),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        mask_feat: torch.Tensor,
        gt=None,
        mode='test',
        msk_feat=None,
        up_embed=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred, attn_out = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            mask_feat=mask_feat,
            gt = gt,
            mode = mode,
            msk_feat=msk_feat,
            up_embed=up_embed
        )

        # Prepare output
        return masks, iou_pred, attn_out

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        mask_feat: torch.Tensor,
        gt=None,
        mode='test',
        msk_feat=None,
        up_embed=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = self.mask_tokens.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # src = src + dense_prompt_embeddings
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        if len(mask_feat.shape)==3:
            mask_feat = mask_feat.unsqueeze(0)
        mask_feat = self.softmax(mask_feat).flatten(start_dim=2)
        flag_resize = 0
        # if msk_feat.shape[-1] != 1024:
        #     msk_feat = msk_feat.resize_(msk_feat.shape[-3], msk_feat.shape[-2],1024)
        #     flag_resize = 1
        
        
        if gt is not None:
            gt_feat = gt.clone()

        # CMAttn noise module
        # if mode =='train':
        #     msk_feat = torch.nn.Dropout(p=0.1, inplace=True)(msk_feat)
        #     gt_feat = gt_feat.resize_(gt_feat.shape[0], h , w).int()
        #     gt_feat = gt_feat.view(gt_feat.shape[0],h*w).unsqueeze(1)
        #     gt_feat = gt_feat.repeat(1,9,1)
        #     lab, cnts = torch.unique(gt_feat, sorted=True, return_counts=True)
        #     unique = torch.stack((lab,cnts),dim=1)
        #     unique_sorted, unique_ind = torch.sort(unique,dim=0)
        #     noise_mean = torch.mean(msk_feat).cuda()
        #     for i,cnt_ind in enumerate(unique_ind[:,1]):
        #         var = noise_list[i]
        #         noise = torch.randn((msk_feat.size())) * var 
        #         noise = noise.cuda() + noise_mean
        #         msk_feat[gt_feat==lab[cnt_ind]] = msk_feat[gt_feat==lab[cnt_ind]] + noise[gt_feat==lab[cnt_ind]]
        msk_feat = self.self_attn(q=msk_feat, k=msk_feat, v=msk_feat)         
           

        msk_feat = self.norm1(msk_feat)
        
        msk_feat = self.self_attn2(q=msk_feat, k=msk_feat, v=msk_feat)
        msk_feat = msk_feat.clone()+self.mlp(msk_feat)
        msk_feat = self.norm2(msk_feat)
        print("self.num_mask_tokens:",self.num_mask_tokens)
        print("msk_feat",msk_feat.shape)
        print("tranpse",msk_feat.transpose( -2, -1).shape)
        msk_feat = self.med_sel(msk_feat.transpose( -2, -1))
        
        # if flag_resize == 1:
        #     msk_feat = msk_feat.resize_(msk_feat.shape[-3], 1024, msk_feat.shape[-1],)
        #     flag_resize = 0
        msk_feat = msk_feat.transpose(-1, -2).view(b, -1, h, w)
        msk_feat = self.softmax(msk_feat)

        # image embedding enhancement
        src = src.clone() + torch.mul(src, msk_feat)

        hs, src, attn_out= self.transformer2(src, pos_src, tokens, mask_feat)
        iou_token_out = hs[:, 0, :]
        # mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        mask_tokens_out = hs[:, 0 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        upscaled_embedding = self.skip_connect(torch.cat((upscaled_embedding,up_embed),dim=1))
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps2[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [b, c, token_num]

        b, c, h, w = upscaled_embedding.shape  # [h, token_num, h, w]
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [1, 4, 256, 256], 256 = 4 * 64, the size of image embeddings

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred, attn_out



class MLP2(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

