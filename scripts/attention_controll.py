from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention, default, exists
from ldm.modules.attention import SpatialTransformer
import itertools
from typing import List, Optional, Union, Tuple
import functools

import copy

from scripts.gaussian_smoothing import GaussianSmoothing

import torch
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat

import numpy as np

class AttentionController:
    def __init__(self, unet, steps:int, bboxes:list, attention_height:int = 16, attention_width:int = 16,lr:float = 0.6):
        '''
        unet:unet
        '''
        self.unet = unet
        self.steps = steps
        self.scale_range = np.linspace(1., 0.5, self.steps)
        self._hook_blocks()
        self.attention_maps = None
        self.attention_height = attention_height
        self.attention_width = attention_width
        self.attention_pixels = attention_height * attention_width

        self.lr = lr #学習してないだろ
        self.bboxes = [[int(s) for s in box.split(" ")] for box in bboxes.split("\n")]
        self.indices_to_alter = [box[0] for box in  self.bboxes]
        self.max_indices = max(self.indices_to_alter)
        print(self.bboxes)
        ##よく分からないものたち gaussian_smoothing関連だと思うけど
        self.sigma = 0.5
        self.kernel_size = 3
        self.smooth_attentions = False
        
        self.scale_factor = 20
        self.thresholds = {0: 0.05, 10: 0.5, 20: 0.8}
        self.max_iter = 25
        
    #CrossAttention(attn2)を全て探す
    def _hook_blocks(self):
        for unet_block in itertools.chain(self.unet.input_blocks, self.unet.output_blocks, [self.unet.middle_block]):
            # if 'CrossAttn' in unet_block.__class__.__name__:
            for module in unet_block.modules():
                if type(module) is SpatialTransformer:
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        new_forward = self._hook_forward_create()
                        basic_transformer_block.attn2.forward = new_forward.__get__(basic_transformer_block.attn2, CrossAttention)
    
    def _reset(self):
        self.attention_maps = None

    def _get_attention_map(self, sim:torch.Tensor):
        if sim.shape[1] != self.attention_pixels:
            return
        attention_map = sim.reshape(1,-1,self.attention_height,self.attention_width,sim.shape[-1])
        if self.attention_maps is None:
            self.attention_maps = attention_map
        else:
            self.attention_maps = torch.cat([self.attention_maps,attention_map])

    def _get_avarage_attention(self):
        return (self.attention_maps.sum(0) / self.attention_maps.shape[0]).mean(0)

    def _compute_max_attention_per_bbox_and_outofbbox(self,attention_map_mean) -> List[torch.Tensor]:
        print(attention_map_mean.shape)

        #webuiは勝手に77までパディングするので注意するトークンまでに限定
        attention_for_text = attention_map_mean[:, :, 1:self.max_indices+1]
        print(attention_for_text.shape)
        attention_for_text = attention_for_text * 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in self.indices_to_alter]

        # Extract the maximum values of bbox and out of bboxes
        bbox_max_indices_list = []
        out_of_bbox_max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if self.smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=self.kernel_size, sigma=self.sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            
            index = [bbox[0]-1 for bbox in self.bboxes].index(i)
            bbox = self.bboxes[index]
            
            # bbox内の大域的な特徴を取得するために平均値を使用しています。
            tmp_image = image.clone()
            bbox_max_indices_list.append(
                tmp_image[bbox[1]:bbox[2], bbox[3]:bbox[4]].mean()
            )

            # bbox外の最大値
            tmp_image = image.clone()
            tmp_image[bbox[1]:bbox[2], bbox[3]:bbox[4]] = 0
            out_of_bbox_max_indices_list.append(tmp_image.max())

        return bbox_max_indices_list, out_of_bbox_max_indices_list

    def _compute_loss(
            self,
            bbox_max_indices_list: List[torch.Tensor],
            out_of_bbox_max_indices_list: List[torch.Tensor],
            return_loss: bool = False,
            lr: float = 0.6) -> torch.Tensor:
            
        in_bbox_losses = [max(0, 1. - curr_max) for curr_max in bbox_max_indices_list]
        out_bbox_losses = out_of_bbox_max_indices_list

        loss = (max(in_bbox_losses) + max(out_bbox_losses)) * lr

        if return_loss:
            return loss, in_bbox_losses, out_bbox_losses
        else:
            return loss

    def _update_latent(self,latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), latents, retain_graph=True)[0]
        print(loss, grad_cond.mean(),step_size)
        latents = latents - step_size * grad_cond
        return latents

    
    
    #attention mapを保存するようhookする
    def _hook_forward_create(self):
        def _hook_forward(_self, x:torch.Tensor, context:torch.Tensor=None, mask=None):
            h = _self.heads
            q = _self.to_q(x)
            context = default(context, x)
            k = _self.to_k(context)
            v = _self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            # force cast to fp32 to avoid overflowing
            # 応急処置ｗ
            if False:
                with torch.autocast(enabled=False, device_type = 'cuda'):
                    q, k = q.float(), k.float()
                    sim = einsum('b i d, b j d -> b i j', q, k) * _self.scale
            else:
                sim = einsum('b i d, b j d -> b i j', q, k) * _self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            sim = sim.softmax(dim=-1)
            
            ######変換部分##########################################
            # save attention map
            self._get_attention_map(sim)
            ########################################################

            out = einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return _self.to_out(out)
        return _hook_forward
