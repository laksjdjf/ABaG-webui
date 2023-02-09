from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.modules.attention import CrossAttention, default, exists
from ldm.modules.attention import SpatialTransformer

##attention.py#################
import torch
from torch import einsum
from einops import rearrange, repeat
#################################

class AttentionController:
    def __init__(self, unet:UNetModel, attention_division:int = 4):
        '''
        unet:unet
        attention_division: Set size of attention map to (h/n, w/n) 
        '''
        self.unet = unet
        self.attention_maps = []
        self.attention_division = attention_division
        self.attention_pixels = None   
        self._hook_blocks()
        
    #CrossAttention(attn2)を全て探す
    def _hook_blocks(self):
        for unet_block in itertools.chain(self.unet.input_blocks, self.unet.output_blocks, [self.unet.middle_block]):
            # if 'CrossAttn' in unet_block.__class__.__name__:
            for module in unet_block.modules():
                if type(module) is SpatialTransformer:
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        basic_transformer_block.attn2.forward = self._hook_forward #forward書き換え

        return blocks
    
    def reset(self):
        self.attention_maps = []
    
    #attention mapを保存するようhookする
    def _hook_forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        
        ######変換部分##########################################
        # save attention map
        if sim.shape[1] * sim.shape[2] == self.attention_pixels: 
            self.attention_maps.append(sim)
        ########################################################

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
