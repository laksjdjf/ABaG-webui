#ABaG for webui

import torch
import numpy as np

import modules.scripts as scripts
import gradio as gr

import importlib

import modules.processing
from modules.processing import StableDiffusionProcessing

import ldm.models.diffusion.ddim as ddim
from scripts.ddim import ABaGDDIMSampler
from scripts.attention_controll import AttentionController


class OrgDDIMSampler(ddim.DDIMSampler):
    pass

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "ABaG for webui"

    def ui(self, is_img2img):
        with gr.Row():
            gr.Markdown(
                """
                Set sampler to DDIM(^q^)
                """)
        with gr.Row():
            enable_abag = gr.Checkbox(label='Enable ABaG', value=False)
            attention_division = gr.Dropdown([2,4,8], label=f"set size of attention map to (h/n,w/n)", value=4)
        with gr.Row():
            bboxes = gr.Textbox(lines = 5, label="bboxes")
        with gr.Row():    
            lr = gr.Slider(0, 3, value=0.6,step=0.05,label='learning rate(?), multiplier of loss')
        with gr.Row():
            thresholds  = gr.Textbox(value = "0:0.05 10:0.5 20:0.8", label="thresholds")
        return [enable_abag, attention_division, bboxes, lr, thresholds]

    def run(self, p: StableDiffusionProcessing, enable_abag, attention_division, bboxes, lr, thresholds):
        if enable_abag:
            attention_controller = AttentionController(
                unet = p.sd_model.model.diffusion_model, 
                steps = p.steps,
                bboxes = bboxes,
                attention_height = p.height // attention_division // 8,
                attention_width = p.width // attention_division // 8,
                lr = float(lr),
                thresholds = thresholds,
            )
            setattr(ABaGDDIMSampler,"attention_controller",attention_controller)
            #DDIMSamplerを改造する
            ddim.DDIMSampler = ABaGDDIMSampler
            #モジュールのリロードで無理やり上の変更を反映
            importlib.reload(modules.processing)
            result = modules.processing.process_images(p)

            #元に戻っていることをお祈りする
            ddim.DDIMSampler = OrgDDIMSampler
            importlib.reload(modules.processing)
        else:
            result = modules.processing.process_images(p)
            
        return result
