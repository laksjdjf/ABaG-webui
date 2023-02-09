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
        return [ebable_abag]
    
    
    def run(self, p: StableDiffusionProcessing, enable_abag):
        if enable_abag:
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
