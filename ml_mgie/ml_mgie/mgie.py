from dataclasses import dataclass
from pathlib import Path

import diffusers
import torch
import transformers

from .base import DEFAULT_DEVICE
from .mgie_llava import LlavaLlamaForCausalLM

# from llava.conversation import conv_templates
# from llava.model import *


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
PATH_LLAVA = Path("./_ckpt/LLaVA-7B-v1")
PATH_MLLM = Path("./_ckpt/mgie_7b/mllm.pt")
PATH_UNET = Path("./_ckpt/mgie_7b/unet.pt")

assert PATH_LLAVA.exists()
assert PATH_MLLM.exists()
assert PATH_UNET.exists()


@dataclass
class MGIEParams:
    device: torch.device = DEFAULT_DEVICE


class MGIE:
    def __init__(self, params: MGIEParams = MGIEParams()) -> None:
        self.params = params
        self.tokenizer: transformers.AutoTokenizer = None
        self.model: LlavaLlamaForCausalLM = None
        self.image_processor: transformers.CLIPImageProcessor = None
        self.image_token_len: int = None
        self.emb: torch.Tensor = None
        self._set_model()
        self.pipe = self._get_pipe()

    def _set_model(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
        model = LlavaLlamaForCausalLM.from_pretrained(
            PATH_LLAVA.absolute(),
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).to(self.params.device)
        image_processor = transformers.CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=torch.float16
        )

        tokenizer.padding_side = "left"
        tokenizer.add_tokens(
            [
                "[IMG0]",
                "[IMG1]",
                "[IMG2]",
                "[IMG3]",
                "[IMG4]",
                "[IMG5]",
                "[IMG6]",
                "[IMG7]",
            ],
            special_tokens=True,
        )
        model.resize_token_embeddings(len(tokenizer))
        ckpt = torch.load(PATH_MLLM, map_location="cpu")  # TO DEVICE?
        model.load_state_dict(ckpt, strict=False)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )

        vision_tower = model.get_model().vision_tower[0]
        vision_tower = transformers.CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.params.device)
        model.get_model().vision_tower[0] = vision_tower
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        # model = model.to(self.params.device)
        model.eval()
        emb = ckpt["emb"].to(self.params.device)
        with torch.inference_mode():
            null = model.edit_head(
                torch.zeros(1, 8, 4096, device=self.params.device, dtype=torch.float16),
                emb,
            )
        print("NULL:", null.shape)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_len = image_token_len
        self.emb = emb
        self.null = null

    def _get_pipe(self) -> diffusers.StableDiffusionInstructPix2PixPipeline:
        pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,  # , safety_checker=None
        ).to(self.params.device)
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.load_state_dict(
            torch.load(PATH_UNET, map_location="cpu")
        )  # TO DEVICE?
        return pipe
