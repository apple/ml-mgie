from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import diffusers
import torch
import transformers
from PIL import Image

from .base import DEFAULT_DEVICE

# from .mgie_llava import LlavaLlamaForCausalLM
from .llava import LlavaLlamaForCausalLM
from .llava_conversation import conv_templates
from .utils import crop_resize

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
        self.pipe: diffusers.StableDiffusionInstructPix2PixPipeline = None
        self._set_model()

    def _set_model(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(PATH_LLAVA)
        model = LlavaLlamaForCausalLM.from_pretrained(
            PATH_LLAVA.absolute(),
            # low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            use_cache=True,
        ).to(self.params.device)
        model.model = model.model.to(self.params.device)
        model.model.vision_tower[0] = model.get_vision_tower().to(self.params.device)
        model.lm_head = model.lm_head.to(self.params.device)
        model.edit_head = model.edit_head.to(self.params.device)

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
        # ckpt = torch.load(PATH_MLLM, map_location=self.params.device)  # TO DEVICE?
        ckpt = torch.load(PATH_MLLM, map_location="cpu")
        # incompatible_keys = model.load_state_dict(ckpt, strict=False, assign=True)
        incompatible_keys = model.load_state_dict(ckpt, strict=False)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )

        vision_tower = model.get_vision_tower()
        vision_tower: transformers.CLIPVisionModel = (
            transformers.CLIPVisionModel.from_pretrained(
                vision_tower.config._name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.params.device)
        )
        model.model.vision_tower[0] = vision_tower
        vision_config: transformers.PretrainedConfig = vision_tower.config
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
        _ = model.eval()
        emb = ckpt["emb"].to(self.params.device)
        with torch.inference_mode():
            null = model.edit_head(
                torch.zeros(1, 8, 4096).half().to(self.params.device),
                emb,
            )

        pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,  # , safety_checker=None
        ).to(self.params.device)
        pipe.set_progress_bar_config(disable=True)
        """pipe.unet.load_state_dict(
            torch.load(PATH_UNET.absolute(), map_location=self.params.device),
            assign=True,
            strict=True,
        )  # TO DEVICE?"""
        pipe.unet.load_state_dict(
            torch.load(PATH_UNET.absolute(), map_location="cpu"),
        )  # TO DEVICE?

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_len = image_token_len
        self.emb = emb
        self.null = null
        self.pipe = pipe

    def prepare_img(self, image: Image.Image) -> torch.Tensor:
        """image: PIL.Image.Image, Pillow RGB image"""
        img = crop_resize(image)
        img = self.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][
            0
        ]
        return img

    def prepare_prompt_id_and_mask(
        self, instruction: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt = f"what will this image be like if {instruction}"
        prompt = (
            prompt
            + "\n"
            + DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
            + DEFAULT_IM_END_TOKEN
        )
        conv = conv_templates["vicuna_v1_1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_tokenized = self.tokenizer(prompt)
        prompt_tensor_ids = torch.as_tensor(prompt_tokenized["input_ids"])
        mask = torch.as_tensor(prompt_tokenized["attention_mask"])
        return prompt_tensor_ids, mask
