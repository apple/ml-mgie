from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import diffusers
import torch
import transformers
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image

from .base import DEFAULT_DEVICE
from .llava_conversation import conv_templates
from .mgie_llava import LlavaLlamaForCausalLM
from .utils import crop_resize, remove_alter

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


@dataclass
class MGIEParams:
    device: torch.device = DEFAULT_DEVICE
    half: bool = False
    models_path: Path = Path("./data")
    seed: int = 13331
    cfg_txt: float = 7.5
    cfg_img: float = 1.5
    max_size: int = 512

    @property
    def dtype(self) -> torch.dtype:
        if self.half:
            return torch.float16
        return torch.float32

    @property
    def mllm_path(self) -> Path:
        assert (path := self.models_path / "mgie_7b/mllm.pt").exists()
        return path

    @property
    def unet_path(self) -> Path:
        assert (path := self.models_path / "mgie_7b/unet.pt").exists()
        return path

    @property
    def llava_path(self) -> Path:
        assert (path := self.models_path / "LLaVA-7B-v1").exists()
        return path


class MGIE:
    def __init__(self, params: MGIEParams = MGIEParams()) -> None:
        self.params = params
        self.tokenizer: transformers.AutoTokenizer = None
        self.model: LlavaLlamaForCausalLM = None
        self.image_processor: transformers.CLIPImageProcessor = None
        self.image_token_len: int = None
        self.emb: torch.Tensor = None
        self.pipe: StableDiffusionInstructPix2PixPipeline = None
        self._set_model()

    def _set_model(self):
        transformers.logging.set_verbosity_error()
        # Prepare llava
        model = LlavaLlamaForCausalLM.from_pretrained(
            self.params.llava_path,
            torch_dtype=self.params.dtype,
            use_cache=True,
        ).to(self.params.device)

        # Prepare CLIP
        image_processor = transformers.CLIPImageProcessor.from_pretrained(
            model.config.mm_vision_tower, torch_dtype=self.params.dtype
        )

        # Prepare tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.params.llava_path)
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
        ckpt = torch.load(self.params.mllm_path, map_location="cpu")
        incompatible_keys = model.load_state_dict(ckpt, strict=False)
        transformers.logging.set_verbosity_warning()

        # Patch model
        model.model.vision_tower[0] = model.get_vision_tower().to(
            self.params.device, dtype=self.params.dtype
        )
        model.lm_head = model.lm_head.to(self.params.device, dtype=self.params.dtype)
        model.edit_head = model.edit_head.to(
            self.params.device, dtype=self.params.dtype
        )
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )

        # Patch CLIP
        vision_tower = model.get_vision_tower()
        vision_tower = transformers.CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path,
            torch_dtype=self.params.dtype,
            low_cpu_mem_usage=True,
        ).to(self.params.device)
        model.model.vision_tower[0] = vision_tower
        # Patch CLIP config
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

        # Prepare placeholders
        emb = ckpt["emb"].to(self.params.device, dtype=self.params.dtype)
        with torch.inference_mode():
            null = model.edit_head(
                torch.zeros(1, 8, 4096).to(self.params.device, dtype=self.params.dtype),
                emb,
            )

        # Prepare diffusion pipeline
        pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix", safety_checker=None
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.load_state_dict(
            torch.load(self.params.unet_path.absolute(), map_location="cpu"),
            strict=True,
        )
        pipe = pipe.to(device=self.params.device, dtype=self.params.dtype)

        # Set attributes
        model.eval()
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
        img_tensor: torch.Tensor = self.image_processor.preprocess(
            img, return_tensors="pt"
        )["pixel_values"][0]
        return img_tensor.to(device=self.params.device, dtype=self.params.dtype)

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
        prompt_tensor_ids = torch.as_tensor(
            prompt_tokenized["input_ids"],
            device=self.params.device,
        )
        mask = torch.as_tensor(
            prompt_tokenized["attention_mask"],
            device=self.params.device,
        )
        return prompt_tensor_ids, mask

    def edit(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, str]:
        """
        image: PIL.Image.Image, Pillow RGB image
        instruction: str, edition to perform on image
        """
        # Prepare inputs
        if self.params.max_size:
            image.thumbnail((self.params.max_size, self.params.max_size))
        img = self.prepare_img(image)
        prompt_tensor_ids, mask = self.prepare_prompt_id_and_mask(instruction)
        with torch.inference_mode():
            out = self.model.generate(
                prompt_tensor_ids.unsqueeze(dim=0),
                images=img.unsqueeze(dim=0),
                attention_mask=mask.unsqueeze(dim=0),
                do_sample=False,
                max_new_tokens=96,
                num_beams=1,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_hidden_states=True,
            )

            out, hid = (
                out["sequences"][0].tolist(),
                torch.cat([x[-1] for x in out["hidden_states"]], dim=1)[0],
            )
            p = out.index(32003) - 1 if 32003 in out else len(hid) - 9
            p = min(p, len(hid) - 9)
            hid = hid[p : p + 8]

            inner_thoughts = remove_alter(self.tokenizer.decode(out))
            emb = self.model.edit_head(hid.unsqueeze(dim=0), self.emb)
            result_image: Image.Image = self.pipe(
                image=image,
                prompt_embeds=emb,
                negative_prompt_embeds=self.null,
                generator=torch.Generator(device=self.params.device).manual_seed(
                    self.params.seed
                ),
                guidance_scale=self.params.cfg_txt,
                image_guidance_scale=self.params.cfg_img,
            ).images[0]
        return result_image, inner_thoughts
