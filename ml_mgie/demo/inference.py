import os
import shutil
from pathlib import Path

import torch
from ml_mgie.llava_conversation import conv_templates
from ml_mgie.mgie import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    MGIE,
    MGIEParams,
)
from ml_mgie.utils import crop_resize, remove_alter
from PIL import Image
from tqdm import tqdm

SEED = 13331
CFG_TXT = 7.5
CFG_IMG = 1.5
params = MGIEParams()
mgie = MGIE(params=params)
input_path = Path("_input")
output_path = Path("_output")
os.makedirs(output_path, exist_ok=True)

ins = [
    "make the frame red",
    "turn the day into night",
    "give him a beard",
    "make cottage a mansion",
    "remove yellow object from dogs paws",
    "change the hair from red to blue",
    "remove the text",
    "increase the image contrast",
    "remove the people in the background",
    "please make this photo professional looking",
    "darken the image, sharpen it",
    "photoshop the girl out",
    "make more brightness",
    "take away the brown filter form the image",
    "add more contrast to simulate more light",
    "dark on rgb",
    "make the face happy",
    "change view as ocean",
    "replace basketball with soccer ball",
    "let the floor be made of wood",
]
for i in tqdm(range(len(ins))):
    image_input_path = input_path / f"{i}.jpg"
    img = crop_resize(Image.open(image_input_path).convert("RGB"))
    instruction = ins[i]

    img = mgie.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    prompt = f"what will this image be like if {instruction}"
    prompt = (
        prompt
        + "\n"
        + DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * mgie.image_token_len
        + DEFAULT_IM_END_TOKEN
    )
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt_tokenized = mgie.tokenizer(prompt)
    prompt_tensor_ids = torch.as_tensor(prompt_tokenized["input_ids"])
    mask = torch.as_tensor(prompt_tokenized["attention_mask"])

    with torch.inference_mode():
        out = mgie.model.generate(
            prompt_tensor_ids.unsqueeze(dim=0).to(params.device),
            images=img.half().unsqueeze(dim=0).to(params.device),
            attention_mask=mask.unsqueeze(dim=0).to(params.device),
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

        p = min(out.index(32003) - 1 if 32003 in out else len(hid) - 9, len(hid) - 9)
        hid = hid[p : p + 8]

        out = remove_alter(mgie.tokenizer.decode(out))
        emb = mgie.model.edit_head(hid.unsqueeze(dim=0), mgie.emb)
        res: Image.Image = mgie.pipe(
            image=Image.open(image_input_path).convert("RGB"),
            prompt_embeds=emb,
            negative_prompt_embeds=mgie.null,
            generator=torch.Generator(device=params.device).manual_seed(SEED),
            guidance_scale=CFG_TXT,
            image_guidance_scale=CFG_IMG,
        ).images[0]
    # Save results before/after
    print(f"Instruction: {instruction}")
    print(f"Output: {out}")
    shutil.copy(image_input_path, output_path / f"{i}-in.jpg")
    res.save(output_path / f"{i}-out.jpg")
