import os
import shutil
from pathlib import Path

import torch
from ml_mgie.mgie import MGIE, MGIEParams
from ml_mgie.utils import remove_alter
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
    image = Image.open(image_input_path).convert("RGB")
    instruction = ins[i]

    # Prepare inputs
    img = mgie.prepare_img(image)
    prompt_tensor_ids, mask = mgie.prepare_prompt_id_and_mask(instruction)
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
        import pdb

        pdb.set_trace()
        # Here out is nonesense: "Pres flash togful calledgot At commitilli split sent"
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
