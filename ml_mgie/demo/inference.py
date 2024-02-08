import os
import torch
import tqdm
from PIL import Image

from ml_mgie.mgie import (
    MGIE,
    MGIEParams,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from ml_mgie.utils import remove_alter, crop_resize
from ml_mgie.llava_conversation import conv_templates

SEED = 13331
mgie = MGIE(MGIEParams(device="cuda"))

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
    img, txt = Image.open("_input/%d.jpg" % (i)).convert("RGB"), ins[i]

    img = mgie.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    txt = "what will this image be like if '%s'" % (txt)
    txt = (
        txt
        + "\n"
        + DEFAULT_IM_START_TOKEN
        + DEFAULT_IMAGE_PATCH_TOKEN * mgie.image_token_len
        + DEFAULT_IM_END_TOKEN
    )
    conv = conv_templates["vicuna_v1_1"].copy()
    conv.append_message(conv.roles[0], txt), conv.append_message(conv.roles[1], None)
    txt = conv.get_prompt()
    txt = mgie.tokenizer(txt)
    txt, mask = torch.as_tensor(txt["input_ids"]), T.as_tensor(txt["attention_mask"])

    with torch.inference_mode():
        out = mgie.model.generate(
            txt.unsqueeze(dim=0).cuda(),
            images=img.half().unsqueeze(dim=0).cuda(),
            attention_mask=mask.unsqueeze(dim=0).cuda(),
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
        res = mgie.pipe(
            image=Image.open("_input/%d.jpg" % (i)).convert("RGB"),
            prompt_embeds=emb,
            negative_prompt_embeds=mgie.null,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
        ).images[0]

    input = Image.open("_input/%d.jpg" % (i)).convert("RGB")
    os.makedirs("_output", exist_ok=True)
    input.save("_output/in-%d.jpg" % (i))
    Image.fromarray(res).save("_output/out-%d.jpg" % (i))
