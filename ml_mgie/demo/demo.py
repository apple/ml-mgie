import os
import shutil
from pathlib import Path

from ml_mgie.mgie import MGIE, MGIEParams
from PIL import Image
from tqdm import tqdm

params = MGIEParams(half=True, seed=13331, cfg_txt=7.5, cfg_img=1.5, max_size=512)
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

    result_image, inner_thought = mgie.edit(
        image=image,
        instruction=instruction,
    )
    print(f"Inner thought: {inner_thought}")
    result_image.save(output_path / f"{i}-out.jpg")
    shutil.copy(image_input_path, output_path / f"{i}-in.jpg")
