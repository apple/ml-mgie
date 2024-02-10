import os
from datetime import datetime
from pathlib import Path

import gradio as gr
from ml_mgie.mgie import MGIE, MGIEParams
from PIL import Image

DEBUG_PATH = Path("debug")
os.makedirs(DEBUG_PATH, exist_ok=True)
mgie = MGIE(params=MGIEParams())


def go_mgie(
    image: Image.Image,
    instruction: str,
    seed: int,
    cfg_txt: float,
    cfg_img: float,
    max_size: int,
    telemetry: bool,
):
    params = MGIEParams(seed=seed, cfg_txt=cfg_txt, cfg_img=cfg_img, max_size=max_size)
    mgie.params = params
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if telemetry:
        image.save(DEBUG_PATH / f"{name}-in.jpg")

    image.thumbnail((params.max_size, params.max_size))
    if telemetry:
        image.save(DEBUG_PATH / f"{name}-thumb.jpg")

    result_image, inner_thoughts = mgie.edit(image=image, instruction=instruction)
    if telemetry:
        result_image.save(DEBUG_PATH / f"{name}-zout.jpg")
    return result_image, inner_thoughts


with gr.Blocks() as app:
    gr.Markdown(
        "# Guiding Instruction-based Image Editing via Multimodal Large Language Models"
    )
    with gr.Row():
        input_image, result_image = [
            gr.Image(
                label="Input Image",
                interactive=True,
                type="pil",
                image_mode="RGB",
            ),
            gr.Image(
                label="Goal Image", type="pil", interactive=False, image_mode="RGB"
            ),
        ]
    with gr.Row():
        instruction, inner_thoughts = [
            gr.Textbox(label="Instruction", interactive=True),
            gr.Textbox(label="Expressive Instruction", interactive=False),
        ]
    with gr.Row():
        telemetry, seed, cfg_txt, cfg_img, max_size = [
            gr.Checkbox(label="Telemetry", value=True, interactive=True),
            gr.Number(value=42, label="Seed", interactive=True, precision=0),
            gr.Number(value=7.5, label="Text CFG", interactive=True),
            gr.Number(value=1.5, label="Image CFG", interactive=True),
            gr.Number(
                minimum=1,
                maximum=1024,
                value=512,
                precision=0,
                label="Maximum Size",
                interactive=True,
            ),
        ]
    with gr.Row():
        btn_sub = gr.Button("Submit")

    btn_sub.click(
        fn=go_mgie,
        inputs=[input_image, instruction, seed, cfg_txt, cfg_img, max_size, telemetry],
        outputs=[result_image, inner_thoughts],
        concurrency_limit=1,
    )


app.queue()
app.launch(server_port=7122)
