from datetime import datetime

import gradio as gr
from ml_mgie.mgie import MGIE, MGIEParams
from PIL import Image

mgie = MGIE(params=MGIEParams(half=True))


def go_mgie(
    image: Image.Image,
    instruction: str,
    seed: int,
    cfg_txt: float,
    cfg_img: float,
    max_size: int,
):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{timestamp} processing image with instruction: {instruction}")

    params = MGIEParams(seed=seed, cfg_txt=cfg_txt, cfg_img=cfg_img, max_size=max_size)
    mgie.params = params

    image.thumbnail((params.max_size, params.max_size))
    result_image, inner_thoughts = mgie.edit(image=image, instruction=instruction)
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
                label="Generated Image", type="pil", interactive=False, image_mode="RGB"
            ),
        ]
    with gr.Row():
        instruction, inner_thoughts = [
            gr.Textbox(label="Instruction", interactive=True),
            gr.Textbox(label="Inner thoughts", interactive=False),
        ]
    with gr.Row():
        seed, cfg_txt, cfg_img, max_size = [
            gr.Number(value=42, label="Seed", interactive=True, precision=0),
            gr.Number(value=7.5, label="Text CFG", interactive=True),
            gr.Number(value=1.5, label="Image CFG", interactive=True),
            gr.Number(
                minimum=1,
                maximum=1024,
                value=512,
                precision=0,
                label="Maximum image size",
                interactive=True,
            ),
        ]
    with gr.Row():
        btn_sub = gr.Button("Submit")

    btn_sub.click(
        fn=go_mgie,
        inputs=[input_image, instruction, seed, cfg_txt, cfg_img, max_size],
        outputs=[result_image, inner_thoughts],
        concurrency_limit=1,
    )


app.queue()
app.launch(server_port=7122)
