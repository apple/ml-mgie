from pathlib import Path

from ml_mgie.mgie import MGIE, MGIEParams
from PIL import Image
from SSIM_PIL import compare_ssim

TEST_IMAGE_PATH = Path(__file__).parents[1] / "data/0.jpg"
TEST_INSTRUCTION = "make the frame red"

params = MGIEParams()
mgie = MGIE(params=params)


def test_prepare():
    # Prepare prompt
    instruction = TEST_INSTRUCTION
    image_path = TEST_IMAGE_PATH
    assert image_path.exists()
    prompt_ids, prompt_mask = mgie.prepare_prompt_id_and_mask(instruction=instruction)
    assert instruction in mgie.tokenizer.decode(prompt_ids)

    # Prepare image
    image = Image.open(image_path).convert("RGB")
    img = mgie.prepare_img(image)
    assert img.shape == (3, 224, 224)


def test_generate():
    # Prepare inputs
    instruction = TEST_INSTRUCTION
    image_path = TEST_IMAGE_PATH
    image = Image.open(image_path).convert("RGB")
    result_image, inner_thoughts = mgie.edit(image, instruction)

    # Check inner thoughts contains basic information
    words = ["frame", "red", "glasses"]
    assert [word in inner_thoughts for word in words].count(True) == len(words)

    # Check result image is not pure hallucination, close to original
    assert compare_ssim(image, result_image, GPU=False) >= 0.8
