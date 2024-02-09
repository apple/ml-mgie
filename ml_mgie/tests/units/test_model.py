import torch
from pathlib import Path
from PIL import Image
from ml_mgie.mgie import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    MGIE,
    MGIEParams,
)
from ml_mgie.utils import remove_alter


TEST_IMAGE_PATH = Path(__file__).parents[1] / "data/0.jpg"
TEST_INSTRUCTION = "make the frame red"
assert TEST_IMAGE_PATH.exists()
params = MGIEParams()
mgie = MGIE(params=params)


def test_prepare():
    # Prepare prompt
    instruction = TEST_INSTRUCTION
    image_path = TEST_IMAGE_PATH
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
        out = out["sequences"][0].tolist()
        out = remove_alter(mgie.tokenizer.decode(out))
        assert not "Pres flash togful calledgot" in out, f"Nonesense: {out}"
