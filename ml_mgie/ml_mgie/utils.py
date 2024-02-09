from PIL import Image


def crop_resize(image: Image.Image, size: int = 512) -> Image.Image:
    w, h = image.size
    if w > h:
        p = (w - h) // 2
        image = image.crop([p, 0, p + h, h])
    elif h > w:
        p = (h - w) // 2
        image = image.crop([0, p, w, p + w])
    image = image.resize([size, size])
    return image


def remove_alter(prompt: str) -> str:  # hack expressive instruction
    if "ASSISTANT:" in prompt:
        prompt = prompt[prompt.index("ASSISTANT:") + 10 :].strip()
    if "</s>" in prompt:
        prompt = prompt[: prompt.index("</s>")].strip()
    if "alternative" in prompt.lower():
        prompt = prompt[: prompt.lower().index("alternative")]
    if "[IMG0]" in prompt:
        prompt = prompt[: prompt.index("[IMG0]")]
    prompt = ".".join([s.strip() for s in prompt.split(".")[:2]])
    if prompt[-1] != ".":
        prompt += "."
    return prompt.strip()
