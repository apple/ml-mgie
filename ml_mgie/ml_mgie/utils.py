# TODO: add typing
def crop_resize(f, sz=512):
    w, h = f.size
    if w > h:
        p = (w - h) // 2
        f = f.crop([p, 0, p + h, h])
    elif h > w:
        p = (h - w) // 2
        f = f.crop([0, p, w, p + w])
    f = f.resize([sz, sz])
    return f


def remove_alter(s):  # hack expressive instruction
    if "ASSISTANT:" in s:
        s = s[s.index("ASSISTANT:") + 10 :].strip()
    if "</s>" in s:
        s = s[: s.index("</s>")].strip()
    if "alternative" in s.lower():
        s = s[: s.lower().index("alternative")]
    if "[IMG0]" in s:
        s = s[: s.index("[IMG0]")]
    s = ".".join([s.strip() for s in s.split(".")[:2]])
    if s[-1] != ".":
        s += "."
    return s.strip()
