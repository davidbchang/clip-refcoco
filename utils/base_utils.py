from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
BICUBIC = Image.BICUBIC


# from https://github.com/lichengunc/refer/blob/master/pyEvalDemo.ipynb
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [d.to(device) for d in inputs]
    elif isinstance(inputs, dict):
        return {k: v.to(device) for k, v in inputs.items()}
    else:
        raise TypeError("`inputs` should be a list of dictionary.")


def resize_transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
