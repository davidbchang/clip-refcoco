import json
from typing import List
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.optim.adam
import random
from dataclasses import dataclass
import clip
sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/utils')
from base_utils import computeIoU


class ReferEvaluateClipReverseDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_file: str,
                 split: str,
                 preprocess,
                 debug=False
                 ):
        """
        Args:
            input_file: provided json file
            split: ['val', 'test'].
            preprocess: CLIP model img preprocessor
        """

        self.data = json.load(open(input_file))
        print(len(self.data))
        if debug:
            self.data = self.data[:100]
        self.split = split
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def get_vis_feats(self, image_id: str) -> torch.Tensor:
        image_prefix = '/gscratch/xlab/changd8/visual_comet/CSE490gFinal/data/datasets/bbox_imgs'
        image_path = Path(image_prefix) / image_id
        return self.preprocess(Image.open(image_path))

    def get_input_ids(self, endings: List[str] or str) -> torch.LongTensor:
        return clip.tokenize(endings)

    def preprocess_text(self, s):
        return s[:150]

    def __getitem__(self, index):
        record = self.data[index]
        sentence = record['sent']
        bboxes_img_id = record['bboxes_img_id']
        bboxes = record['bboxes']
        gt_box = record['gt_box']

        text = self.preprocess_text(sentence)
        input_ids = self.get_input_ids(text)

        vis_feats = torch.cat(tuple(self.get_vis_feats(img_id).unsqueeze(0) for img_id in bboxes_img_id), 0)

        return {
            "input_ids": input_ids,
            "vis_feats": vis_feats,
            "bboxes_img_id": bboxes_img_id,
            "bboxes": bboxes,
            "gt_box": gt_box
        }


class ReferClipReverseDataset(torch.utils.data.Dataset):

    def __init__(self,
                 input_file: str,
                 split: str,
                 preprocess,
                 debug=False
                 ):
        """
        Args:
            input_file: provided json file
            split: ['train', 'val']
            preprocess: CLIP model img preprocessor
        """

        self.data = json.load(open(input_file))
        print(len(self.data))
        if debug:
            self.data = self.data[:100]
        self.split = split
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def get_vis_feats(self, image_id: str) -> torch.Tensor:
        image_prefix = '/gscratch/xlab/changd8/visual_comet/CSE490gFinal/data/datasets/bbox_imgs'
        image_path = Path(image_prefix) / image_id
        return self.preprocess(Image.open(image_path))

    def get_input_ids(self, endings: List[str] or str) -> torch.LongTensor:
        return clip.tokenize(endings)

    def preprocess_text(self, s):
        return s[:150]

    def __getitem__(self, index):
        # use faster rcnn predicted regions
        def get_pred_boxes(boxes, img_ids, gt):
            while True:
                max_len = len(boxes) if len(boxes) < 4 else 4
                rand_pred_boxes = random.sample(list(enumerate(boxes)), max_len)
                indices, rand_pred_boxes = [[i for i, j in rand_pred_boxes], [j for i, j in rand_pred_boxes]]
                for i, box in enumerate(rand_pred_boxes):
                    if computeIoU(box, gt) > 0.5:
                        label = i
                        rand_img_ids = [img_ids[i] for i in indices]
                        return rand_pred_boxes, rand_img_ids, label

        # use COCO gt regions
        def get_gt_boxes(boxes, img_ids, gt):
            while True:
                max_len = len(boxes) if len(boxes) < 4 else 4
                rand_boxes = random.sample(list(enumerate(boxes)), max_len)
                indices, rand_boxes = [[i for i, j in rand_boxes], [j for i, j in rand_boxes]]
                for i, box in enumerate(rand_boxes):
                    if box == gt:
                        label = i
                        rand_img_ids = [img_ids[i] for i in indices]
                        return rand_boxes, rand_img_ids, label

        record = self.data[index]
        sentence = record['sent']
        bboxes_img_id = record['bboxes_img_id']
        bboxes = record['bboxes']
        gt_box = record['gt_box']
        pred_boxes, pred_boxes_img_id, label = get_gt_boxes(bboxes, bboxes_img_id, gt_box)

        text = self.preprocess_text(sentence)
        input_ids = self.get_input_ids(text)

        vis_feats = torch.cat(tuple(self.get_vis_feats(img_id).unsqueeze(0) for img_id in pred_boxes_img_id), 0)

        return {
            "input_ids": input_ids,
            "vis_feats": vis_feats,
            "bboxes_img_id": bboxes_img_id,
            "bboxes": bboxes,
            "gt_box": gt_box,
            "labels": label
        }


@dataclass
class ReferClipCollator:
    """
    Data collator that will dynamically pad the inputs for vision and language multiple choice.
    """
    def __call__(self, batch):
        bboxes_img_id_batch = []
        bboxes_batch = []
        gt_box_batch = []
        for example in batch:
            bboxes_img_id_batch.append(example['bboxes_img_id'])
            bboxes_batch.append(example['bboxes'])
            gt_box_batch.append(example['gt_box'])

        input_ids_batch = pad_sequence([e['input_ids'] for e in batch], batch_first=True)
        vis_feats_batch = pad_sequence([e['vis_feats'] for e in batch], batch_first=True)
        assert vis_feats_batch.shape[1] == 4
        labels_batch = torch.tensor([e['labels'] for e in batch])

        batch = {
            "input_ids": input_ids_batch,
            "vis_feats": vis_feats_batch,
            'bboxes_img_id': bboxes_img_id_batch,
            'bboxes': bboxes_batch,
            'gt_box': gt_box_batch,
            "labels": labels_batch
        }

        return batch



