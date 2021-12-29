import json
from pathlib import Path
import os
from matplotlib import colors as mcolors
from collections import defaultdict
import cv2
import opencv_draw_tools as cv2_tools
from pprint import pprint
from refer import REFER


COCO_IMAGE_DIR = './data/images/'
BBOX_DIR = './data/datasets/bbox_imgs'

colors = ['limegreen', 'red', 'blue', 'yellow']
colors = [(255 * mcolors.to_rgba(color)[2], 255 * mcolors.to_rgba(color)[1], 255 * mcolors.to_rgba(color)[0])
          for color in colors]


# get detected boxes for each img id
def img_to_pred_bboxes(dataset, splitBy):
    path = './detections/{}_{}/res101_coco_minus_refer_notime_dets.json'.format(dataset, splitBy)
    path = Path(path)
    with open(path, 'rb') as f:
        pred_data = json.load(f)

    img_to_preds_dict = defaultdict(list)
    for i, example in enumerate(pred_data):
        img_to_preds_dict[example['image_id']].append(example['box'])

    return img_to_preds_dict


# from https://github.com/lichengunc/refer/blob/master/pyEvalDemo.ipynb
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union


# create training data using COCO gt boxes
def create_ref_data_from_gt(data_root, dataset, splitBy, split, output_file):
    refer = REFER(data_root, dataset=dataset, splitBy=splitBy)
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids)
    print(len(img_ids))  # 3811 val refs

    mc_data = []
    for i, img_id in enumerate(img_ids):
        if i % 1000 == 0:
            print(output_file, 'Ref:', i, len(img_ids))

        anns = refer.imgToAnns[img_id]
        num_boxes = len(anns)
        if num_boxes < 2:
            continue

        image_path = COCO_IMAGE_DIR + 'train2014/' + 'COCO_train2014_' + str(img_id).zfill(12) + '.jpg'
        img = cv2.imread(image_path)
        bboxes_img_id = []
        bboxes = []
        for ann in anns:
            bbox = ann['bbox']
            bboxes.append(bbox)
            x1, y1, w, h = [int(k) for k in bbox]

            img_id_new = 'COCO_train2014_' + str(img_id).zfill(12) + '_' + str(ann['id']) + '.jpg'
            # if os.path.isfile(os.path.join(BBOX_DIR, img_id_new)):
            #     continue

            image_boxes = img.copy()
            image_boxes = cv2_tools.select_zone(image_boxes, (x1, y1, x1 + w, y1 + h), alpha=0.5, color=colors[0],
                                                thickness=2,
                                                filled=True)
            image_boxes = cv2.rectangle(image_boxes, (x1, y1), (x1 + w, y1 + h), color=colors[0], thickness=2)
            cv2.imwrite(os.path.join(BBOX_DIR, img_id_new), image_boxes)
            bboxes_img_id.append(img_id_new)

        refs = refer.imgToRefs[img_id]
        for ref in refs:
            gt_box = refer.getRefBox(ref['ref_id'])

            for sentence in ref['sentences']:
                mc_dict = {
                    'sent': sentence['sent'],
                    'bboxes': bboxes,
                    'bboxes_img_id': bboxes_img_id,
                    'gt_box': gt_box
                }
                mc_data.append(mc_dict)

    with open(output_file, 'w') as output_json:
        json.dump(mc_data, output_json)

    print('len:', len(mc_data))


# create training data from detected boxes
def create_ref_data(data_root, dataset, splitBy, split, output_file):
    refer = REFER(data_root, dataset=dataset, splitBy=splitBy)
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids)
    print(len(img_ids))  # 3811 val refs

    img_to_preds_dict = img_to_pred_bboxes(dataset, splitBy)

    mc_data = []
    for i, img_id in enumerate(img_ids):
        if i % 1000 == 0:
            print(output_file, 'Ref:', i, len(img_ids))

        region_proposals = img_to_preds_dict[img_id]
        if len(region_proposals) < 2:
            continue

        image_path = COCO_IMAGE_DIR + 'train2014/' + 'COCO_train2014_' + str(img_id).zfill(12) + '.jpg'
        img = cv2.imread(image_path)
        pred_bboxes = []
        for j, pred_box in enumerate(region_proposals):
            x1, y1, w, h = [int(k) for k in pred_box]

            img_id_new = 'COCO_train2014_' + str(img_id).zfill(12) + '_' + str(j) + '.jpg'
            # if os.path.isfile(os.path.join(BBOX_DIR, img_id_new)):
            #     continue

            image_boxes = img.copy()
            image_boxes = cv2_tools.select_zone(image_boxes, (x1, y1, x1 + w, y1 + h), alpha=0.5, color=colors[0],
                                                thickness=2,
                                                filled=True)
            image_boxes = cv2.rectangle(image_boxes, (x1, y1), (x1 + w, y1 + h), color=colors[0], thickness=2)
            cv2.imwrite(os.path.join(BBOX_DIR, img_id_new), image_boxes)
            pred_bboxes.append(img_id_new)

        refs = refer.imgToRefs[img_id]
        for ref in refs:
            gt_box = refer.getRefBox(ref['ref_id'])

            num_targets = 0
            for pred_box in region_proposals:
                if computeIoU(pred_box, gt_box) > 0.5:
                    num_targets += 1
            if num_targets != 1:
                continue

            for sentence in ref['sentences']:
                mc_dict = {
                    'sent': sentence['sent'],
                    'bboxes': region_proposals,
                    'bboxes_img_id': pred_bboxes,
                    'gt_box': gt_box
                }
                mc_data.append(mc_dict)

    with open(output_file, 'w') as output_json:
        json.dump(mc_data, output_json)

    print('len:', len(mc_data))


def create_evaluation_data(data_root, dataset, splitBy, split, output_file):
    refer = REFER(data_root, dataset=dataset, splitBy=splitBy)
    ref_ids = refer.getRefIds(split=split)
    img_ids = refer.getImgIds(ref_ids)
    print(len(img_ids))  # 3811 val refs

    img_to_preds_dict = img_to_pred_bboxes(dataset, splitBy)

    mc_data = []
    for i, img_id in enumerate(img_ids):
        if i % 100 == 0:
            print(output_file, 'Ref:', i, len(img_ids))

        region_proposals = img_to_preds_dict[img_id]
        if len(region_proposals) < 2:
            continue

        image_path = COCO_IMAGE_DIR + 'train2014/' + 'COCO_train2014_' + str(img_id).zfill(12) + '.jpg'
        img = cv2.imread(image_path)
        pred_bboxes = []
        for j, pred_box in enumerate(region_proposals):
            x1, y1, w, h = [int(k) for k in pred_box]

            img_id_new = 'COCO_train2014_' + str(img_id).zfill(12) + '_' + str(j) + '.jpg'
            # if os.path.isfile(os.path.join(BBOX_DIR, img_id_new)):
            #     continue

            image_boxes = img.copy()
            image_boxes = cv2_tools.select_zone(image_boxes, (x1, y1, x1 + w, y1 + h), alpha=0.5, color=colors[0],
                                                thickness=2,
                                                filled=True)
            image_boxes = cv2.rectangle(image_boxes, (x1, y1), (x1 + w, y1 + h), color=colors[0], thickness=2)
            cv2.imwrite(os.path.join(BBOX_DIR, img_id_new), image_boxes)
            pred_bboxes.append(img_id_new)

        refs = refer.imgToRefs[img_id]
        for ref in refs:
            gt_box = refer.getRefBox(ref['ref_id'])

            for sentence in ref['sentences']:
                mc_dict = {
                    'sent': sentence['sent'],
                    'bboxes': region_proposals,
                    'bboxes_img_id': pred_bboxes,
                    'gt_box': gt_box
                }
                mc_data.append(mc_dict)

    with open(output_file, 'w') as output_json:
        json.dump(mc_data, output_json)

    print('len:', len(mc_data))


if __name__ == '__main__':
    # create_ref_data_from_gt('./data', 'refcoco', 'unc', 'train', './data/datasets/train_refcoco_unc.json')
    # create_evaluation_data('./data', 'refcoco', 'unc', 'testA', './data/datasets/testA_refcoco_unc_dets.json')
    # create_ref_data_from_gt('./data', 'refcoco+', 'unc', 'train', './data/datasets/train_refcoco+_unc.json')
    create_evaluation_data('./data', 'refcoco+', 'unc', 'testA', './data/datasets/testA_refcoco+_unc_dets.json')



