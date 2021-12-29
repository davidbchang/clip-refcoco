import argparse
import json
import logging
import os
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm, trange
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.optim.adam
from torch.utils.data import DataLoader, SequentialSampler

import clip
import sys

sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/models')
sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/utils')
sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/dataloaders')
print(sys.path)
from clip_classifier import ClipClassifierReverse
from base_utils import computeIoU, to_device, resize_transform
from clip_refexp_datasets import ReferClipReverseDataset, ReferEvaluateClipReverseDataset

logger = logging.getLogger(__name__)

CLIP_CLASSES = ['RN101', 'RN50', 'RN50x16', 'RN50x4', 'ViT-B/16', 'ViT-B/32']
torch.cuda.empty_cache()


def get_model(model_name, device, state_dict=None):
    model, preprocess = clip.load(model_name, device)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, preprocess


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def evaluate(args, vit_model, rn_model, eval_dataset_vit, eval_dataset_rn, postfix: Optional[str] = "latest") -> Dict[str, Dict[str, float]]:
    """
    Args:
        `postfix`: name for evaluation. Usually defined by epoch and iteration.

    Returns:
        Dict[str, Dict[str, float]]: {postfix [str] : {key [str] : value [float]} }
    """
    eval_output_dir = args.eval_output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler_vit = SequentialSampler(eval_dataset_vit)
    eval_dataloader_vit = DataLoader(eval_dataset_vit, sampler=eval_sampler_vit,
                                     batch_size=args.per_gpu_eval_batch_size, drop_last=False, shuffle=False)
    eval_sampler_rn = SequentialSampler(eval_dataset_rn)
    eval_dataloader_rn = DataLoader(eval_dataset_rn, sampler=eval_sampler_rn,
                                    batch_size=args.per_gpu_eval_batch_size, drop_last=False, shuffle=False)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(postfix))
    vit_model.eval()
    rn_model.eval()
    labels = []
    preds = []
    val_correct = 0
    num_labels = 0
    predictions_list = []
    tqdm_eval_loader_vit = tqdm(eval_dataloader_vit, desc="Evaluating")
    tqdm_eval_loader_rn = tqdm(eval_dataloader_rn, desc="Evaluating")

    for i, (vit_input, rn_input) in enumerate(zip(tqdm_eval_loader_vit, tqdm_eval_loader_rn)):
        sent = vit_input.pop('sent')
        pred_boxes_img_id = vit_input.pop('bboxes_img_id')
        pred_boxes = vit_input.pop('bboxes')
        gt_box = vit_input.pop('gt_box')
        vit_input = to_device(vit_input, args.device)

        rn_sent = rn_input.pop('sent')
        rn_pred_boxes_img_id = rn_input.pop('bboxes_img_id')
        rn_pred_boxes = rn_input.pop('bboxes')
        rn_gt_box = rn_input.pop('gt_box')

        rn_input = to_device(rn_input, args.device)
        with torch.no_grad():
            vit_outputs = vit_model(**vit_input)
            rn_outputs = rn_model(**rn_input)
            ensemble_preds = [vit_outputs[0], rn_outputs[0]]
            scores = torch.mean(torch.stack(ensemble_preds), dim=0)
            assert scores.shape == vit_outputs[0].shape

        softmax_scores = torch.softmax(scores, dim=1)

        predicted = torch.argmax(scores, dim=1)
        correct = 0
        correct_flag = 0
        for j, pred_id in enumerate(predicted):
            pred_box = pred_boxes[pred_id]
            if computeIoU(pred_box, gt_box) > 0.5:
                correct += 1
                correct_flag = 1
        val_correct += correct
        num_labels += len(predicted)

        accuracy = val_correct / num_labels
        tqdm_eval_loader_vit.set_description_str(f"[Acc]: {accuracy :.4f}")

        # testing
        for j in range(len(predicted)):
            percentages = [round(100 * s, 2) for s in softmax_scores[0].tolist()]
            new_predicted = predicted.item()
            new_pred_boxes = [[k.item() for k in box] for box in pred_boxes]
            new_gt_boxes = [[k.item() for k in box] for box in gt_box]
            predictions = {
                'gt_box': new_gt_boxes,
                'predicted': new_predicted,
                'sent': sent[0],
                'pred_boxes_img_id': pred_boxes_img_id,
                'pred_boxes': new_pred_boxes,
                'percentages': percentages,
                'correct_flag': correct_flag
            }
            predictions_list.append(predictions)

    with open('../referring_expressions/data/datasets/testA_refcoco+_unc_clip_rn50x16_vitb32_predictions.json', 'w') as output_json:
        json.dump(predictions_list, output_json)

    # Calculate accuracy
    accuracy = val_correct / num_labels

    # save & update eval output
    result = {postfix: {"accuracy": accuracy}}
    output_eval_file = os.path.join(eval_output_dir, "metrics.json")
    results = {}
    if os.path.exists(output_eval_file):
        results = json.load(open(output_eval_file))  # update result file instead of overwriting it
    results.update(result)
    with open(output_eval_file, "w") as writer:
        logger.info(f"***** Eval results {postfix} *****")
        logger.info(str(result))
        writer.write(json.dumps(results))
        writer.close()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="RN50x16", type=str, choices=CLIP_CLASSES,
                        help="clip model to use.")
    parser.add_argument("--eval_file", default='../referring_expressions/data/datasets/testA_refcoco+_unc_dets.json', type=str,
                        help="Input eval file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--vit_model_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--rn_model_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--resize", action='store_true',
                        help="resize instead of center crop for preprocessing.")

    parser.add_argument("--eval_output_dir", default=None, type=str,
                        help="Directory to write results to. Defaults to output_dir")
    parser.add_argument("--tb_dir", default=None, type=str,
                        help="Directory to write tensorboard to. Defaults to output_dir")

    parser.add_argument("--split", type=str, default='val',
                        help="split to use for prediction (val/test)")
    parser.add_argument("--debug", action='store_true',
                        help="Run with smaller data size")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    print(args)
    if args.eval_output_dir is None:
        args.eval_output_dir = args.output_dir
    if args.tb_dir is None:
        args.tb_dir = args.output_dir

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print('ARGS.nGPU:', args.n_gpu)
    args.device = device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model
    clip_model_vit, preprocess_vit = get_model('ViT-B/32', device)
    clip_model_rn, preprocess_rn = get_model('RN50x16', device)
    if args.resize:
        logger.info("Using resize transform...")
        preprocess_vit = resize_transform(clip_model_vit.visual.input_resolution)
        preprocess_rn = resize_transform(clip_model_rn.visual.input_resolution)
    vit_model = ClipClassifierReverse(clip_model_vit).to(device)
    convert_models_to_fp32(vit_model)
    rn_model = ClipClassifierReverse(clip_model_rn).to(device)
    convert_models_to_fp32(rn_model)

    # Get Dataset for all splits and update config
    eval_dataset_vit = ReferEvaluateClipReverseDataset(args.eval_file, split='val', preprocess=preprocess_vit, debug=args.debug)
    eval_dataset_rn = ReferEvaluateClipReverseDataset(args.eval_file, split='val', preprocess=preprocess_rn, debug=args.debug)

    # Evaluate only
    vit_checkpoint_path = os.path.join(args.vit_model_dir, 'checkpoint-best/model.pt')
    print(vit_checkpoint_path)
    vit_checkpoint = torch.load(vit_checkpoint_path)

    print(f"Loading checkpoint from {vit_checkpoint}; Epoch: {vit_checkpoint['epoch']}")
    vit_model.load_state_dict(vit_checkpoint['state_dict'])
    print(f"Evaluating from file: {args.eval_file}")

    vit_model_to_eval = vit_model.module if hasattr(vit_model, 'module') else vit_model

    rn_checkpoint_path = os.path.join(args.rn_model_dir, 'checkpoint-best/model.pt')
    print(rn_checkpoint_path)
    rn_checkpoint = torch.load(rn_checkpoint_path)

    print(f"Loading checkpoint from {rn_checkpoint}; Epoch: {rn_checkpoint['epoch']}")
    rn_model.load_state_dict(rn_checkpoint['state_dict'])
    print(f"Evaluating from file: {args.eval_file}")

    rn_model_to_eval = rn_model.module if hasattr(rn_model, 'module') else rn_model

    result = evaluate(args, vit_model_to_eval, rn_model_to_eval, eval_dataset_vit, eval_dataset_rn, postfix="best_eval_only")
    print("Result:", result)


if __name__ == "__main__":
    main()
