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


def evaluate(args, ref_model, dataset, postfix: Optional[str] = "latest") -> Dict[str, Dict[str, float]]:
    """
    Args:
        `postfix`: name for evaluation. Usually defined by epoch and iteration.

    Returns:
        Dict[str, Dict[str, float]]: {postfix [str] : {key [str] : value [float]} }
    """
    eval_output_dir = args.eval_output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, num_workers=8,
                                 batch_size=args.per_gpu_eval_batch_size, drop_last=False,
                                 shuffle=False)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(postfix))
    ref_model.eval()
    labels = []
    preds = []
    val_correct = 0
    num_labels = 0
    predictions_list = []
    tqdm_eval_loader = tqdm(eval_dataloader, desc="Evaluating")
    print(len(dataset))

    for i, data_input in enumerate(tqdm_eval_loader):
        pred_boxes_img_id = data_input.pop('bboxes_img_id')
        pred_boxes = data_input.pop('bboxes')
        gt_box = data_input.pop('gt_box')
        inputs = to_device(data_input, args.device)
        with torch.no_grad():
            outputs = ref_model(**inputs)
            # scores = outputs[0].to('cpu')
        # softmax_scores = torch.softmax(scores, dim=1)

        predicted = torch.argmax(outputs[0], dim=1)
        correct = 0
        for j, pred_id in enumerate(predicted):
            pred_box = pred_boxes[pred_id]
            if computeIoU(pred_box, gt_box) > 0.5:
                correct += 1
        val_correct += correct
        num_labels += len(predicted)

        accuracy = val_correct / num_labels
        tqdm_eval_loader.set_description_str(f"[Acc]: {accuracy :.4f}")

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
    parser.add_argument("--eval_file", default='../data/datasets/testA_refcoco_unc_dets.json', type=str,
                        help="Input eval file")
    parser.add_argument("--output_dir", type=str, required=True,
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
    clip_model, preprocess = get_model(args.model_name, device)
    if args.resize:
        logger.info("Using resize transform...")
        preprocess = resize_transform(clip_model.visual.input_resolution)
    ref_model = ClipClassifierReverse(clip_model).to(device)
    convert_models_to_fp32(ref_model)

    # Get Dataset for evaluation
    eval_dataset = ReferEvaluateClipReverseDataset(args.eval_file, split='val', preprocess=preprocess, debug=args.debug)

    # Evaluate only
    ref_checkpoint_path = os.path.join(args.ref_model_dir, 'checkpoint-best/model.pt')
    print(ref_checkpoint_path)
    ref_checkpoint = torch.load(ref_checkpoint_path)

    print(f"Loading checkpoint from {ref_checkpoint_path}; Epoch: {ref_checkpoint['epoch']}")
    ref_model.load_state_dict(ref_checkpoint['state_dict'])
    print(f"Evaluating from file: {args.eval_file}")

    ref_model_to_eval = ref_model.module if hasattr(ref_model, 'module') else ref_model

    result = evaluate(args, ref_model_to_eval, eval_dataset, postfix="best_eval_only")

    print("Result:", result)


if __name__ == "__main__":
    main()
