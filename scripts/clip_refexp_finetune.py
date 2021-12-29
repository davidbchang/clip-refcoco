# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

# Distributed training and logging from
# https://github.com/huggingface/transformers/blob/27c1b656cca75efa0cc414d3bf4e6aacf24829de/examples/run_lm_finetuning.py


import argparse
import json
import logging
import os
import shutil
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm, trange
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tensorboardX import SummaryWriter

from transformers import (AdamW, Adafactor, get_linear_schedule_with_warmup)

import torch
import torch.optim.adam
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import clip
import sys

sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/models')
sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/utils')
sys.path.insert(0, '/mmfs1/gscratch/xlab/changd8/visual_comet/CSE490gFinal/dataloaders')
print(sys.path)
from clip_classifier import ClipClassifierReverse
from base_utils import computeIoU, to_device, resize_transform
from clip_refexp_datasets import (
    ReferClipReverseDataset, ReferEvaluateClipReverseDataset, ReferClipCollator
)

logger = logging.getLogger(__name__)
torch.cuda.empty_cache()
CLIP_CLASSES = ['RN101', 'RN50', 'RN50x16', 'RN50x4', 'ViT-B/16', 'ViT-B/32']


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


def train(args, train_dataset, eval_dataset, model):
    def save_and_evaluate(args, model, epoch, global_step, best_acc, tb_writer=None):
        """ Helper to save checkpoint, evaluate, and update tensorboard. """
        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format("best"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training

        # Eval
        logging.info(f"Evaluate epoch {epoch}: iter {global_step}")
        eval_postfix = f"epoch{epoch}_step{global_step}"

        # no support for distributed evaluation
        result: Dict[Dict] = evaluate(args, model_to_save, eval_dataset, eval_postfix)
        cur_acc = result[eval_postfix]["accuracy"]
        for key, value in result[eval_postfix].items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"eval_{key}", value, global_step)
                tb_writer.flush()

        # for mc, save the best model
        if cur_acc > best_acc:
            best_acc = cur_acc
            save_dict = {"accuracy": best_acc,
                         "epoch": epoch,
                         "split": eval_dataset.split,
                         "state_dict": model_to_save.state_dict()
                         }
            torch.save(save_dict, os.path.join(output_dir, 'model.pt'))
            logging.info(f"Saving best model at `{output_dir}` Epoch: {epoch}")

        return best_acc

    # train the model
    if args.local_rank in [-1, 0]:
        tb_dir = os.path.join(args.tb_dir, "tb")
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else RandomSampler(
        train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, num_workers=8,
                                  batch_size=args.train_batch_size, drop_last=True,
                                  collate_fn=ReferClipCollator())

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    if args.adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "eps": args.adam_epsilon,
        }
    optimizer_kwargs["lr"] = args.learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)

    if args.fp16:
        try:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        except ImportError:
            raise ImportError(
                "Please install pytorch >= 1.6.0 to use fp16 training.")

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  ARGS.TRAIN_BATCH_SIZE:", args.train_batch_size)

    global_step = 0
    best_acc = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    model.train()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])

    for epoch in train_iterator:
        logging.info("\n\n*** Starting Epoch: {} ***\n\n".format(epoch))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])

        train_correct = 0
        num_labels = 0
        for step, data_input in enumerate(epoch_iterator):
            bboxes_img_id = data_input.pop('bboxes_img_id')
            bboxes = data_input.pop('bboxes')
            gt_box = data_input.pop('gt_box')
            inputs = to_device(data_input, args.device)
            model.train()

            if args.fp16:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            loss = outputs[0]

            labels = inputs["labels"].detach()
            scores = outputs[1].detach()
            preds = torch.argmax(scores, dim=1)

            correct = torch.sum(torch.eq(preds, labels))
            train_correct += correct
            num_labels += len(labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            # update
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # clip gradient
                if args.max_grad_norm > 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # optimizer
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    convert_models_to_fp32(model)
                    optimizer.step()

                # scheduler
                scheduler.step()

                model.zero_grad()
                global_step += 1

                # log loss
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    log_step = args.train_batch_size * global_step
                    avg_logging_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar('loss', avg_logging_loss, global_step)
                    tb_writer.flush()
                    logging_loss = tr_loss
                    logging.info(f"Train Loss: {avg_logging_loss} [Acc]: {(train_correct / num_labels):.4f}"
                                 f" \t(Epoch: {epoch} \t Num of data: {log_step})")

                # Eval and save model checkpoint
                if not args.save_every_epoch and args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.local_rank in [-1, 0]:
                        best_acc = save_and_evaluate(args, model, epoch, global_step, best_acc, tb_writer)
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        ''' End of Epoch '''
        if args.save_every_epoch and args.local_rank in [-1, 0]:
            best_acc = save_and_evaluate(args, model, epoch, global_step, best_acc, tb_writer)
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    # Done training
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, postfix: Optional[str] = "latest") -> Dict[str, Dict[str, float]]:
    """
    Args:
        `postfix`: name for evaluation. Usually defined by epoch and iteration.

    Returns:
        Dict[str, Dict[str, float]]: {postfix [str] : {key [str] : value [float]} }
    """

    assert args.local_rank <= 0, "Evaluate does not support distributed launch"

    eval_output_dir = args.eval_output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, num_workers=8,
                                 batch_size=args.per_gpu_eval_batch_size, drop_last=False,
                                 shuffle=False)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(postfix))
    model.eval()
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
            outputs = model(**inputs)
            # scores = outputs[0].to('cpu')
        # softmax_scores = torch.softmax(scores, dim=1)

        predicted = torch.argmax(outputs[0].detach(), dim=1)

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

    ## Required parameters
    parser.add_argument("--model_name", default="RN50x16", type=str, choices=CLIP_CLASSES,
                        help="clip model to use.")
    parser.add_argument("--train_file", default='../data/datasets/train_refcoco_unc.json', type=str,
                        help="Input train file")
    parser.add_argument("--eval_file", default='../data/datasets/testA_refcoco_unc_dets.json', type=str,
                        help="Input eval file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # train options
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

    # train parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--adafactor", action='store_true',
                        help="Use Adafactor optimizer, else use Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # logs
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_every_epoch", action='store_true',
                        help="Evaluate and save every end of epoch, instead of every `save_steps`.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    # parallel
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through torch.cuda.amp) instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    print(args)
    if args.eval_output_dir is None:
        args.eval_output_dir = args.output_dir
    if args.tb_dir is None:
        args.tb_dir = args.output_dir

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import debugpy
        print("Waiting for debugger attach")
        debugpy.listen(args.server_ip, args.server_port)
        debugpy.wait_for_client()

    # Setup CUDA, GPU & distributed training
    args.distributed = args.local_rank != -1
    if not args.distributed or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        print('ARGS.nGPU:', args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    # Load model
    clip_model, preprocess = get_model(args.model_name, device)
    if args.resize:
        logger.info("Using resize transform...")
        preprocess = resize_transform(clip_model.visual.input_resolution)
    model = ClipClassifierReverse(clip_model).to(device)
    convert_models_to_fp32(model)

    # Get Dataset for all splits and update config
    dataset = ReferClipReverseDataset(args.train_file, split='train', preprocess=preprocess, debug=args.debug)
    eval_dataset = ReferEvaluateClipReverseDataset(args.eval_file, split='val', preprocess=preprocess, debug=args.debug)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    # Training
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process saves the output directory and model.
        file_handler = logging.FileHandler(os.path.join(args.output_dir, 'train.log'), mode="w", encoding=None, delay=False)
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            # Good practice: save your training arguments
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        if args.local_rank == 0:
            torch.distributed.barrier()  # End of barrier to make sure only the first process saves the output directory and model.

        ''' Begin actual training '''
        global_step, tr_loss = train(args, dataset, eval_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        if args.local_rank in [-1, 0]:
            # Saving best-practices in the end: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
            logger.info("Saving latest model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            # Save a trained model in the end, configuration and tokenizer using `save_pretrained()`.
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'clip_model.bin'))

        if args.local_rank == 0:
            torch.distributed.barrier()
    # Evaluate only
    else:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint-best/model.pt')
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        print(f"Loading checkpoint from {checkpoint_path}; Epoch: {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Evaluating from file: {args.eval_file}")

        model_to_eval = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        # no support for distributed evaluation
        result = evaluate(args, model_to_eval, eval_dataset, postfix="best_eval_only")
        print("Result:", result)


if __name__ == "__main__":
    main()
