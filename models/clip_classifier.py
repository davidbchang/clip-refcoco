import numpy as np
import clip
import torch
import torch.nn as nn


class ClipClassifierReverse(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        # CLIP logit_scale https://github.com/openai/CLIP/blob/main/clip/model.py
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.normalize = True

        self.loss_mc = torch.nn.CrossEntropyLoss()

    # gt_vis_feats
    def forward(self, vis_feats, input_ids, labels=None):
        batch_size, options, c, w, h = vis_feats.size()  # B x N x 3 x 384 x 384
        vis_feats = vis_feats.view(batch_size * options, c, w, h)  # B*N x 3 x 384 x 384
        input_ids = input_ids.squeeze(1)  # B x 1 x 77 -> B x 77
        text_features = self.clip_model.encode_text(input_ids)  # B X D=768
        image_features = self.clip_model.encode_image(vis_feats)  # B*N x D

        if self.normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # mc loss
        image_features = image_features.view(batch_size, options, -1)  # B x N x D
        text_features = text_features.unsqueeze(2)  # B x D X 1
        logits = torch.bmm(image_features, text_features)  # B x N x 1
        logits = logits.squeeze(2)  # B x N

        logits = logits * logit_scale

        outputs = (logits,)
        if labels is not None:
            loss_mc = self.loss_mc(logits, labels.view(-1))
            outputs = (loss_mc,) + outputs
            return outputs  # , logits_per_image, logits_per_text

        return outputs

        # return logits_per_image, logits_per_text
