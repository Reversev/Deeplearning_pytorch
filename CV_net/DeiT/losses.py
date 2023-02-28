# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/26 9:22
# @Author: 'IReverser'
# @ FileName: losses.py
import torch
import torch.nn as nn
from torch.nn import functional as F


class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self,
                 base_criterion: nn.Module,
                 teacher_model: nn.Module,
                 distillation_type: str,
                 alpha: float,
                 tau: float):
        super(DistillationLoss, self).__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs (class_token), outputs_kd (dist_token)]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # batchmean等于KLDiv的真正数学定义
            distillation_loss = F.kl_div(F.log_softmax(outputs_kd / T, dim=1),
                                         F.log_softmax(teacher_outputs / T, dim=1),
                                         reduction="mean",
                                         log_target=True) * (T ** 2)
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


