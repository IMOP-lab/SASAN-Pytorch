from __future__ import annotations
import warnings
from collections.abc import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, LossReduction, look_up_option, pytorch_after

class DiceLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
        if self.other_act is not None:
            input = self.other_act(input)
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                target = target[:, 1:]
                input = input[:, 1:]
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis
        intersection = torch.sum(target * input, dim=reduce_axis)
        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o
        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)
        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)
        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)
        elif self.reduction == LossReduction.NONE.value:
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return f

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes=4, scaling_factor=10):
        super().__init__()
        self.num_classes = num_classes
        self.scaling_factot = scaling_factor

    def forward(self, inputs, targets):
        targets = one_hot(targets, self.num_classes)
        inputs_boundary = F.avg_pool3d(inputs, kernel_size=3, stride=1, padding=1) - inputs
        targets_boundary = F.avg_pool3d(targets, kernel_size=3, stride=1, padding=1) - targets
        boundary_loss = F.mse_loss(inputs_boundary, targets_boundary)*self.scaling_factot 
        return boundary_loss

class BoundaryReaLossBase(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def baseloss(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)
        
        return loss

    def compute_loss(self, inputs, target):
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        loss = 0.0
        for i in range(self.n_classes):
            loss += self.baseloss(inputs[:, i], target[:, i])
        return loss

class BoundaryReaLoss(BoundaryReaLossBase):
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i).float()
            tensor_list.append(temp_prob)
        output_tensor = torch.stack(tensor_list, dim=1)
        return output_tensor.squeeze(2)
        
    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0.0
        for i in range(inputs.shape[2]):
            slice_input = inputs[:, :, i, :, :]
            slice_target = target[:, :, i, :, :]
            for j in range(0, self.n_classes):
                loss += self.baseloss(slice_input[:, j], slice_target[:, j])
        return loss / (self.n_classes * inputs.shape[2])

class SMLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: torch.Tensor | None = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_boundary: float = 1.0,
        weight = [1, 2, 3, 4],
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
        )
        # self.boundary = BoundaryLoss()
        self.boundary = BoundaryReaLoss(n_classes=4)
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_ce < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        if lambda_boundary < 0.0: 
            raise ValueError("lambda_boundary should be no less than 0.0.")
        self.lambda_dice = nn.Parameter(torch.tensor(lambda_dice))
        self.lambda_ce = nn.Parameter(torch.tensor(lambda_ce))
        self.lambda_boundary = nn.Parameter(torch.tensor(lambda_boundary))
        self.old_pt_ver = not pytorch_after(1, 10)

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif self.old_pt_ver:
            warnings.warn(
                f"Multichannel targets are not supported in this older Pytorch version {torch.__version__}. "
                "Using argmax (as a workaround) to convert target to a single channel."
            )
            target = torch.argmax(target, dim=1)
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)
        return self.cross_entropy(input, target)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )
        dice_loss = self.dice(input, target)
        # print("dice_loss:",dice_loss)
        ce_loss = self.ce(input, target)
        # print("ce_loss:",ce_loss)
        boundary_loss = self.boundary(input, target)
        # print("boundary_loss:",boundary_loss)
        total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss + self.lambda_boundary * boundary_loss

        if dice_loss < 0.2 and ce_loss < 0.2:

            if dice_loss > ce_loss and dice_loss > boundary_loss:
                self.lambda_dice = nn.Parameter(self.lambda_dice * (1+0.005))
            elif dice_loss > ce_loss or dice_loss > boundary_loss:
                self.lambda_dice = nn.Parameter(self.lambda_dice * (1+0.001))

            elif ce_loss > dice_loss and ce_loss > boundary_loss:
                self.lambda_ce = nn.Parameter(self.lambda_ce * (1+0.005))
            elif ce_loss > dice_loss or ce_loss > boundary_loss:
                self.lambda_ce = nn.Parameter(self.lambda_ce * (1+0.001))   

            elif boundary_loss > dice_loss and boundary_loss > ce_loss:
                self.lambda_boundary = nn.Parameter(self.lambda_boundary * (1+0.005))
            elif boundary_loss > dice_loss or boundary_loss > ce_loss:
                self.lambda_boundary = nn.Parameter(self.lambda_boundary * (1+0.001))   

            total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss + self.lambda_boundary * boundary_loss
        else:
            total_loss = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        new_lambda_dice = nn.Parameter(F.softplus(self.lambda_dice))
        new_lambda_ce = nn.Parameter(F.softplus(self.lambda_ce))
        new_lambda_boundary = nn.Parameter(F.softplus(self.lambda_boundary))
        normalizing_factor = new_lambda_dice + new_lambda_ce + new_lambda_boundary
        new_lambda_dice = new_lambda_dice / normalizing_factor
        new_lambda_ce = new_lambda_ce / normalizing_factor
        new_lambda_boundary = new_lambda_boundary / normalizing_factor
        
        with torch.no_grad():
            self.lambda_dice.data = nn.Parameter(new_lambda_dice).data
            self.lambda_ce.data = nn.Parameter(new_lambda_ce).data
            self.lambda_boundary.data = nn.Parameter(new_lambda_boundary).data
            
        loss_dict = {
                'total_loss': total_loss,
                'dice_loss': dice_loss,
                'ce_loss': ce_loss,
                'boundary_loss': boundary_loss,
            }
        return total_loss,loss_dict