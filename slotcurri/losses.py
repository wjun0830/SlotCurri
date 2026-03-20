from typing import Any, Dict, Optional, Tuple

import einops
import torch
from torch import nn

from slotcurri import modules, utils
from pytorch_msssim import ssim

@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
            self.loss_fn_none = getattr(torch.nn, loss)(reduction="none", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, return_none=False) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)
        if return_none:
            return self.loss_fn_none(prediction, target)
        return self.loss_fn(prediction, target)

class SSIMLoss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # print('SSIM3D loss target shape : ', target.shape)
        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # print('SSIM3D loss prediction shape : ', prediction.shape)
        # Convert to dimension order (batch, positions, dims)
        # prediction = self.to_canonical_dims(prediction)
        #
        # if self.pred_dims:
        #     prediction = prediction[..., self.pred_dims]


        return prediction

    def forward(self, feat_recon, feat_orig):
        # feat_recon, feat_orig: (B, C, T, H, W)
        B, C, T, H, W = feat_recon.shape
        eps = 1e-6

        # orig_flat = feat_orig.reshape(B, -1)
        # min_o     = orig_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
        # max_o     = orig_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
        # recon_flat= feat_recon.reshape(B, -1)
        # min_r     = recon_flat.min(dim=1)[0].view(B, 1, 1, 1, 1)
        # max_r     = recon_flat.max(dim=1)[0].view(B, 1, 1, 1, 1)
        # feat_orig_norm  = (feat_orig  - min_o) / (max_o - min_o + eps)
        # feat_recon_norm = (feat_recon - min_r) / (max_r - min_r + eps)

        orig_flat = feat_orig.reshape(B, -1)
        recon_flat = feat_recon.reshape(B, -1)
        all_flat = torch.cat([orig_flat, recon_flat], dim=1)
        min_val = all_flat.min(dim=1)[0].view(B, 1, 1, 1, 1).detach()
        max_val = all_flat.max(dim=1)[0].view(B, 1, 1, 1, 1).detach()

        feat_orig_norm = (feat_orig - min_val) / (max_val - min_val + eps)
        feat_recon_norm = (feat_recon - min_val) / (max_val - min_val + eps)

        loss_ssim3d = 1.0 - ssim(
            feat_recon_norm, feat_orig_norm.detach(),
            data_range=1.0,   # [0,1]
            size_average=True,
            win_size=3
        )
        return loss_ssim3d

        # ssim3d = SSIM3D(window_size=3, reduction='mean')
        #
        # loss_ssim3d = 1.0 - ssim3d(x, y)
        # return loss_ssim3d

class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)

class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, slots, _):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, D = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        loss = self.criterion(ss, target)
        return loss