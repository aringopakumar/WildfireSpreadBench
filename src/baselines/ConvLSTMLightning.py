from typing import Any, List

import torch.nn as nn

from .BaseModel import BaseModel
from .utae_paps_models.convlstm import ConvLSTM, ConvLSTM_Seg


class ConvLSTM_Seg_multi_layers(ConvLSTM_Seg):
    """_summary_ ConvLSTM class where the number of layers can be set. 
    """
    def __init__(
        self,
        num_classes,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        pad_value=0,
    ):
        super(ConvLSTM_Seg, self).__init__()
        self.convlstm_encoder = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
            num_layers=num_layers,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value


class ConvLSTMLightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        img_height_width: List[int],
        kernel_size: List[int],
        num_layers: int,
        *args: Any,
        **kwargs: Any
    ):
        # Accept lists from the config (newer jsonargparse fails to adapt
        # Tuple[int, int] hints), then convert to tuples. ConvLSTM's internals
        # unpack input_size and require kernel_size to be a tuple
        # (see ConvLSTM._check_kernel_size_consistency).
        img_height_width = tuple(img_height_width)
        kernel_size = tuple(kernel_size)

        # Pop so load_from_checkpoint (which replays saved hparams via **kwargs)
        # does not collide with required_img_size, which we set explicitly below
        # from img_height_width.
        kwargs.pop("required_img_size", None)

        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            required_img_size=img_height_width, # Important: ConvLSTM requires a fixed input size.
            *args,
            **kwargs
        )

        self.model = ConvLSTM_Seg_multi_layers(
            num_classes=1,
            input_size=img_height_width,
            input_dim=n_channels,
            hidden_dim=64,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )
