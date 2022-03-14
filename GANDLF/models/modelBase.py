# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from . import networks
from GANDLF.models.seg_modules.average_pool import (
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)
from .__init__ import global_gan_models_dict


# this will no longer be requires once base pytorch moves >= 1.10.1
from torch.nn.common_types import _size_6_t
from torch.nn.modules.utils import _ntuple
from torch.nn.modules.padding import _ReflectionPadNd
from typing import Tuple
class ReflectionPad3d(_ReflectionPadNd):
    # copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/padding.html#ReflectionPad3d
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ReflectionPad3d(1)
        >>> input = torch.arange(8, dtype=torch.float).reshape(1, 1, 2, 2, 2)
        >>> m(input)
        tensor([[[[[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]],
                  [[7., 6., 7., 6.],
                   [5., 4., 5., 4.],
                   [7., 6., 7., 6.],
                   [5., 4., 5., 4.]],
                  [[3., 2., 3., 2.],
                   [1., 0., 1., 0.],
                   [3., 2., 3., 2.],
                   [1., 0., 1., 0.]]]]])
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)
# this will no longer be requires once base pytorch moves >= 1.10.1

class ModelBase(nn.Module):
    """
    This is the base model class that all other architectures will need to derive from
    """

    def __init__(self, parameters):
        """
        This defines all defaults that the model base uses

        Args:
            parameters (dict): This is a dictionary of all parameters that are needed for the model.
        """
        super(ModelBase, self).__init__()
        self.model_name = parameters["model"]["architecture"]

        if self.model_name in global_gan_models_dict:
            
            if not ("architecture_gen" in parameters["model"]):
                sys.exit("The 'model' parameter needs 'architecture_gen' key to be defined")

            if not ("architecture_disc" in parameters["model"]):
                sys.exit("The 'model' parameter needs 'architecture_disc' key to be defined")

            self.loss_mode = parameters["model"]["loss_mode"]
            #gan mode will be added to parameter parser
            self.gen_model_name = parameters["model"]["architecture_gen"]
            self.disc_model_name = parameters["model"]["architecture_disc"]
            self.dev = parameters["device"]
            parameters["model"]["amp"] = False
            self.amp = parameters["model"]["amp"]
            self.amp, self.device, self.gpu_ids= networks.device_parser(self.amp, self.dev)
            self.final_convolution_layer = None
            
        else:
            
            self.final_convolution_layer = self.get_final_layer(
            parameters["model"]["final_layer"]
            )
            
        self.n_dimensions = parameters["model"]["dimension"]
        self.n_channels = parameters["model"]["num_channels"]
        if "num_classes" in parameters["model"]:
            self.n_classes = parameters["model"]["num_classes"]
        else:
            self.n_classes = len(parameters["model"]["class_list"])
        self.base_filters = parameters["model"]["base_filters"]
        self.norm_type = parameters["model"]["norm_type"]
        self.patch_size = parameters["patch_size"]
        self.batch_size = parameters["batch_size"]
        self.amp = parameters["model"]["amp"]
        
        # based on dimensionality, the following need to defined:
        # convolution, batch_norm, instancenorm, dropout

        if self.n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.linear_interpolation_mode = "bilinear"
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            self.ReflectionPad = nn.ReflectionPad2d
            self.GlobalAvgPool = GlobalAveragePooling2D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

        elif self.n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.linear_interpolation_mode = "trilinear"
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            self.ReflectionPad = ReflectionPad3d # the "nn.ReflectionPad3d" class requires pytorch >= 1.10.1
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)
            
            
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for parameters in net.parameters():
                    parameters.requires_grad = requires_grad

            self.GlobalAvgPool = GlobalAveragePooling3D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

    def get_final_layer(self, final_convolution_layer):
        """
        This function gets the final layer of the model.

        Args:
            final_convolution_layer (str): The final layer of the model as a string.

        Returns:
            Functional: sigmoid, softmax, or None
        """
        none_list = [
            "none",
            None,
            "None",
            "regression",
            "classification_but_not_softmax",
            "logits",
            "classification_without_softmax",
        ]

        if final_convolution_layer in ["sigmoid", "sig"]:
            final_convolution_layer = torch.sigmoid

        elif final_convolution_layer in ["softmax", "soft"]:
            final_convolution_layer = F.softmax

        elif final_convolution_layer in none_list:
            final_convolution_layer = None

        return final_convolution_layer

    def get_norm_type(self, norm_type, dimensions):
        """
        This function gets the normalization type for the model.

        Args:
            norm_type (str): Normalization type as a string.
            dimensions (str): The dimensionality of the model.

        Returns:
            _InstanceNorm or _BatchNorm: The normalization type for the model.
        """
        if dimensions == 3:
            if norm_type == "batch":
                norm_type = nn.BatchNorm3d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm3d
            else:
                norm_type = None
        elif dimensions == 2:
            if norm_type == "batch":
                norm_type = nn.BatchNorm2d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm2d
            else:
                norm_type = None

        return norm_type
