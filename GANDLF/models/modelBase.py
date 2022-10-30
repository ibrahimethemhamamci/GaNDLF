# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""

import torch.nn as nn
import torch.nn.functional as F
from . import networks

from acsconv.converters import ACSConverter, Conv3dConverter, SoftACSConverter

from GANDLF.utils import (get_linear_interpolation_mode,
                          get_filename_extension_sanitized,
                         )
from GANDLF.utils.modelbase import get_modelbase_final_layer
from GANDLF.models.seg_modules.average_pool import (
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)

import sys

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
        
        gan_model_names_list = [
            "sdnet",
            "pix2pix",
            "pix2pixHD",
            "cycleGAN",
            "dcgan",
        ]
            
        self.model_name = parameters["model"]["architecture"]
        self.n_dimensions = parameters["model"]["dimension"]
        self.n_channels = parameters["model"]["num_channels"]
        if "num_classes" in parameters["model"]:
            self.n_classes = parameters["model"]["num_classes"]
            if self.model_name in gan_model_names_list and self.n_classes==0:    
                self.n_classes=self.n_channels
        else:
            self.n_classes = len(parameters["model"]["class_list"])
            if self.model_name in gan_model_names_list and self.n_classes==0:    
                self.n_classes=self.n_channels
        self.base_filters = parameters["model"]["base_filters"]
        self.norm_type = parameters["model"]["norm_type"]
        self.patch_size = parameters["patch_size"]
        self.batch_size = parameters["batch_size"]
        self.amp = parameters["model"]["amp"]
        
        if self.model_name in gan_model_names_list:    
            try:
                self.input_range = parameters["range_images"]["input_range"]
            except KeyError:
                self.input_range=[0,1]
                
            try:
                self.output_range = parameters["range_images"]["output_range"]
            except KeyError:
                self.output_range=[0,1]
            
            if not ("architecture_gen" in parameters["model"]):
                sys.exit("The 'model' parameter needs 'architecture_gen' key to be defined")

            if not ("architecture_disc" in parameters["model"]):
                sys.exit("The 'model' parameter needs 'architecture_disc' key to be defined")
            
            try:
                self.loss_mode = parameters["model"]["loss_mode"]
            except KeyError:
                self.loss_mode = "vanilla"
            #gan mode will be added to parameter parser
            self.gen_model_name = parameters["model"]["architecture_gen"]
            self.disc_model_name = parameters["model"]["architecture_disc"]
            self.dev = parameters["device"]
            parameters["model"]["amp"] = False
            self.amp, self.device, self.gpu_ids= networks.device_parser(self.amp, self.dev)
        
        self.final_convolution_layer = self.get_final_layer(
            parameters["model"]["final_layer"]
        )

        self.linear_interpolation_mode = get_linear_interpolation_mode(
            self.n_dimensions
        )

        self.sigmoid_input_multiplier = parameters["model"].get(
            "sigmoid_input_multiplier", 1.0
        )

        # based on dimensionality, the following need to defined:
        # convolution, batch_norm, instancenorm, dropout
        if self.n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            self.GlobalAvgPool = GlobalAveragePooling2D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)
            self.converter = None

        elif self.n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            self.GlobalAvgPool = GlobalAveragePooling3D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

            # define 2d to 3d model converters
            converter_type = parameters["model"].get("converter_type", "soft").lower()
            self.converter = SoftACSConverter
            if converter_type == "acs":
                self.converter = ACSConverter
            elif converter_type == "conv3d":
                self.converter = Conv3dConverter

        else:
            raise ValueError(
                "GaNDLF only supports 2D and 3D computations. {}D computations are not currently supported".format(
                    self.n_dimensions
                )
            )

    def get_final_layer(self, final_convolution_layer):
        return get_modelbase_final_layer(final_convolution_layer)

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

    
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
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