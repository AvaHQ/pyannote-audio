# KZ: Adapted from Pyannote's PyanNet class:
# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/segmentation/PyanNet.py

# LICENSE from https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/segmentation/PyanNet.py
# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict
from .transformer_utils import make_model


class TransformerPyanNet(Model):
    """PyanNet segmentation model

    SincNet > Transformer > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    transformer: dict, optional
        Keyword arguments passed to the Transformer layer.
        (including SincNet parameters)
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    TRANSFORMER_DEFEAULTS = {"stride": 10,
                            "N": 8,
                            "d_model": 60,
                            "d_ff": 2048,
                            "h": 6,
                            "dropout": 0.1}

    def __init__(
        self,
        transformer: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)
        transformer = merge_dict(self.TRANSFORMER_DEFEAULTS, transformer)
        self.save_hyperparameters("transformer", "linear")

        self.transformer = make_model(**transformer)

        if linear["num_layers"] < 1:
            return


    def build(self):
        in_features = 60

        self.classifier = nn.Linear(in_features, len(self.specifications.classes))
        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.transformer.encode(waveforms)

        return self.activation(self.classifier(outputs))