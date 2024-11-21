# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from sonar.models.mutox.builder import (
    MutoxClassifierBuilder,
    MutoxConfig,
    create_mutox_model,
)
from sonar.models.mutox.classifier import MutoxClassifier
from sonar.models.mutox.loader import convert_mutox_checkpoint

# Builder tests


@pytest.mark.parametrize("input_size", [256, 512, 1024])
@pytest.mark.parametrize("device", [torch.device("cpu")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mutox_classifier_builder(input_size, device, dtype):
    """Test MutoxClassifierBuilder initializes a model with correct configuration and dtype."""
    config = MutoxConfig(input_size=input_size)
    builder = MutoxClassifierBuilder(config, device=device, dtype=dtype)
    model = builder.build_model()

    # Check if model layers are correctly initialized with shapes
    assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"
    assert all(
        isinstance(layer, nn.Sequential) for layer in model.model_all.children()
    ), "All layers should be instances of nn.Sequential"

    test_input = torch.zeros((5, input_size), device=device, dtype=dtype)
    result = model(test_input)
    assert result.shape == (5, 1), f"Expected output shape (5, 1), got {result.shape}"


@pytest.mark.parametrize("input_size", [256, 512])
def test_create_mutox_model(input_size):
    """Test create_mutox_model function to confirm it creates a model with the specified config."""
    config = MutoxConfig(input_size=input_size)
    model = create_mutox_model(config, device=torch.device("cpu"))

    # Check if the created model has the expected structure and behavior
    test_input = torch.zeros((3, input_size))
    result = model(test_input)
    assert result.shape == (3, 1), f"Expected output shape (3, 1), got {result.shape}"
    assert isinstance(model, nn.Module), "Model should be an instance of nn.Module"


# Classifier tests


def test_mutox_classifier_forward():
    """Test that MutoxClassifier forward pass returns expected output shape."""
    test_model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    model = MutoxClassifier(test_model)

    test_input = torch.randn(3, 10)
    output = model(test_input)
    assert output.shape == (
        3,
        1,
    ), f"Expected output shape (3, 1), but instead got {output.shape}"


def test_mutox_classifier_forward_with_output_prob():
    """Test that MutoxClassifier forward pass applies sigmoid when output_prob=True."""
    test_model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
    )
    model = MutoxClassifier(test_model)

    test_input = torch.randn(3, 10)

    output = model(test_input, output_prob=True)

    assert output.shape == (
        3,
        1,
    ), f"Expected output shape (3, 1), but instead got {output.shape}"

    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Expected output values to be within the range [0, 1]"


def test_mutox_config():
    """Test that MutoxConfig stores the configuration for a model."""
    config = MutoxConfig(input_size=512)
    assert (
        config.input_size == 512
    ), f"Config input_size should be 512, but got {config.input_size}"


#  Loader tests


def test_convert_mutox_checkpoint():
    """Test convert_mutox_checkpoint correctly filters keys in the checkpoint."""
    checkpoint = {
        "model_all.layer1.weight": torch.tensor([1.0]),
        "model_all.layer1.bias": torch.tensor([0.5]),
        "non_model_key": torch.tensor([3.0]),
    }
    config = MutoxConfig(input_size=1024)
    converted = convert_mutox_checkpoint(checkpoint, config)

    # Verify only 'model_all.' keys are retained in the converted dictionary
    assert "model" in converted, "Converted checkpoint should contain a 'model' key"
    assert (
        "model_all.layer1.weight" in converted["model"]
    ), "Expected 'model_all.layer1.weight'"
    assert (
        "model_all.layer1.bias" in converted["model"]
    ), "Expected 'model_all.layer1.bias'"
    assert "non_model_key" not in converted["model"], "Unexpected 'non_model_key'"
