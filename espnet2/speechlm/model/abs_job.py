#!/usr/bin/env python3
"""Abstract base class for job templates in SpeechLM training."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch.nn as nn


class AbsJobTemplate(ABC):
    """Abstract base class for training job templates.

    This class defines the interface for job templates that handle
    model and data collation function creation for different tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the job template with configuration.

        Args:
            config: Dictionary containing job configuration parameters.
        """
        self.config = config

    @abstractmethod
    def build_preprocessor(self) -> Callable:
        """Build and return the data collation function for PyTorch DataLoader.

        The collate_fn is used by PyTorch DataLoader to combine multiple
        samples into a batch. It handles all necessary preprocessing steps
        to convert a list of individual sample dictionaries into a single
        batch dictionary with tensors.

        The returned function should have the signature:
            collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]

        Typical preprocessing steps include:
        - Padding sequences to the same length within a batch
        - Converting numpy arrays or lists to PyTorch tensors
        - Applying any necessary transformations or augmentations
        - Creating attention masks or other auxiliary tensors
        - On-the-fly tokenization of text inputs
        - Resampling audio to target sample rates
        - Feature extraction (e.g., mel-spectrogram computation)

        By convention, the returned callable is usually implemented as a class
        with a __call__ method rather than a simple function, allowing it to
        maintain state (e.g., tokenizer instances, preprocessing parameters).

        Note: When using PyTorch DataLoader with num_workers > 0, the collate_fn
        is executed in the worker subprocesses, not the main process. This means:
        - Heavy preprocessing can be parallelized across workers
        - The collate_fn object is pickled and sent to each worker
        - Any state/resources should be picklable or initialized in __init__
        - Avoid using CUDA operations in collate_fn as workers don't have GPU access

        Returns:
            A callable function that takes a list of sample dictionaries
            and returns a batch dictionary suitable for model input.
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model.

        Returns:
            A PyTorch model instance.
        """
        raise NotImplementedError
