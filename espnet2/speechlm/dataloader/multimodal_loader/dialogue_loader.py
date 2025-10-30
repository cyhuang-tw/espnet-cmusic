#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Dialogue data loading utilities supporting multimodal conversation formats."""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf


class DialogueReader:
    """Dict-like dialogue reader for multimodal conversation data.

    Loads dialogue data from a folder containing JSON files. Each JSON file
    contains a dictionary mapping example_id to messages, where messages is a
    list of triplet tuples: (role, modality, content).

    Format:
    - Each .json file contains: {example_id: [(role, modality, content), ...]}
    - role: Must be one of ["user", "assistant", "system"]
    - modality: Must be one of ["text", "audio"]
    - content:
        - For text: string content
        - For audio: path to audio file (will be loaded as numpy array)

    Args:
        dialogue_folder: Path to folder containing .json dialogue files
        valid_ids: List of valid IDs to keep (optional, keeps all if None)
    """

    VALID_ROLES = {"user", "assistant", "system"}
    VALID_MODALITIES = {"text", "audio"}

    def __init__(self, dialogue_folder: str, valid_ids: Optional[List[str]] = None):
        self.dialogues = {}
        dialogue_path = Path(dialogue_folder)

        assert dialogue_path.exists(), f"Dialogue folder not found: {dialogue_folder}"
        assert dialogue_path.is_dir(), f"Expected a folder, but got: {dialogue_folder}"

        # Convert valid_ids to set for faster lookup
        valid_ids_set = set(valid_ids) if valid_ids else None

        # Find and load all .json files in the folder
        json_files = list(dialogue_path.glob("*.json"))
        assert json_files, f"No .json files found in {dialogue_folder}"

        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert isinstance(
                data, dict
            ), f"Invalid format in {json_file}: expected dict, got {type(data)}"

            for example_id, messages in data.items():
                # Skip if not in valid_ids
                if valid_ids_set and example_id not in valid_ids_set:
                    continue

                # Store raw messages - validation happens in __getitem__
                self.dialogues[example_id] = messages

    def __getitem__(
        self, key: str
    ) -> List[Tuple[str, str, Union[str, Tuple[np.ndarray, int]]]]:
        """Get dialogue messages by ID with validation and content loading.

        Returns:
            List of tuples where each tuple is (role, modality, content).
            Content format depends on modality:
            - text: string
            - audio: (audio_array, sample_rate) where audio_array has shape
                    [num_channels, num_samples]
        """
        messages = self.dialogues[key]

        assert isinstance(messages, list), f"Invalid messages for {key}: expected list"

        validated = []
        for i, msg in enumerate(messages):
            # Convert list to tuple if necessary
            if isinstance(msg, list):
                msg = tuple(msg)

            assert isinstance(msg, tuple) and len(msg) == 3, (
                f"Invalid message at index {i} for {key}: "
                f"expected (role, modality, content) triplet"
            )

            role, modality, content = msg

            assert role in self.VALID_ROLES, (
                f"Invalid role '{role}' at index {i} for {key}: "
                f"must be one of {self.VALID_ROLES}"
            )

            assert modality in self.VALID_MODALITIES, (
                f"Invalid modality '{modality}' at index {i} for {key}: "
                f"must be one of {self.VALID_MODALITIES}"
            )

            # Validate and process content based on modality
            if modality == "text":
                assert isinstance(content, str), (
                    f"Invalid text content at index {i} for {key}: "
                    f"expected string, got {type(content)}"
                )
                processed_content = content
            elif modality == "audio":
                # Load audio file
                audio_path = Path(content)
                assert (
                    audio_path.exists()
                ), f"Audio file not found at index {i} for {key}: {content}"

                # Load audio using soundfile
                audio_data, sample_rate = sf.read(audio_path, dtype="float32")

                # Ensure shape is [num_channels, num_samples]
                if audio_data.ndim == 1:
                    # Single channel - add channel dimension
                    audio_data = audio_data[np.newaxis, :]
                elif audio_data.ndim == 2:
                    # Multi-channel - transpose from [samples, channels] to [channels, samples]
                    audio_data = audio_data.T
                else:
                    raise ValueError(
                        f"Unexpected audio shape at index {i} for {key}: {audio_data.shape}"
                    )

                processed_content = (audio_data, sample_rate)

            validated.append((role, modality, processed_content))

        return validated

    def __contains__(self, key: str) -> bool:
        """Check if ID exists."""
        return key in self.dialogues

    def __len__(self) -> int:
        """Return number of dialogues."""
        return len(self.dialogues)

    def keys(self):
        """Return iterator over IDs."""
        return self.dialogues.keys()

    def values(self):
        """Return iterator over dialogues."""
        # Note: returns raw values without validation
        return self.dialogues.values()

    def items(self):
        """Return iterator over (id, dialogue) pairs."""
        # Note: returns raw items without validation
        return self.dialogues.items()
