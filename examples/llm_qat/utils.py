# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os
import types
from contextlib import contextmanager
from functools import partial

import datasets
import torch
import transformers
from peft import LoraConfig, TaskType
from transformers import default_data_collator

IGNORE_INDEX = -100


@contextmanager
def local_main_process_first():
    """Context manager to run code on local rank 0 first (per-node).

    Uses a file lock so only one process per node downloads the dataset.
    Avoids global barriers that timeout on clusters without shared filesystems.
    """
    if not torch.distributed.is_initialized():
        yield
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    lock_file = "/tmp/.dataset_download.lock"

    if local_rank == 0:
        yield
        # Signal to other local ranks that download is done
        with open(lock_file, "w") as f:
            f.write("done")
    else:
        # Wait for local rank 0 to finish
        import time

        while not os.path.exists(lock_file):
            time.sleep(1)
        yield


DATASET_REGISTRY = {
    "Daring-Anteater": "nvidia/Daring-Anteater",
}


def _normalize_messages(sample):
    """Normalize a dataset sample into a list of (role, content) tuples.

    Supports two formats:
      - Daring-Anteater style: {"conversations": [{"from": "User"|"Assistant", "value": "..."}], "system": "..."}
      - OpenAI messages style:  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
    """
    if "messages" in sample:
        # OpenAI messages format
        return [
            (msg["role"].lower(), msg["content"])
            for msg in sample["messages"]
        ]
    elif "conversations" in sample:
        # Daring-Anteater / NVIDIA conversation format
        messages = []
        if sample.get("system"):
            messages.append(("system", sample["system"]))
        for turn in sample["conversations"]:
            role = turn["from"].lower()  # "User" -> "user", "Assistant" -> "assistant"
            messages.append((role, turn["value"]))
        return messages
    else:
        raise ValueError(f"Unknown dataset format. Sample keys: {list(sample.keys())}")


def _tokenize_messages(messages, tokenizer, max_length):
    """Tokenize a normalized message list into input_ids, attention_mask, labels.

    Only assistant turns are used as training targets; all other roles are masked.
    """
    all_input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
    all_labels = [IGNORE_INDEX] if tokenizer.bos_token_id else []

    for role, content in messages:
        input_ids = tokenizer.encode(content + "\n", add_special_tokens=False)
        labels = input_ids if role == "assistant" else [IGNORE_INDEX] * len(input_ids)

        all_input_ids.extend(input_ids)
        all_labels.extend(labels)

        if len(all_input_ids) > max_length:
            break

    all_input_ids.append(tokenizer.eos_token_id)
    all_labels.append(IGNORE_INDEX)
    all_attention_mask = [1] * len(all_input_ids)

    cur_seq_length = len(all_input_ids)
    if cur_seq_length < max_length:
        pad_token = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        all_input_ids += [pad_token] * (max_length - cur_seq_length)
        all_attention_mask += [0] * (max_length - cur_seq_length)
        all_labels += [IGNORE_INDEX] * (max_length - cur_seq_length)

    return {
        "input_ids": all_input_ids[:max_length],
        "attention_mask": all_attention_mask[:max_length],
        "labels": all_labels[:max_length],
    }


_dataset_cache = {}


def get_chat_dataset(
    dataset_name: str,
    tokenizer: transformers.AutoTokenizer,
    split="train",
    max_length=4096,
    train_size=0,
    eval_size=0,
    hf_token=None,
):
    """Load and tokenize a chat dataset. Supports any HuggingFace dataset with either
    Daring-Anteater conversation format or OpenAI messages format.

    Args:
        dataset_name: A key in DATASET_REGISTRY (e.g. "Daring-Anteater") or a HuggingFace
                      dataset ID (e.g. "baseten/gamma-paste-text-train-v3").
        tokenizer: The tokenizer to use.
        split: "train" or "test".
        max_length: Maximum sequence length.
        train_size: Number of training samples (0 = use all available minus eval).
        eval_size: Number of eval samples (0 = default 2000).
        hf_token: Optional HuggingFace token for private datasets.
    """
    hf_dataset_id = DATASET_REGISTRY.get(dataset_name, dataset_name)

    def process_and_tokenize(sample):
        messages = _normalize_messages(sample)
        return _tokenize_messages(messages, tokenizer, max_length)

    if hf_dataset_id in _dataset_cache:
        dataset = _dataset_cache[hf_dataset_id]
    else:
        with local_main_process_first():
            load_kwargs = {"token": hf_token} if hf_token else {}
            dataset = datasets.load_dataset(hf_dataset_id, split="train", **load_kwargs)
            # Shuffle and subsample the dataset
            eval_size = 2000 if eval_size == 0 else eval_size
            train_size = len(dataset) - eval_size if train_size == 0 else train_size
            assert train_size + eval_size <= len(dataset) and train_size > 0 and eval_size > 0, (
                "not enough data for train-eval split"
            )
            dataset = dataset.shuffle(seed=42).select(range(train_size + eval_size))
            dataset = dataset.map(process_and_tokenize, remove_columns=list(dataset.features))
            dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        _dataset_cache[hf_dataset_id] = dataset
    return dataset[split]


# Keep backward-compatible alias
def get_daring_anteater(tokenizer, split="train", max_length=4096, train_size=0, eval_size=0):
    return get_chat_dataset("Daring-Anteater", tokenizer, split, max_length, train_size, eval_size)


def make_supervised_data_module(
    dataset="Daring-Anteater",
    tokenizer: transformers.PreTrainedTokenizer = None,
    train_size: int = 0,
    eval_size: int = 0,
    hf_token=None,
) -> dict:
    """Make dataset and collator for supervised fine-tuning.

    Accepts any HuggingFace dataset ID or a key from DATASET_REGISTRY.
    The dataset must have either OpenAI messages format or Daring-Anteater conversation format.
    """
    train_dataset = get_chat_dataset(
        dataset, tokenizer, "train", tokenizer.model_max_length, train_size, eval_size, hf_token
    )
    val_dataset = get_chat_dataset(
        dataset, tokenizer, "test", tokenizer.model_max_length, train_size, eval_size, hf_token
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": default_data_collator,
    }


def get_lora_config():
    return LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )


def monkey_patch_training_step_to_fix_memory_leak(trainer):
    def new_func(original_f_name, trainer, *args, **kwargs):
        gc.collect()
        return getattr(trainer, original_f_name)(*args, **kwargs)

    for f_name in ["training_step", "prediction_step", "_load_best_model"]:
        setattr(trainer, "_original_" + f_name, getattr(trainer, f_name))
        setattr(
            trainer, f_name, types.MethodType(partial(new_func, "_original_" + f_name), trainer)
        )


def get_metrics_with_perplexity(metrics):
    """Add perplexity to the metrics."""
    if "eval_loss" in metrics:
        metrics["perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])))
    return metrics
