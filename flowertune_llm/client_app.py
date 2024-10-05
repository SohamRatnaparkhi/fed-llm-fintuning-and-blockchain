"""flowertune-llm: A Flower / FlowerTune app."""

import math
import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.config import unflatten_dict
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
import transformers
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting, load_data,
    replace_keys)
from flowertune_llm.models import (get_model, get_parameters, set_parameters)

# Avoid warnings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# from accelerate import DistributedDataParallelKwargs
# from accelerate import DistributedDataParallelKwargs

# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

class FlowerClient(NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        formatting_prompts_func,
        data_collator,
        num_rounds,
        tokenizer,
        model
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_argumnets = transformers.TrainingArguments(
            **train_cfg.training_arguments)
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = data_collator
        self.num_rounds = num_rounds
        self.trainset = trainset
        # self.training_arguments.fp16 = True
        # self.training_arguments.ddp_kwargs = ddp_kwargs

        # instantiate model
        # self.model, self.tokenizer = get_model(model_cfg)
        self.model = model
        self.tokenizer = tokenizer

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_argumnets.learning_rate = new_lr
        self.training_argumnets.output_dir = config["save_path"]

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                num_train_epochs = 3, # Set this for 1 full training run.
                max_steps = 10,
                learning_rate = new_lr,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = config["save_path"],
            ),
        )

        # Do local training
        results = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    model, tokenizer = get_model(cfg.model)
    # Let's get the client partition~
    client_trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(tokenizer)

    return FlowerClient(
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        trainset=client_trainset,
        formatting_prompts_func=formatting_prompts_func,
        data_collator=data_collator,
        num_rounds=num_rounds,
        tokenizer=tokenizer,
        model=model
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
