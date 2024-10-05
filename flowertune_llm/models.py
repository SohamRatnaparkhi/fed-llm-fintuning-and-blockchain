import math

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from flwr.common.typing import NDArrays
from unsloth import FastLanguageModel

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

model = None
tokenizer = None

def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024**3)
    max_memory = f"{free_in_GB-2}GB"

    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    print(f'No. of gpus = {n_gpus}')
    print(f'Max memory = {max_memory}')

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=model_cfg.quantization == 4,
    #     load_in_8bit=model_cfg.quantization == 8,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     llm_int8_threshold=6.0,
    #     llm_int8_has_fp16_weight=False,
    #     llm_int8_enable_fp32_cpu_offload=True
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     max_memory=max_memory,
    #     pretrained_model_name_or_path=model_cfg.name,
    #     local_files_only=True,
    #     quantization_config=quantization_config,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     use_cache=False if model_cfg.gradient_checkpointing else True
    # )

    # model = prepare_model_for_kbit_training(
    #     model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    # )

    # model_peft_config_map = {
    #     "falcon": "falcon",
    #     "llama-3.2": "llama",
    #     "llama-3.1": "llama",
    #     "llama2": "llama",  # Add this line to catch "models--llama2"
    #     "gemma": "gemma",
    #     "phi": "phi",
    #     "xgen": "xgen",
    #     "mixtral": "mixtral",
    #     "gpt2": "gpt2"
    # }

    # peft_config = get_config_based_on_model(model_cfg.lora.peft_lora_r, model_cfg.lora.peft_lora_alpha)
    
    # def get_model_type(model_name):
    #     for key in model_peft_config_map:
    #         if key in model_name.lower():
    #             return model_peft_config_map[key]
    #     raise ValueError(f"Unsupported model: {model_name}")

    # # Usage
    # model_type = get_model_type(model_cfg.name)
    # peft_config = peft_config[model_type]

    # return get_peft_model(model, peft_config)
    global model
    global tokenizer
    if model is not None:
        return model, tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg.name,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=model_cfg.quantization == 4,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_cfg.lora.peft_lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = model_cfg.lora.peft_lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model, tokenizer


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]

def get_config_based_on_model(r, alpha): 

    falcon_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        modules_to_save=["word_embeddings", "lm_head"]
    )


    llama_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )

    gemma_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )


    phi_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        modules_to_save=["wte", "lm_head"]
    )


    xgen_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )

    mixtral_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["embed_tokens", "lm_head"]
    )


    gpt2_peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.075,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
        modules_to_save=["wte", "wpe", "lm_head"]
    )

    return {
        "falcon": falcon_peft_config,
        "llama": llama_peft_config,
        "gemma": gemma_peft_config,
        "phi": phi_peft_config,
        "xgen": xgen_peft_config,
        "mixtral": mixtral_peft_config,
        "gpt2": gpt2_peft_config,
    }


