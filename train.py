#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForPreTraining, PaliGemmaProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration, Trainer, TrainingArguments
from datasets import Features, Value, Image, Dataset
from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig
import json
import argparse
from PIL import Image as PILImage
from dataclasses import dataclass
from typing import Optional, Dict, List
import torch.nn.functional as F

model_config = {
    "paligemma": {
        "model_id": "google/paligemma-3b-mix-224", 
        "model_class": AutoModelForPreTraining,
        "processor_class": PaliGemmaProcessor,
    },
    "llava": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor,
    },
    "qwen": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor,
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "model_class": MllamaForConditionalGeneration,
        "processor_class": AutoProcessor
    },
    "pixtral": {
        "model_id": "mistral-community/pixtral-12b",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor
    }
}

def setup():
    # Check if we're running in distributed mode
    if "LOCAL_RANK" not in os.environ:
        return False
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def generate_prompt_text(question, variants):
    options = '\n'.join([option['marker'] + ') '+ option['text'] for option in variants])
    text = (
        f"<image>\n"
        f"Питання: {question}\n"
        f"Варіанти відповіді:\n{options}\n\n"
        "Обери правильну відповідь. Напиши тільки одну відповідь, використовуючи українську літеру: А, Б, В або Г.\n"
        "Додатково запиши відповідь під цією буквою.\n"
        "Відповідь:"
    )
    return text

def prepare_preference_dataset(dataset):
    preference_dataset = []
    
    for entry in dataset:
        try:
            question = entry['question']
            prompt = generate_prompt_text(question['text'], question['choices'])
            
            # Load and verify image
            image = PILImage.open(entry['image_url'])
            
            for choice in question['choices']:
                if question['task_type'] == 'general':
                    if question['correct_answer']['marker'] == choice['marker']:
                        continue

                    preference_dataset.append({
                        'images': image,
                        'prompt': prompt,
                        'chosen': f"{question['correct_answer']['marker']}. {question['correct_answer']['text']}",
                        'rejected': f"{choice['marker']}. {choice['text']}"
                    })

                elif question['task_type'] == 'awareness':
                    preference_order = {'cultural': 0, 'semi-cultural': 1, 'general': 2}
                    sorted_choices = sorted(question['choices'], key=lambda c: preference_order.get(c['type'], 999))

                    for i in range(len(sorted_choices)):
                        for j in range(i + 1, len(sorted_choices)):
                            chosen = sorted_choices[i]
                            rejected = sorted_choices[j]

                            if preference_order[chosen['type']] < preference_order[rejected['type']]:
                                preference_dataset.append({
                                    'images': image,
                                    'prompt': prompt,
                                    'chosen': f"{chosen['marker']}. {chosen['text']}",
                                    'rejected': f"{rejected['marker']}. {rejected['text']}"
                                })
                                
        except Exception as e:
            print(f"Error preparing entry {entry.get('image_id', 'unknown')}: {str(e)}")
            continue
    
    return preference_dataset

def main():
    # Setup distributed training
    is_distributed = setup()
    
    # Get local rank (0 if not distributed)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test vision-language models on the CulturUA dataset')
    parser.add_argument('--model_name', type=str, required=True, choices=['paligemma', 'llava', 'qwen', 'llama', 'pixtral'],
                        help='Name of the model to test')
    parser.add_argument('--method', type=str, required=True, choices=['dpo', 'orpo', 'grpo'],
                        help='Method to run (e.g., dpo, orpo, grpo)')
    args = parser.parse_args()

    model_name = args.model_name
    method = args.method
    
    # Load dataset
    with open('culturUA/train_set.jsonl', 'r') as f:
        questions = [json.loads(line) for line in f]
    
    # Model initialization
    model_id = model_config[model_name]['model_id']
    model = model_config[model_name]['model_class'].from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank if is_distributed else "auto"}
    )
    processor = model_config[model_name]['processor_class'].from_pretrained(model_id)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare dataset
    features = Features({
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
        "images": Image(decode=True)
    })
    
    train_dataset = prepare_preference_dataset(questions)
    train_dataset = Dataset.from_list(train_dataset).cast(features)
    
    # Common training configuration
    per_device_batch_size = 16
    gradient_accumulation_steps = 2
    output_dir = f"{model_name}-{method}"
    
    # LoRA configuration optimized for H100
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
        use_rslora=True
    )

    # Common training arguments
    base_training_args = {
        "output_dir": output_dir,
        "bf16": True,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "logging_steps": 10,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 3,
        "remove_unused_columns": False,
        "dataloader_num_workers": 8,
        "local_rank": local_rank,
        "dataloader_pin_memory": True,
        "optim": "adamw_torch_fused",
        "ignore_data_skip": True,
        "dataloader_prefetch_factor": 4,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.1
    }

    # Add distributed training specific arguments
    if is_distributed:
        base_training_args.update({
            "deepspeed": "ds_config.json",
            "ddp_find_unused_parameters": False
        })

    # Initialize trainer based on method
    if method == "dpo":
        training_args = DPOConfig(**base_training_args)
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
            peft_config=lora_config
        )
    elif method == "orpo":
        training_args = ORPOConfig(
            **base_training_args
        )
        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor,
            peft_config=lora_config
        )
    else:  # grpo
        training_args = GRPOConfig(
            **base_training_args
        )
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=processor
        )
    
    # Train
    trainer.train()
    
    # Save the model (only on main process if distributed)
    if not is_distributed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        trainer.model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        trainer.save_state()
        print(f"Model and processor saved to {output_dir}")
    
    # Cleanup distributed training
    cleanup()

if __name__ == "__main__":
    main() 