#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForPreTraining, PaliGemmaProcessor, LlavaNextProcessor, LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, LlavaForConditionalGeneration
import json
import argparse
from PIL import Image as PILImage
import re
from tqdm import tqdm

def paligemma_inference(question, variants, image, model, processor):
    text = generate_prompt_text(question, variants)
    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
            
    # Convert inputs to correct dtype
    inputs = {k: v.to(dtype=model.dtype) if torch.is_floating_point(v) else v 
                for k, v in inputs.items()}
            
    # Generate answer with explicit parameters
    input_len = inputs["input_ids"].shape[-1]
    generation = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True
    )
    answer = processor.decode(generation[0][input_len:], skip_special_tokens=True)

    return answer

def llava_inference(question, variants, image, model, processor):
    text = generate_prompt_text(question, variants)

    conversation = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": text},
              {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    model_inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    model_inputs = model_inputs.to(dtype=model.dtype)

    output = model.generate(**model_inputs, max_new_tokens=100)
    decoded_output = processor.decode(output[0], skip_special_tokens=True)

    answer = extract_answer(decoded_output, delimiter='[/INST]')

    return answer

def qwen_inference(question, variants, image, model, processor):
    text = generate_prompt_text(question, variants)

    conversation = [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": text},
              {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    model_inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    model_inputs = model_inputs.to(dtype=model.dtype)

    output = model.generate(**model_inputs, max_new_tokens=50)
    decoded_output = processor.decode(output[0], skip_special_tokens=True).split(processor.tokenizer.eos_token)[-1]
    answer = extract_answer(decoded_output, delimiter="assistant")

    return answer

def llama_inference(question, variants, image, model, processor):
    text = generate_prompt_text(question, variants)

    prompt = f"<|image|><|begin_of_text|>{text}"

    model_inputs = processor(image, prompt, return_tensors="pt").to(model.device)

    output = model.generate(**model_inputs, max_new_tokens=30)
    decoded_output = processor.decode(output[0])

    answer = extract_answer(decoded_output, delimiter="Відповідь:")
    return answer

def pixtral_inference(question, variants, image, model, processor):
    text = generate_prompt_text(question, variants)

    prompt = f"<s>[INST]{text}\n[IMG][/INST]"

    model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    model_inputs = model_inputs.to(dtype=model.dtype)

    generate_ids = model.generate(**model_inputs, max_new_tokens=500)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = extract_answer(output, 'Відповідь:')

    return answer

model_config = {
    "paligemma": {
        "model_id": "google/paligemma-3b-mix-224", 
        "model_class": AutoModelForPreTraining,
        "processor_class": PaliGemmaProcessor,
        "inference_method": paligemma_inference
    },
    "llava": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor,
        "inference_method": llava_inference
    },
    "qwen": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
        "processor_class": AutoProcessor,
        "inference_method": qwen_inference
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "model_class": MllamaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "inference_method": llama_inference
    },
    "pixtral": {
        "model_id": "mistral-community/pixtral-12b",
        "model_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "inference_method": pixtral_inference
    }
}

def setup():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

def extract_answer(answer, delimiter=None):
    LAT_TO_CYR = {
        'A': 'А',
        'B': 'Б',
        'C': 'В',
        'D': 'Г'
    }
    if delimiter and delimiter in answer:
        answer = answer.split(delimiter)[-1].strip()
    re_match = re.search(r'\b[ABCDАБВГ]\b', answer.upper())
    raw_marker = re_match.group(0) if re_match else ''
    marker = LAT_TO_CYR.get(raw_marker, raw_marker)
    return marker

def extract_answer_type(options, marker_to_find):
    for item in options:
        if item["marker"] == marker_to_find:
            return item["type"]
    return ""

def generate_prompt_text(question, variants):
    options = '\n'.join([f"{option['marker']}) {option['text']}" for option in variants])
    text = (
        f"<image>\n"
        f"Питання: {question}\n"
        f"Варіанти відповіді:\n{options}\n\n"
        "Обери правильну відповідь. Напиши тільки одну відповідь, використовуючи українську літеру: А, Б, В або Г.\n"
        "Додатково запиши відповідь під цією буквою.\n"
        "Відповідь:"
    )
    return text


def test_model(model_name, model, processor, test_data, local_rank, world_size):
    model.eval()
    results = []
    
    # Calculate start and end indices for this GPU
    total_samples = len(test_data)
    samples_per_gpu = total_samples // world_size
    start_idx = local_rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if local_rank != world_size - 1 else total_samples
    
    # Get this GPU's portion of the data
    gpu_test_data = test_data[start_idx:end_idx]
    
    with torch.no_grad():
        for entry in tqdm(gpu_test_data, desc=f"GPU processing"):
            image_url = entry['image_url']
                
            image = PILImage.open(image_url)
            question = entry['question']

            # Prepare input with explicit max length
            answer = model_config[model_name]['inference_method'](question['text'], question['choices'], image, model, processor)
            
            # Extract answer letter and type
            answer_letter = extract_answer(answer)
            answer_type = extract_answer_type(question['choices'], answer_letter) if question['task_type'] == 'awareness' else ""

            # Prepare result entry
            result = {
                "image": image_url,
                "image_id": entry.get('image_id', ''),
                "question": question['text'],
                "question_id": question.get('id', ''),
                "type": question.get('type', ''),
                "task_type": question['task_type'],
                "correct-answer": question['correct_answer']['text'] if question['task_type'] == 'general' else "",
                "answer": answer,
                "correct-answer-letter": question['correct_answer']['marker'] if question['task_type'] == 'general' else "",
                "answer-letter": answer_letter,
                "answer-type": answer_type,
                "gpu_id": local_rank
            }
            results.append(result)
    
    # Gather results from all GPUs
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, results)
    
    # Flatten results if on rank 0
    if local_rank == 0:
        flattened_results = []
        for res in all_results:
            if res:  # Check if results exist
                flattened_results.extend(res)
        print(f"Total results gathered: {len(flattened_results)}")
        return flattened_results
    return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test vision-language models on the CulturUA dataset')
    parser.add_argument('--model_name', type=str, required=True, choices=['paligemma', 'llava', 'qwen', 'llama', 'pixtral'],
                        help='Name of the model to test')
    parser.add_argument('--method', type=str, choices=['dpo', 'orpo', 'grpo'],
                        help='Method to run (e.g., dpo, orpo, grpo)')
    args = parser.parse_args()

    model_name = args.model_name
    method = args.method

    # Setup distributed testing
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Load model configuration
    config = model_config[model_name]
    
    # Initialize model and processor
    processor = config["processor_class"].from_pretrained(config["model_id"])
    model = config["model_class"].from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Load test data
    with open("culturUA/test_set.jsonl", 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    # Run test
    results = test_model(model_name, model, processor, test_data, local_rank, world_size)

    # Save results if this is the main process
    if local_rank == 0:
        output_file = f"results_{model_name}_{method}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")

    cleanup()

if __name__ == "__main__":
    main() 