import os
import sys
import re
import torch
import pandas as pd
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from unsloth.chat_templates import apply_chat_template
from datasets import Dataset
from sentence_transformers import CrossEncoder
from transformers import TrainerCallback

steps = 3000
base_model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
output_dir = "llm-grpo-toddler-tiny-1-r3"
downscale_factor = 20.0
lora_rank = 32
max_prompt_length = 96
max_seq_length = max_prompt_length * 3

print("Initializing GRPO Training")
print(f"Steps: {steps}")
print(f"Model: {base_model_id}")
print(f"Output Directory: {output_dir}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    fast_inference=False,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.65,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

training_args = GRPOConfig(
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    per_device_train_batch_size=256,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_prompt_length,
    num_train_epochs=4,
    max_steps=steps,
    report_to="none",
    temperature=0.8,
    reward_weights=[1.0, 1.0 / downscale_factor, 1.0],
)

# Rewards
def compute_child_likeness_reward(prompts, completions, **kwargs) -> list[float]:
    if "<|im_start|>" in prompts[0]:
        prompts = [p.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0] for p in prompts]
    elif "<|start_header_id|>" in prompts[0]:
        prompts = [p.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[-1].split("<|eot_id|>", 1)[0] for p in prompts]
    else:
        raise ValueError("Unknown prompt format.")
    responses = [[c] for c in completions]
    return model_reward.predict(responses, batch_size=128).tolist()

def compute_coherence_score(prompts, completions, **kwargs) -> list[float]:
    if "<|im_start|>" in prompts[0]:
        prompts = [p.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0] for p in prompts]
    elif "<|start_header_id|>" in prompts[0]:
        prompts = [p.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[-1].split("<|eot_id|>", 1)[0] for p in prompts]
    else:
        raise ValueError("Unknown prompt format.")
    pairs = list(zip(prompts, completions))
    scores = model_coherence.predict(pairs, batch_size=128).tolist()
    if torch.rand(1).item() < 0.01:
        print("Example Coherence Score:")
        print(f"Prompt: {prompts[0]}\nCompletion: {completions[0]}\nScore: {scores[0]}")
    return scores

def compute_length_reward(prompts, completions, **kwargs) -> list[float]:
    if "<|im_start|>" in prompts[0]:
        prompts = [p.split("<|im_start|>user\n", 1)[-1].split("<|im_end|>", 1)[0] for p in prompts]
    elif "<|start_header_id|>" in prompts[0]:
        prompts = [p.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[-1].split("<|eot_id|>", 1)[0] for p in prompts]
    elif "<start_of_turn>" in prompts[0]:
        prompts = [p.split("\n\n", 1)[-1].split("<end_of_turn>\n", 1)[0] for p in prompts]
    else:
        raise ValueError("Unknown prompt format.")
    scores = []
    for completion in completions:
        words = completion.strip().split()
        punct = len(re.findall(r'[.!?,;:]', completion))
        base = 1.0 if 2 <= len(words) <= 7 else 0.5
        multiplier = 1.0 if punct <= 1 else 1.0 / punct
        scores.append(base * multiplier)
    return scores

class CustomLogger(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs["logs"]
        entry = {
            "Step": state.global_step,
            "Loss": logs.get("training_loss"),
            "Reward": logs.get("reward"),
            "Reward Std": logs.get("reward_std"),
            "KL": logs.get("kl"),
            "Child Reward": logs.get("rewards/compute_reward_model_score"),
            "Coherence Reward": logs.get("rewards/compute_reward_model_score2"),
            "Length Reward": logs.get("rewards/length_reward"),
        }
        self.history.append(entry)

        save_checkpoints = [500, 1000, 1500, 2000, 3000, 5000]
        if state.global_step in save_checkpoints:
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained_merged(
                f"{output_dir}/step-{state.global_step}",
                tokenizer,
                save_method="merged_16bit",
                maximum_memory_usage=0.8,
            )

custom_logger = CustomLogger()

print("\nLoading dataset...")
data = pd.read_csv('path/to/child_response_pairs')
data = data.drop_duplicates(subset=['text'])
data.text = data.text.str.strip()
data = data[data.text.str.endswith('?')]
data = data[data.text.str.len() > 40]
data = data.rename(columns={'text': 'prompt'})
data = data.reset_index(drop=True)

print(f"Total prompts: {len(data)}")
if "Llama" in base_model_id:
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    tokenizer.chat_template = tokenizer.chat_template.replace(
        """{{- "Cutting Knowledge Date: December 2023\n" }}\n{{- "Today Date: " + date_string + "\n\n" }}""", "")
    data['prompt'] = data['prompt'].apply(lambda x: tokenizer.apply_chat_template(
        [{'role': 'system', 'content': "Answer the dialogue question in a short and concise manner."},
         {'role': 'user', 'content': x}], tokenize=False) + "<|start_header_id|>user<|end_header_id|>\n\n")
else:
    data['prompt'] = data['prompt'].apply(lambda x: tokenizer.apply_chat_template(
        [{'role': 'system', 'content': "Answer the dialogue question in a short and concise manner."},
         {'role': 'user', 'content': x}], tokenize=False) + "assistant\n")

dataset = Dataset.from_pandas(data)

print("Loading reward models...")
model_reward = CrossEncoder("path/to/childlikeness_reward", device='cuda', max_length=256)
model_coherence = CrossEncoder("path/to/coherence_reward", device='cuda', max_length=356)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        compute_child_likeness_reward,
        compute_coherence_score,
        compute_length_reward,
    ],
    args=training_args,
    train_dataset=dataset,
)
trainer.add_callback(custom_logger)

print("\nStarting training...\n")
results = trainer.train()


