import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

print("="*60)   
print("PRAGNA-1B English to Hindi Translation")
print("="*60)

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

hf_repo_name = "hdahiya/pragna-1b-hindi-translator-optimal"
local_save_path = "./pragna-translator-optimal"

# ==========================================
# 2. DATA LOADING (100k Train, 1k Eval)
# ==========================================
print("Downloading Datasets...")
dataset_train = load_dataset("ai4bharat/samanantar", "hi", split="train[:100000]")
dataset_eval = load_dataset("ai4bharat/samanantar", "hi", split="train[100000:101000]")

# ==========================================
# 3. MODEL & TOKENIZER INITIALIZATION
# ==========================================
model_id = "soketlabs/pragna-1b"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Soket AI Pragna-1B into VRAM on GPU 0...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},  # <-- Prevents the multi-GPU DataParallel crash!
)

# Maximum Accuracy LoRA Settings
peft_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    target_modules="all-linear", 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 4. TRAINING WITH LOSS TRACKING ONLY
# ==========================================
def format_prompt(example):
    return f"Translate English to Hindi.\nEnglish: {example['src']}\nHindi: {example['tgt']}{tokenizer.eos_token}"

args = SFTConfig(
    output_dir=local_save_path,
    per_device_train_batch_size=4,      # 4 is very safe for a 1.25B model on an 80GB A100
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      # 4 x 8 = 32 effective batch size
    gradient_checkpointing=True,        # Prevents Out of Memory (OOM)
    learning_rate=2e-4,
    num_train_epochs=3, 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.05, 
    
    eval_strategy="steps",      # Tests validation loss every 500 steps
    eval_steps=500,             
    save_strategy="steps",      # Saves checkpoint every 500 steps
    save_steps=500,
    load_best_model_at_end=True,       # Ensures optimal accuracy by keeping the lowest loss weight
    metric_for_best_model="eval_loss", 
    greater_is_better=False,           
    
    logging_steps=50,
    bf16=True,
    optim="paged_adamw_32bit",
    max_length=512,
    push_to_hub=True, 
    hub_model_id=hf_repo_name,
)

trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset_train,
    eval_dataset=dataset_eval, # Tracks eval_loss during training
    peft_config=peft_config, 
    processing_class=tokenizer, 
    formatting_func=format_prompt,
    args=args, 
)

print("\n Starting HPC Training...")
trainer.train()

print("Saving the BEST performing weights...")
trainer.model.save_pretrained(local_save_path)
trainer.push_to_hub("Optimal Weights - Lowest Eval Loss")

print("\n Training Complete! Optimal weights saved locally and to Hugging Face.")
print(" You can now run your separate evaluation script.")