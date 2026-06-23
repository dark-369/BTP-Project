import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Author: Harsh Dahiya

print("="*60)
print("PARAM-1: Hindi to English Fine Tuning ")
print("="*60)

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

# Updated repo to separate these weights from your En->Hi model
hf_repo_name = "hdahiya/param-1-english-translator-bf16-control"
local_save_path = "./param-translator-eng-bf16-control"

# ==========================================
# 2. DATA LOADING (100k Train, 1k Eval)
# ==========================================
print("Downloading Datasets...")
dataset_train = load_dataset("ai4bharat/samanantar", "hi", split="train[:100000]")
dataset_eval = load_dataset("ai4bharat/samanantar", "hi", split="train[100000:101000]")

# ==========================================
# 3. LOCAL MODEL SETUP
# ==========================================
local_model_dir = "./param-1-fixed"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# ==========================================
# 4. LOAD MODEL IN PURE BFLOAT16 (NO CACHE HACK)
# ==========================================
print("Loading BharatGen Param-1 into VRAM natively in BFloat16...")

# The use_cache=False hack is completely removed.
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True 
)

# ==========================================
# 5. EXPANDED LORA CONFIGURATION (Matched to Sarvam)
# ==========================================
peft_config = LoraConfig(
    r=64,           # Increased brain capacity
    lora_alpha=128, # Scaled alpha
    target_modules="all-linear", # Aggressive targeting just like Sarvam
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 6. TRAINING CONFIG (Matched to Sarvam)
# ==========================================
def format_prompt(example):
    # Flipped for Hindi to English direction
    return f"Translate Hindi to English.\nHindi: {example['tgt']}\nEnglish: {example['src']}{tokenizer.eos_token}"

args = SFTConfig(
    output_dir=local_save_path,
    per_device_train_batch_size=8,      # Matched to Sarvam
    per_device_eval_batch_size=8,       # Matched to Sarvam
    gradient_accumulation_steps=4,      # Matched to Sarvam
    gradient_checkpointing=True,        
    
    learning_rate=2e-4,                 # Matched aggressive LR
    num_train_epochs=3, 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.05, 
    
    eval_strategy="steps",      
    eval_steps=500,             
    save_strategy="steps",      
    save_steps=500,
    load_best_model_at_end=True,       
    metric_for_best_model="eval_loss", 
    greater_is_better=False,           
    
    logging_steps=50,
    
    # NATIVE A100 PRECISION
    bf16=True,
    fp16=False,
    
    optim="paged_adamw_32bit",
    max_length=512,                     # Matched expanded context window
    push_to_hub=True, 
    hub_model_id=hf_repo_name,
)

trainer = SFTTrainer(
    model=model, 
    train_dataset=dataset_train,
    eval_dataset=dataset_eval, 
    peft_config=peft_config, 
    processing_class=tokenizer, 
    formatting_func=format_prompt,
    args=args, 
)

print("\n Starting HPC Training & Simultaneous Evaluation...")
# Removed resume_from_checkpoint so it starts fresh for the new task
trainer.train(resume_from_checkpoint="./param-translator-eng-bf16-control/checkpoint-3000")

print("Saving the BEST performing weights...")
trainer.model.save_pretrained(local_save_path)
trainer.push_to_hub("Optimal Weights (BF16 Control) - Lowest Eval Loss")

print("\n Training Complete! Optimal weights saved.")