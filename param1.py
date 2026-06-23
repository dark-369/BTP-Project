import torch
import os
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login, snapshot_download

# Author: Harsh Dahiya

print("="*60)
print("PARAM-1: English to Hindi Training (PURE BFLOAT16)")
print("="*60)

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

hf_repo_name = "hdahiya/param-1-hindi-translator-optimal"
local_save_path = "./param-translator-optimal"

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Downloading Datasets...")
dataset_train = load_dataset("ai4bharat/samanantar", "hi", split="train[:100000]")
dataset_eval = load_dataset("ai4bharat/samanantar", "hi", split="train[100000:101000]")

# ==========================================
# 3. LOCAL MODEL SETUdqP
# ==========================================
model_id = "bharatgenai/Param-1"
local_model_dir = "./param-1-fixed"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

# ==========================================
# 4. LOAD MODEL IN PURE BFLOAT16 (NO 4-BIT)
# ==========================================
print("Loading BharatGen Param-1 into VRAM natively...")

# NO BitsAndBytesConfig. Loading in pure bf16 natively for the A100.
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True 
)

model.config.use_cache = False

# ==========================================
# 5. STANDARD LORA CONFIGURATION
# ==========================================
peft_config = LoraConfig(
    r=32,          
    lora_alpha=64, 
    # Since we aren't quantizing, we can safely target all core layers again
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 6. TRAINING CONFIG
# ==========================================
def format_prompt(example):
    return f"Translate English to Hindi.\nEnglish: {example['src']}\nHindi: {example['tgt']}{tokenizer.eos_token}"

args = SFTConfig(
    output_dir=local_save_path,
    per_device_train_batch_size=4,      
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      
    gradient_checkpointing=True,        
    
    max_grad_norm=0.3,           
    learning_rate=5e-5, # Back to a healthy learning rate
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
    max_length=256,
    
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

print("\n Starting HPC Training...")
trainer.train()

print("Saving the BEST performing weights...")
trainer.model.save_pretrained(local_save_path)
trainer.push_to_hub("Optimal Weights - Lowest Eval Loss")

print("\n Training Complete! Optimal weights saved.")