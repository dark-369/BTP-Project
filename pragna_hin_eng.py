import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Author: Harsh Dahiya

print("="*60)
print("PRAGNA-1B: Hindi to English Translation (PURE BFLOAT16)")
print("="*60)

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

# NEW SAVING PATHS FOR HINDI-TO-ENGLISH TO PREVENT OVERWRITING
hf_repo_name = "hdahiya/pragna-1b-hin-eng-translator-bf16"
local_save_path = "./pragna-hin-eng-translator-bf16"

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

print("Loading Soket AI Pragna-1B into VRAM natively in BFloat16...")
# NO 4-BIT. Loading natively for the A100.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},  # Prevents the multi-GPU DataParallel crash
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
# 4. TRAINING WITH SIMULTANEOUS EVALUATION
# ==========================================
# REVERSED FOR HINDI -> ENGLISH
# In Samanantar: 'tgt' is Hindi, 'src' is English
def format_prompt(example):
    return f"Translate Hindi to English.\nHindi: {example['tgt']}\nEnglish: {example['src']}{tokenizer.eos_token}"

args = SFTConfig(
    output_dir=local_save_path,
    per_device_train_batch_size=4,      
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,      
    gradient_checkpointing=True,        
    learning_rate=2e-4,
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
    max_length=512,
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
trainer.train(resume_from_checkpoint="./pragna-hin-eng-translator-bf16/checkpoint-2000")

print("Saving the BEST performing weights...")
trainer.model.save_pretrained(local_save_path)
trainer.push_to_hub("Optimal Weights (BF16) - Lowest Eval Loss")

print("\n Training Complete! Optimal Hindi-to-English BF16 weights saved locally and to Hugging Face.")