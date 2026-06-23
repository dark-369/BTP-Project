
import torch
import evaluate
import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login
from tqdm import tqdm
import os

print("="*60)
print("SARVAM-1 English to Hindi Fine Tuning")
print("="*60)

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
# IMPORTANT: Replace with your actual Hugging Face Token
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

hf_repo_name = "hdahiya/sarvam-1-hindi-translator-optimal"
local_save_path = "./sarvam-translator-optimal"
report_path = "./BTech_Sarvam_Optimal_Report.txt"

# ==========================================
# 2. DATA LOADING (100k Train, 1k Eval)
# ==========================================
print("Downloading Datasets...")
# Training Set: 100,000 sentences
dataset_train = load_dataset("ai4bharat/samanantar", "hi", split="train[:100000]")
# Evaluation Set: 1,000 unseen sentences for practice tests & final metrics
dataset_eval = load_dataset("ai4bharat/samanantar", "hi", split="train[100000:101000]")

# ==========================================
# 3. MODEL & TOKENIZER INITIALIZATION
# ==========================================
model_id = "sarvamai/sarvam-1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Sarvam-1 into VRAM...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Expanded Brain Capacity for Maximum Accuracy
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
def format_prompt(example):
    return f"Translate English to Hindi.\nEnglish: {example['src']}\nHindi: {example['tgt']}{tokenizer.eos_token}"

args = SFTConfig(
    output_dir=local_save_path,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3, 
    lr_scheduler_type="cosine", 
    warmup_ratio=0.05, 
    
    # --- THE SIMULTANEOUS EVAL & EARLY STOPPING LOGIC ---
    eval_strategy="steps",      # Run a practice test every 500 steps
    eval_steps=500,             
    save_strategy="steps",      # Save a checkpoint every 500 steps
    save_steps=500,
    load_best_model_at_end=True,       # MAGIC: At the end, load the best weights!
    metric_for_best_model="eval_loss", # Track lowest validation loss
    greater_is_better=False,           # Lower loss = better
    # ----------------------------------------------------
    
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
    eval_dataset=dataset_eval, # <-- Added so the eval logic has data to test on
    peft_config=peft_config, 
    processing_class=tokenizer, 
    formatting_func=format_prompt,
    args=args, 
)

print("\n Starting Training & Simultaneous Evaluation...")
trainer.train()

# This guarantees the saved model is the absolute best checkpoint, not just the last one
print("Saving the BEST performing weights...")
trainer.model.save_pretrained(local_save_path)
trainer.push_to_hub("Optimal Weights - Lowest Eval Loss")

# ==========================================
# 5. FINAL METRICS ON 1,000 SENTENCES
# ==========================================
print("\n Running Final BLEU, chrF, and METEOR calculation on 1,000 sentences...")
trainer.model.eval()

# Load all three metrics
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
meteor_metric = evaluate.load("meteor") # <-- Added METEOR

predictions = []
references = []

for i in tqdm(range(len(dataset_eval))):
    english_text = dataset_eval[i]['src'] 
    ground_truth = dataset_eval[i]['tgt'] 
    
    prompt = f"Translate English to Hindi.\nEnglish: {english_text}\nHindi:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = trainer.model.generate(
            **inputs, 
            max_new_tokens=80, 
            temperature=0.3, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = generated_text.split("Hindi:")[-1].strip()
    
    predictions.append(translation)
    references.append([ground_truth]) 

# Calculate Standard Scores
bleu_score = bleu_metric.compute(predictions=predictions, references=references)['score']
chrf_score = chrf_metric.compute(predictions=predictions, references=references)['score']

# Calculate METEOR (Requires a flat list of references)
flat_references = [ref[0] for ref in references]
meteor_score = meteor_metric.compute(predictions=predictions, references=flat_references)['meteor']

report_content = f"""==================================================
B.TECH PROJECT: SARVAM-1 OPTIMAL RUN (EVAL-TRACKED)
==================================================
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Hardware: PARAM Rudra
Training Data: 100,000 sentences
Validation Data: 1,000 sentences
Checkpoint Strategy: Lowest Eval Loss (load_best_model_at_end=True)

1. QUANTITATIVE METRICS (Evaluated on the 1,000 test set)
--------------------------------------------------
Final BLEU Score:   {bleu_score:.2f}
Final chrF Score:   {chrf_score:.2f}
Final METEOR Score: {(meteor_score * 100):.2f}
==================================================
"""

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_content)

print(f"\n Pipeline Complete!")
print(f" Report saved locally to: {report_path}")
print(f" Final BLEU Score on 1,000 sentences: {bleu_score:.2f}")
