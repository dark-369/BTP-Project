import torch
import evaluate
import datetime
import gc
import time
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login, HfApi, create_repo

# Author: Harsh Dahiya

print("="*60)
print("B.TECH PROJECT: PARAM-1 DIAGNOSTIC EVALUATION (SAMANANTAR)")
print("="*60)

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

# Load Metrics with HPC Network Retry Logic
print("Loading Evaluation Metrics...")
def load_metric_with_retry(metric_name, retries=5):
    for attempt in range(retries):
        try:
            return evaluate.load(metric_name)
        except Exception as e:
            print(f"Network timeout loading {metric_name}. Retrying ({attempt+1}/{retries})...")
            time.sleep(5)
    raise Exception(f"Failed to load {metric_name} after {retries} attempts.")

bleu_metric = load_metric_with_retry("sacrebleu")
chrf_metric = load_metric_with_retry("chrf")
comet_metric = load_metric_with_retry("comet") 

# Load the Samanantar Evaluation Split (1,000 Unseen Sentences)
print("Loading Samanantar 1k Evaluation Split...")
dataset_eval = load_dataset("ai4bharat/samanantar", "hi", split="train[100000:101000]")
print(f"Successfully loaded {len(dataset_eval)} Samanantar sentences.")

# ==========================================
# 2. MODEL CONFIGURATION (PARAM-1 ONLY)
# ==========================================
models_to_test = [
    {
        "name": "Param-1",
        "base_id": "./param-1-fixed", 
        "adapter_id": "hdahiya/param-1-hindi-translator-optimal",
        "trust_remote_code": True
    }
]

results = {}

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
for config in models_to_test:
    model_name = config["name"]
    print(f"\n" + "="*40)
    print(f" EVALUATING: {model_name}")
    print("="*40)
    
    # 3A. Load Tokenizer & Base Model Natively in BFloat16
    print(f"Loading {model_name} Tokenizer & Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_id"], trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_id"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0}, # Hardcoded to GPU 0
        trust_remote_code=config["trust_remote_code"]
    )

    # 3B. Attach Your Optimal Weights (LoRA)
    print(f"Attaching {model_name} Optimal Adapters...")
    model = PeftModel.from_pretrained(base_model, config["adapter_id"])
    model.eval()

    # 3C. Run Inference
    predictions = []
    references = []
    sources = [] 
    
    print(f"Running Inference on {len(dataset_eval)} Samanantar sentences for {model_name}...")
    
    for i in tqdm(range(len(dataset_eval))):
        # Samanantar uses simple 'src' and 'tgt' keys
        english_text = dataset_eval[i]['src'] 
        ground_truth = dataset_eval[i]['tgt'] 
        
        prompt = f"Translate English to Hindi.\nEnglish: {english_text}\nHindi:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=80, 
                temperature=0.3, 
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False # Crucial for Param-1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = generated_text.split("Hindi:")[-1].strip()
        
        predictions.append(translation)
        references.append([ground_truth])
        sources.append(english_text)

    # 3D. AGGRESSIVE MEMORY CLEANUP
    print(f"Wiping {model_name} from VRAM to make room for COMET scoring...")
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 3E. Calculate Scores
    print(f"Calculating Metrics for {model_name}...")
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)['score']
    chrf_score = chrf_metric.compute(predictions=predictions, references=references)['score']
    
    flat_references = [ref[0] for ref in references]
    comet_results = comet_metric.compute(predictions=predictions, references=flat_references, sources=sources)
    comet_score = comet_results['mean_score'] * 100 
    
    results[model_name] = {
        "BLEU": bleu_score,
        "chrF": chrf_score,
        "COMET": comet_score
    }
    
    print(f"{model_name} Results -> BLEU: {bleu_score:.2f} | chrF: {chrf_score:.2f} | COMET: {comet_score:.2f}")

# ==========================================
# 4. GENERATE REPORT & PUSH TO HUGGING FACE
# ==========================================
print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE. GENERATING REPORT...")

report_path = "./Param1_Samanantar_Diagnostic.txt"
report_content = f"""==================================================
B.TECH PROJECT: PARAM-1 DIAGNOSTIC (SAMANANTAR)
==================================================
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Hardware: PARAM Rudra (A100 80GB)
Task: English to Hindi Translation
Benchmark Data: Samanantar 1k Evaluation Split

PARAM-1 METRICS (Native BF16):
--------------------------------------------------
 BLEU Score:  {results['Param-1']['BLEU']:.2f}
 chrF Score:  {results['Param-1']['chrF']:.2f}
 COMET Score: {results['Param-1']['COMET']:.2f}
==================================================
"""

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_content)

print("Pushing Diagnostic Report to Hugging Face...")
api = HfApi()
hf_report_repo = "hdahiya/BTech-Translation-Metrics"

create_repo(repo_id=hf_report_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=report_path,
    path_in_repo="Param1_Samanantar_Diagnostic.txt",
    repo_id=hf_report_repo,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"\nSUCCESS! Diagnostic metrics have been saved and uploaded.")