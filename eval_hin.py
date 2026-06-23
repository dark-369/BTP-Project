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
print("B.TECH PROJECT: COMPREHENSIVE 3-MODEL EVALUATION (HINDI TO ENGLISH)")
print("="*60)

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

# Load Metrics with Retry Logic
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

# Load the IN22-Gen Benchmark (Hindi to English)
print("Loading AI4Bharat IN22 Benchmark...")
dataset_eval = load_dataset("ai4bharat/IN22-Gen", "default", split="test")
print(f"Successfully loaded {len(dataset_eval)} benchmark sentences.")

# ==========================================
# 2. MODEL CONFIGURATIONS
# ==========================================
# PARAM-1 EVALUATED FIRST to immediately catch architectural crashes
models_to_test = [
    {
        "name": "Param-1",
        "base_id": "./param-1-fixed", 
        "adapter_id": "hdahiya/param-1-english-translator-bf16-control", 
        "trust_remote_code": True,
        "needs_cache_hack": True  # Crucial for Param-1 architecture
    },
    {
        "name": "Pragna-1B",
        "base_id": "soketlabs/pragna-1b",
        "adapter_id": "hdahiya/pragna-1b-hin-eng-translator-bf16", 
        "trust_remote_code": False,
        "needs_cache_hack": False
    },
    {
        "name": "Sarvam-1",
        "base_id": "sarvamai/sarvam-1",
        "adapter_id": "hdahiya/sarvam-1-hin-eng-translator-bf16", 
        "trust_remote_code": False,
        "needs_cache_hack": False
    }
]

results = {}

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
for config in models_to_test:
    model_name = config["name"]
    print(f"\n" + "="*40)
    print(f" EVALUATING: {model_name} (Hindi -> English)")
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
    
    print(f"Running Inference on {len(dataset_eval)} IN22 sentences for {model_name}...")
    for i in tqdm(range(len(dataset_eval))):
        
        # REVERSED FOR HINDI -> ENGLISH
        hindi_text = dataset_eval[i]['hin_Deva'] 
        ground_truth_english = dataset_eval[i]['eng_Latn']
        
        prompt = f"Translate Hindi to English.\nHindi: {hindi_text}\nEnglish:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        # Generation configuration
        gen_kwargs = {
            "max_new_tokens": 80, 
            "temperature": 0.3, 
            "pad_token_id": tokenizer.eos_token_id
        }
        
        # Apply the hack only if it's Param-1
        if config["needs_cache_hack"]:
            gen_kwargs["use_cache"] = False
            
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = generated_text.split("English:")[-1].strip()
        
        predictions.append(translation)
        references.append([ground_truth_english])
        sources.append(hindi_text) # Source is now Hindi

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
    # COMET requires predictions, references, and sources (which are Hindi here)
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
print("ALL EVALUATIONS COMPLETE. GENERATING REPORT...")

report_path = "./BTech_Comparative_Report_IN22_Hin_Eng.txt"
report_content = f"""==================================================
B.TECH PROJECT: FINAL ARCHITECTURE COMPARISON (HINDI TO ENGLISH)
==================================================
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Author: Harsh Dahiya
Hardware: PARAM Rudra (A100 80GB)
Task: Hindi to English Translation
Benchmark Data: IN22-Gen Benchmark
Control Variables: Native BF16, Rank=64, Target_Modules=all-linear

FINAL QUANTITATIVE METRICS (Hindi -> English):
--------------------------------------------------
"""

for model_name, metrics in results.items():
    report_content += f"\n[{model_name.upper()}]\n"
    report_content += f" - BLEU Score:  {metrics['BLEU']:.2f}\n"
    report_content += f" - chrF Score:  {metrics['chrF']:.2f}\n"
    report_content += f" - COMET Score: {metrics['COMET']:.2f}\n"

report_content += "\n=================================================="

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_content)

print("Pushing Report to Hugging Face...")
api = HfApi()
hf_report_repo = "hdahiya/BTech-Translation-Metrics"

create_repo(repo_id=hf_report_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=report_path,
    path_in_repo="BTech_Comparative_Report_IN22_Hin_Eng.txt",
    repo_id=hf_report_repo,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"\nSUCCESS! Hindi-to-English metrics have been saved locally as '{report_path}'.")
print(f"You can view your published report anytime at: https://huggingface.co/datasets/{hf_report_repo}")