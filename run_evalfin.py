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
print("B.TECH PROJECT: FINAL 1:1 COMPARATIVE EVALUATION (IN22-GEN)")
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

# Load the IN22-Gen Benchmark (English to Hindi)
print("Loading AI4Bharat IN22 Benchmark...")
dataset_eval = load_dataset("ai4bharat/IN22-Gen", "default", split="test")
print(f"Successfully loaded {len(dataset_eval)} benchmark sentences.")

# ==========================================
# 2. HARDCODED BF16 RESULTS & PARAM-1 CONFIG
# ==========================================
# Hardcoding your pristine pure BF16 results for the final report
results = {
    "Pragna-1B": {
        "BLEU": 8.64,
        "chrF": 32.39,
        "COMET": 63.62
    },
    "Sarvam-1": {
        "BLEU": 19.92,
        "chrF": 47.67,
        "COMET": 77.02
    }
}

# We are ONLY evaluating your new Param-1 Control weights
models_to_test = [
    {
        "name": "Param-1",
        "base_id": "./param-1-fixed", 
        "adapter_id": "hdahiya/param-1-hindi-translator-bf16-control", # Your new optimal control weights
        "trust_remote_code": True
    }
]

# ==========================================
# 3. EVALUATION LOOP
# ==========================================
for config in models_to_test:
    model_name = config["name"]
    print(f"\n" + "="*40)
    print(f" EVALUATING: {model_name} (Native BF16 Control)")
    print("="*40)

    # 3A. Load Tokenizer & Base Model Natively in BFloat16
    print(f"Loading {model_name} Tokenizer & Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_id"], trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_id"],
        torch_dtype=torch.bfloat16,
        device_map={"": 0}, # Hardcoded to GPU 0 to prevent memory spread
        trust_remote_code=config["trust_remote_code"]
    )

    # 3B. Attach Your BF16 Control Weights (LoRA)
    print(f"Attaching {model_name} Optimal Adapters...")
    model = PeftModel.from_pretrained(base_model, config["adapter_id"])
    model.eval()

    # 3C. Run Inference
    predictions = []
    references = []
    sources = [] 
    
    print(f"Running Inference on {len(dataset_eval)} IN22 sentences for {model_name}...")
    for i in tqdm(range(len(dataset_eval))):
        english_text = dataset_eval[i]['eng_Latn'] 
        ground_truth = dataset_eval[i]['hin_Deva']
        
        prompt = f"Translate English to Hindi.\nEnglish: {english_text}\nHindi:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=80, 
                temperature=0.3, 
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False # successfully removed per your control test parameters!
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
print("ALL EVALUATIONS COMPLETE. GENERATING REPORT...")

report_path = "./BTech_Comparative_Report_IN22_Final_Control.txt"
report_content = f"""==================================================
B.TECH PROJECT: FINAL 1:1 ARCHITECTURE COMPARISON
==================================================
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Author: Harsh Dahiya
Hardware: PARAM Rudra (A100 80GB)
Task: English to Hindi Translation
Benchmark Data: IN22-Gen Benchmark
Control Variables: Native BF16, Rank=64, Target_Modules=all-linear

FINAL QUANTITATIVE METRICS:
--------------------------------------------------
"""

# Iterate through all three models to build the final report
for model_name, metrics in results.items():
    report_content += f"\n[{model_name.upper()}]\n"
    report_content += f" - BLEU Score:  {metrics['BLEU']:.2f}\n"
    report_content += f" - chrF Score:  {metrics['chrF']:.2f}\n"
    report_content += f" - COMET Score: {metrics['COMET']:.2f}\n"

report_content += "\n=================================================="

# Save locally
with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_content)

# Push to Hugging Face
print("Pushing Report to Hugging Face...")
api = HfApi()
hf_report_repo = "hdahiya/BTech-Translation-Metrics"

create_repo(repo_id=hf_report_repo, repo_type="dataset", exist_ok=True, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj=report_path,
    path_in_repo="BTech_Comparative_Report_IN22_Final_Control.txt",
    repo_id=hf_report_repo,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"\nSUCCESS! Final 1:1 control metrics have been saved locally as '{report_path}'.")
print(f"You can view your published report anytime at: https://huggingface.co/datasets/{hf_report_repo}")