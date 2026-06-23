import torch
import evaluate
import datetime
import gc
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login, HfApi, create_repo

# Author: Harsh Dahiya

print("="*60)
print("B.TECH PROJECT: COMPREHENSIVE 3-MODEL EVALUATION (IN22-GEN)")
print("="*60)

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
print(" --------- Logged into Hugging Face successfully! --------------")

# Load Metrics
print("Loading Evaluation Metrics...")
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
comet_metric = evaluate.load("comet") 

# Load the IN22-Gen Benchmark for English to Hindi
print("Loading AI4Bharat IN22 Benchmark...")
dataset_eval = load_dataset("ai4bharat/IN22-Gen", "default", split="test")
print(f"Successfully loaded {len(dataset_eval)} benchmark sentences.")

# ==========================================
# 2. MODEL CONFIGURATIONS
# ==========================================
# Reordered to evaluate Param-1 FIRST to catch any architectural bugs immediately.
models_to_test = [
    {
        "name": "Param-1",
        "base_id": "./param-1-fixed", 
        "adapter_id": "hdahiya/param-1-hindi-translator-optimal",
        "use_4bit": False, 
        "trust_remote_code": True
    },
    {
        "name": "Pragna-1B",
        "base_id": "soketlabs/pragna-1b",
        "adapter_id": "hdahiya/pragna-1b-hindi-translator-optimal",
        "use_4bit": False, 
        "trust_remote_code": False
    },
    {
        "name": "Sarvam-1",
        "base_id": "sarvamai/sarvam-1",
        "adapter_id": "hdahiya/sarvam-1-hindi-translator-optimal",
        "use_4bit": False, 
        "trust_remote_code": False
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
    
    # 3A. Setup Precision
    bnb_config = None
    if config["use_4bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 3B. Load Tokenizer & Base Model
    print(f"Loading {model_name} Tokenizer & Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_id"], trust_remote_code=config["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_id"],
        quantization_config=bnb_config if config["use_4bit"] else None,
        torch_dtype=torch.bfloat16 if not config["use_4bit"] else None,
        device_map="auto",
        trust_remote_code=config["trust_remote_code"]
    )

    # 3C. Attach Your Optimal Weights (LoRA)
    print(f"Attaching {model_name} Optimal Adapters...")
    model = PeftModel.from_pretrained(base_model, config["adapter_id"])
    model.eval()

    # 3D. Run Inference
    predictions = []
    references = []
    sources = [] 
    
    print(f"Running Inference on {len(dataset_eval)} IN22 sentences for {model_name}...")
    for i in tqdm(range(len(dataset_eval))):
        # Using the specific keys provided by the IN22-Gen dataset schema
        english_text = dataset_eval[i]['eng_Latn'] 
        ground_truth = dataset_eval[i]['hin_Deva']
        
        prompt = f"Translate English to Hindi.\nEnglish: {english_text}\nHindi:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=80, 
                temperature=0.3, 
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False # THE FIX: Bypasses the DynamicCache bug in Param-1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = generated_text.split("Hindi:")[-1].strip()
        
        predictions.append(translation)
        references.append([ground_truth])
        sources.append(english_text)

    # 3E. AGGRESSIVE MEMORY CLEANUP
    print(f"Wiping {model_name} from VRAM to make room for COMET scoring...")
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 3F. Calculate Scores
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

report_path = "./BTech_Comparative_Report.txt"
report_content = f"""==================================================
B.TECH PROJECT: FINAL ARCHITECTURE COMPARISON (IN22-GEN)
==================================================
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Author: Harsh Dahiya
Hardware: PARAM Rudra (A100 80GB)
Task: English to Hindi Translation
Benchmark Data: IN22-Gen Benchmark

FINAL QUANTITATIVE METRICS:
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
    path_in_repo="BTech_Comparative_Report_IN22.txt",
    repo_id=hf_report_repo,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"\nSUCCESS! Final metrics have been saved and uploaded.")
print(f"You can view your report anytime at: https://huggingface.co/datasets/{hf_report_repo}")