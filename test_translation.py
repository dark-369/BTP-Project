import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import os

print("="*60)
print("LIVE INFERENCE: SARVAM-1 ENGLISH TO HINDI")
print("="*60)

model_id = "sarvamai/sarvam-1"
local_save_path = "./sarvam-translator-optimal"
output_file = "./BTech_Sarvam_Translations.txt"

print("Loading tokenizer and optimal model weights into VRAM...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    local_save_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

test_sentences = [
    "I am very happy to see the successful completion of my project.",
    "Artificial intelligence and machine learning are transforming the future of technology.",
    "Please let me know if you need any further assistance with this task.",
    "The weather is incredibly beautiful today, let's go outside for a walk.",
    "To create a new session in Tmux, you can use a simple terminal command."
]

stored_results = []
report_header = f"""==================================================
SARVAM-1 FINE-TUNED TRANSLATION SAMPLES
Date Completed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Hardware: PARAM Rudra
==================================================
"""

print("\nTranslating sentences...\n")
for idx, sentence in enumerate(test_sentences):
    prompt = f"Translate English to Hindi.\nEnglish: {sentence}\nHindi:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=80, 
            temperature=0.3, 
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translation = generated_text.split("Hindi:")[-1].strip()
    
    result_string = f"Sentence {idx+1}:\nEnglish: {sentence}\nHindi:   {translation}\n{'-'*50}"
    stored_results.append(result_string)
    print(result_string)

with open(output_file, "w", encoding="utf-8") as file:
    file.write(report_header)
    file.write("\n\n".join(stored_results))

print(f"\nSuccess! All 5 translations have been saved locally to: {output_file}")
