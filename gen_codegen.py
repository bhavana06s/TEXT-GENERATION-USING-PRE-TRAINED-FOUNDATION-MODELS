from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
def generate_code(prompt, max_new_tokens=80):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)
#1.Codegenerationfromanatural-languageinstruction
prompt1="#WriteaPythonfunctiontocheckifanumberisprime\ndefis_prime(n):"
print("GeneratedFunction:\n",generate_code(prompt1))
#2.Debuggingafaultysnippet
buggy_code="""#Thefollowingfunctionshouldreturnthefactorialofn,buthasabug.Fixit.
deffactorial(n):
result=0
foriinrange(1,n+1):
result=result*i
returnresult
#Correctedfunction:
deffactorial_fixed(n):"""
print("\nDebugSuggestion:\n",generate_code(buggy_code,max_new_tokens=60))
