import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# MODEL_PATH = "Upstage/SOLAR-10.7B-Instruct-v1.0"
MODEL_PATH = "kyujinpy/Sakura-SOLAR-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
)
max_length = 4096

f = open('hs_text.txt', 'r')
wrong_text = f.read()
f.close()

def prompting_n_tokenize(wrong_text, tokenizer):
    conversation = [ {'role': 'user', 'content': f'오타가 있는 문장이 아래 문장인데, 이 문장을 오타를 수정해서 완전한 문장으로 다시 작성해줘. 오타가 있는 부분만 수정하고 나머지 부분은 건들지 말아줘. : \n###텍스트: {wrong_text}'} ] 
    # print('conversation :',conversation)
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    # print('prompt :',prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # print('inputs :',inputs)

    return inputs

def fix_sentence(inputs, tokenizer, model, max_length):
    outputs = model.generate(**inputs, use_cache=True, max_length=max_length)
    # print('outputs :',outputs)
    output_text = tokenizer.decode(outputs[0]) 
    # print('output_text :',output_text)
    return output_text

total_output_text2 = ''
if len(wrong_text) > 2000:
    chunks = [wrong_text[i:i+2000] for i in range(0, len(wrong_text), 2000)]
    for chunk in chunks:
        chunk_inputs = prompting_n_tokenize(chunk, wrong_text, tokenizer)
        total_output_text2 += fix_sentence(chunk_inputs, tokenizer, model, max_length)
chunk_inputs2 = prompting_n_tokenize(wrong_text, tokenizer)
total_output_text2 += fix_sentence(chunk_inputs2, tokenizer, model, max_length)
f = open('hs_after.txt', 'w')
f.write(total_output_text2)
f.close()