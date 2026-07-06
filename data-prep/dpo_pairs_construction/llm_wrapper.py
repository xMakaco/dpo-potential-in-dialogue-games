"""Defines clients to call LLMs to generate reflection comments.
It contains clients for Azure OpenAI, Deepseek, and Qwen local models.
To replicate the project verbatim, the Azure OpenAI client should be used.
API keys need to be defined in a key.json file stored in the same directory as this script.
"""

import json
import os
import random
import time
from openai import OpenAI, AzureOpenAI

KEY_PATH = os.path.join(os.path.dirname(__file__), "key.json")

class QwenLocalClient:
    def __init__(self, model_id):
        # lazy imports to keep the API-only clients importable even when
        # torch can't be loaded (e.g. host under memory pressure)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
            trust_remote_code=True
        ).to(self.device)

    def generate_comment(self, messages, max_new_tokens=512, temperature=0.2):
        import torch
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
class GptAzureClient:
    def __init__(self, model_id, path_to_key=KEY_PATH):
        self.model_id = model_id

        with open(path_to_key, 'r') as f:
            key = json.load(f)

        self.client = AzureOpenAI(
            api_key = key["azure_openai_compatible"]["api_key"],
            azure_endpoint = key["azure_openai_compatible"]["base_url"],
            api_version="2024-12-01-preview"
        )

   
    def generate_comment(self, messages, max_completion_tokens=512, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    time.sleep(wait)
                else:
                    raise

class DeepseekClient:
    def __init__(self, model_id, path_to_key=KEY_PATH):
        self.model_id = model_id

        with open(path_to_key, 'r') as f:
            key = json.load(f)

        self.client = OpenAI(
            api_key = key["deepseek_openai_compatible"]["api_key"],
            base_url = key["deepseek_openai_compatible"]["base_url"]
        )

    def generate_comment(self, messages, max_completion_tokens=512, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    time.sleep(wait)
                else:
                    raise
    
