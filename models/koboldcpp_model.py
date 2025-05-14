import os
#import fcntl
import utils
import requests
import json


from transformers import AutoTokenizer, AutoModelForCausalLM

class KoboldModel(object):
    def __init__(self, model_name, device, dtype, **kwargs):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)
    

    def call_kobold_api(self, endpoint, request_type, payload=""): 
        # We establish our base configuration
        base_url = "http://localhost:5001"
        headers = {"Content-Type": "application/json"}

        # We build the full URL
        url = base_url + endpoint

        if request_type == "GET":
            response = requests.get(url, headers=headers)
        else:
            response = requests.post(url, json=payload, headers=headers)
                    
        return response.json()

    def generate(self, prompt, return_prompt=False, image_files=None, temperature=None, max_new_tokens=None):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        generate_endpoint = "/api/v1/generate"
        payload = {
            "prompt": prompt,
            "max_length": max_new_tokens,
            "temperature": temperature,
        }
        
        response = self.call_kobold_api(generate_endpoint, "POST", payload) 
        output = response["results"][0]["text"]
        return output