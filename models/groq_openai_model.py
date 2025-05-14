import os
from groq import Groq

class GroqOpenAIModel(object):
    def __init__(self, model_name, device, dtype, **kwargs):
        self.model_name = model_name.split("/")[-1] # get everythign after the / for the model name
        self.device = device
        self.dtype = dtype

        self.temperature = kwargs.get("temperature", 1.0)
        self.top_k = kwargs.get("top_k", 50)
        self.top_p = kwargs.get("top_p", 0.9)
        self.num_beams = kwargs.get("num_beams", 1)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.max_new_tokens = kwargs.get("max_new_tokens", 256)
        self.min_new_tokens = kwargs.get("min_new_tokens", 0)

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    def get_messages(self, prompt, splits=["system", "user"]):
        messages = []
        for split in splits:
            start_tag = f"<{split}>"
            end_tag = f"</{split}>"
            if start_tag not in prompt or end_tag not in prompt:
                continue

            start_idx = prompt.find(start_tag)
            end_idx = prompt.find(end_tag)

            messages.append({
                "role": split,
                "content": prompt[start_idx + len(start_tag):end_idx].strip()
            })

        if len(messages) == 0:
            messages.append({
                "role": "user",
                "content": prompt
            })

        return messages

    def generate(self, prompt, return_prompt=False, image_files=None, temperature=None, max_new_tokens=None):
        if temperature is None:
            temperature = self.temperature
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        messages = self.get_messages(prompt)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p = self.top_p,
        )
        output = response.choices[0].message.content.strip()
        # can check response.choices[0].finish_reason to see if the model stopped due to the stop sequence (i.e. finish_reason == "stop")
        return output