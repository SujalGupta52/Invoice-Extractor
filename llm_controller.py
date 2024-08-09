from llama_cpp import Llama
import json


class LLM_controller:
    def __init__(
        self,
        model_path,
        verbose=True,
        temperature=0.5,
    ):
        self.temperature = temperature
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=verbose,
            flash_attn=True,
            n_ctx=2048,
        )

    def generate(self, message):
        out = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": """You are tasked with extracting information from invoices

                        Rules:
                        - leave unknown fields blank
                        - Keep all numerical values as int with no commas 

                        Respond WITH NO OTHER TEXT AS SUFFIX OR PREFIX but a JSON document exactly like this example JSON document:
                        {
                        "customer_details": object with customer information,
                        "products": Array of products with amounts,
                        "total_amount":  Total amount of the invoice, payable amount or amount after tax,
                        }
                        """,
                },
                {
                    "role": "user",
                    "content": f"Input Data: {message}",
                },
            ],
            temperature=self.temperature,
        )["choices"][0]["message"]["content"]
        return out

    def parse_json_from_llm(self, message_with_json):
        str = message_with_json.replace("\n", "")
        start_idx = -1
        end_idx = -1
        for i in range(len(str)):
            if str[i] == "{":
                start_idx = i
                break
        for i in range(len(str) - 1, -1, -1):
            if str[i] == "}":
                end_idx = i
                break
        json_output = str[start_idx : end_idx + 1]
        return json.loads(json_output)
