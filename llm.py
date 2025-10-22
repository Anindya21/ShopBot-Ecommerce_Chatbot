from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from functools import lru_cache
 


def load_llm():
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-270m-it",
        device_map="auto" if device == 0 else "cpu",
        )

    pipe = pipeline(
                task="text-generation", 
                model=model,
                tokenizer=tokenizer, 
                model_kwargs={"dtype": torch.float16},
                max_new_tokens=256,
                truncation=True
                )
    
    return HuggingFacePipeline(pipeline=pipe)

@lru_cache()
def get_llm():
    return load_llm()

