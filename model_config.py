from transformers import BitsAndBytesConfig

EIGHT_BIT = BitsAndBytesConfig(load_in_8bit=True)

MODEL_CONFIG = {
    "meta-llama/Llama-3.2-1B-Instruct": {"batch_size": 30, "device_map": {'': 0}},
    "utter-project/EuroLLM-1.7B-Instruct": {"batch_size": 30, "device_map": {'': 0}},
    "meta-llama/Llama-3.2-3B-Instruct": {"batch_size": 15, "device_map": {'': 0}},
    "mistralai/Mistral-7B-Instruct-v0.3": {"batch_size": 10, "device_map": {'': 0}},
    "BioMistral/BioMistral-7B": {"batch_size": 10, "device_map": {'': 0}},
    "LeoLM/leo-hessianai-7b-chat": {"batch_size": 6, "device_map": {'': 0}},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"batch_size": 7, "device_map": {'': 0}},
    "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct": {"batch_size": 7, "device_map": {'': 0}},
    "mistralai/Mistral-Nemo-Instruct-2407": {"batch_size": 10, "device_map": {'': 0}, "quantization_config": EIGHT_BIT}, # fits on one 48 GB GPU in 8 bit quantization
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"batch_size": 1, "device_map": "auto", "quantization_config": EIGHT_BIT}, # needs all 3 GPUs
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"batch_size": 4, "device_map": "auto", "quantization_config": EIGHT_BIT}, # use only two GPUs
}