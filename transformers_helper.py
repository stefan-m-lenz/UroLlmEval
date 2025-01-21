from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
import time
from math import ceil
from model_config import MODEL_CONFIG


def get_model_batch_size_from_config(model):
    if model in MODEL_CONFIG.keys():
        batch_size = MODEL_CONFIG[model]["batch_size"]
    else:
        batch_size = 1
        print(f"Using a default batch size of one for the model {model}")
    return batch_size


def remove_system_header(text):
    system_header_indicator = "<|start_header_id|>system<|end_header_id|>"
    message_end_indicator = "<|eot_id|>"
    start_index = text.find(system_header_indicator)
    if start_index == -1:
        return text

    end_index = text.find(message_end_indicator, start_index + len(system_header_indicator))
    if end_index == -1:
        return text

    # Return the text with the portion removed
    return text[:start_index] + text[(end_index + len(message_end_indicator)):]



class TransformersInferencer:
    def __init__(self):
        self.current_model = None
        self.tokenizer = None
        self.model = None


    # Prepare the input text using the chat template
    # https://huggingface.co/docs/transformers/main/chat_templating
    def _create_chat(self, messages):
        """Prepare a single chat input string from a list of messages."""
        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            use_default_system_prompt=False
        )

        if messages[0]["role"] != "system":
            # The system header of Llama models contains the current date,
            # which prevents reproducibility
            chat = remove_system_header(chat)

        return chat


    def batch_query_model(self, queries, batch_size=None, show_progress=True):
        """Process a list of queries in sorted batches by length to manage GPU memory usage."""
        if not isinstance(queries, list):
            raise ValueError("queries should be a list of query dictionaries")

        model_name = queries[0]["model"]
        if not self.current_model or self.current_model != model_name:
            self.load_model(model_name)

        if not batch_size:
            batch_size = get_model_batch_size_from_config(model_name)

        # Extract shared parameters from the first query
        temperature = queries[0]["options"]["temperature"]
        max_new_tokens = queries[0]["options"]["num_predict"]
        do_sample = temperature > 0
        temperature = temperature if do_sample else None
        top_p = 0.9 if do_sample else None

        # Sort queries by input length for more efficient batching
        def get_message_len(query):
            return len(str(query[1]["messages"]))
        sorted_queries = sorted(enumerate(queries), key=get_message_len, reverse=True)
        sorted_indices, sorted_queries = zip(*sorted_queries)

        # Calculate the number of sub-batches
        num_batches = ceil(len(sorted_queries) / batch_size)
        sorted_responses = [None] * len(sorted_queries)  # Placeholder for sorted responses

        for i in range(num_batches):
            batch_start_time = time.time()

            # Create a sub-batch from sorted queries
            sub_batch = sorted_queries[i * batch_size : (i + 1) * batch_size]

            # Prepare inputs for this sub-batch
            chats = [self._create_chat(query["messages"]) for query in sub_batch]
            inputs = self.tokenizer(
                chats,
                return_tensors="pt",
                padding="longest"
            ).to("cuda")

            # Generate responses for this sub-batch
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode responses and store them in the correct position
            sub_batch_responses = self.tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            sorted_responses[i * batch_size : (i + 1) * batch_size] = sub_batch_responses

            # Clear memory after each sub-batch
            del inputs, outputs
            torch.cuda.empty_cache()

            # Show progress if enabled
            if show_progress:
                batch_elapsed_time = time.time() - batch_start_time
                print(f'Batch {i+1}/{num_batches} processed in {batch_elapsed_time} seconds.', end='\r', flush=True)

        # Reorder responses to match the original query order
        unsorted_responses = [None] * len(sorted_responses)
        for original_index, response in zip(sorted_indices, sorted_responses):
            unsorted_responses[original_index] = response

        # Final progress print for the last batch
        if show_progress:
            print("\n")

        return unsorted_responses


    def query_model(self, query):
        return self.batch_query_model(queries = [query], batch_size=1)[0]


    def load_model(self, model_name):
        if self.current_model and self.current_model != model_name:
            self.unload_model()
        self.current_model = model_name

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.current_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Enable quantization if needed
        if self.current_model in MODEL_CONFIG.keys() and "quantization_config" in MODEL_CONFIG[self.current_model]:
            quantization_config = MODEL_CONFIG[self.current_model]["quantization_config"]
        else: # no quantization config, set to None
            quantization_config = None

        # Use specified device_map parameter
        if self.current_model in MODEL_CONFIG.keys() and "device_map" in MODEL_CONFIG[self.current_model]:
            device_map = MODEL_CONFIG[self.current_model]["device_map"]
        else:
            device_map = None

        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.current_model,
            device_map=device_map,
            quantization_config=quantization_config
        )


    def unload_model(self):
        # Clear the model and tokenizer
        del self.model
        del self.tokenizer
        del self.current_model
        self.model = None
        self.tokenizer = None
        self.current_model = None

        # Force garbage collection
        gc.collect()

        # Clear the GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __del__(self):
        self.unload_model()