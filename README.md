# UroLlmEval - benchmarking LLMs for tumor documentation on a set of German urological doctors's notes

## Overview

This repository contains Python code that builds on the `transformers` package to evaluate large language models (LLMs) for tumor documentation on a data set of text snippets from urological doctor's notes.
[The data](https://huggingface.co/datasets/stefan-m-lenz/UroLlmEvalSet) is hosted on Hugging Face. It will be downloaded via the `datasets` package when running the evaluation.

More information about the data set and the design of the evaluation can be found in the article:

> Lenz, S., Ustjanzew, A., Jeray, M., Panholzer, T. (2025). Can open source large language models be used for tumor documentation in Germany?  - An evaluation on urological doctors’ notes. [arXiv preprint](http://arxiv.org/abs/2501.12106).


## Setup
The project was developed using Python 3.12.
To run it, you need to have Python and pip installed on your system.

Follow these steps to get the evaluation up and running:

1. Clone or download this repository to your local machine:
    ```bash
    git clone https://github.com/stefan-m-lenz/UroLlmEval.git
    ```

2. Navigate into the project directory:
    ```bash
    cd UroLlmEval
    ```

3. Create and activate an virtual environment:
    ```bash
    python3 -m venv venv
    #On Unix or MacOS
    source venv/bin/activate
    # On Windows (Powershell)
    .\venv\Scripts\Activate.ps1
    ```

4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

5. Get access to the models on Hugging Face.

    For running the benchmark with the configured models, you need to be granted access to the following gated models on Hugging Face:
    - [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
    - [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
    - [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
    - [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
    - [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
    - [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
    - [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)

    (For more information about gated models, refer to the [Hugging Face documentation](https://huggingface.co/docs/hub/models-gated).)

    The following models are further more included in the analysis for which you do not need to apply for access:
    - [EuroLLM-1.7B-Instruct](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct)
    - [BioMistral-7B](https://huggingface.co/BioMistral/BioMistral-7B)
    - [leo-hessianai-7b-chat](https://huggingface.co/LeoLM/leo-hessianai-7b-chat)
    - [Llama-3.1-SauerkrautLM-8b-Instruct](https://huggingface.co/LeoLM/leo-hessianai-7b-chat)

    For acessing the gated models, you need to configure your HuggingFace access token.
    If you haven't already added set up your access token, create a token in your Hugging Face account
    via **Profile > Settings > Access Tokens**.
    Make sure the token allows read access to all necessary models.
    Then configure the token in your terminal session:
    ```bash
    # On Unix or MacOS
    export HF_TOKEN=your_personal_access_token
    # On Windows(Powershell)
    $env:HF_TOKEN="your_personal_access_token"
    ```

## Configuration

In the file [`model_config.py`](model_config.py), it is possible to (re-)configure the models and to define how they can be loaded onto the GPU.
The script was developed on a machine with 3 NVIDIA A40 GPUs, each having 48 GB of VRAM.
For different setups, the `batch_size`, `device_map` and the `quantization_config` parameters must possibly be changed according to the availability of GPU RAM.

The models are defined via their Hugging Face identifier in the `model_config.py`.
The `quantization_config` and `device_map` parameters are used for loading the models via the `AutoModelForCausalLM.from_pretrained()` function from the `transformers` library.
The `batch_size` is then used for batched inference.


## Usage

Run the evaluation:
```bash
python ./run_evaluation.py
```

While running the evaluation, files will be created for each of the model/prompt combinations.
These files contain the queries and the answers from the model and have names following the pattern `stepX_*.csv` (X = 1,2,3).
After each step is completed, a file `analysis_stepX.csv` is created, which summarizes the outcome using evaluation metrics.
The execution of the evaluation can be interrupted via `Ctrl+C` if needed.
When run again, the script will check the files and resume from the last file (model/prompt combination) that had been finished before the interrupt happened.
At the end of the evaluation, HTML tables are created in the folder `output/summary` that display the evaluation metrics in a more readable way.

After the evaluation script has been run, the following script may be used to create plots from the analysis results:

```bash
python ./create_figures.py
```

The plots are then created in the folder `output/summary/plots`.

## Citation

If you use this code or the evaluation data in your work, please consider citing the [article](http://arxiv.org/abs/2501.12106) preprint:

```bibtex
@article{UroLlmEval_2025,
  title        = {Can open source large language models be used for tumor documentation in {Germany}? - {An} evaluation on urological doctors' notes},
  author       = {Lenz, Stefan and Ustjanzew, Arsenij and Jeray, Marco and Panholzer, Torsten},
  year         = {2025},
  month        = {Jan},
  journal      = {arXiv preprint},
  volume       = {arXiv:2501.12106},
  doi          = {10.48550/arXiv.2501.12106},
  url          = {http://arxiv.org/abs/2501.12106},
  publisher    = {arXiv}
}
```

