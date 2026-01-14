# Project DaseR Context

## Project Overview
DaseR is a development workspace focused on **Ray** framework usage and customization, specifically interacting with **vLLM** for large model inference. The project contains examples of running data processing and inference pipelines on GPUs and includes the Ray source code as a submodule for potential low-level modifications.

## Key Files & Directories

- **`example.py`**: A Python script demonstrating how to use Ray Data with vLLM for batch inference (sentiment analysis on IMDb dataset). It includes custom logic for:
    - Managing CUDA visible devices.
    - Setting up a `VLLMSentimentPredictor` class using Ray actors.
    - Handling distributed inference via `map_batches`.
- **`ray/`**: A git submodule pointing to the `ray-project/ray` repository. This is intended for checking out the Ray source code to make local modifications.
- **`README.md`**: Contains basic setup instructions, specifically for installing a specific version of Ray (`2.53.0`) and setting up the development environment.

## Setup & Usage

### 1. Environment Setup
The project requires a specific version of Ray.

```bash
pip install -U "ray[default]==2.53.0"
```

To work with the Ray source code (ensure the submodule is initialized first):

```bash
# Initialize submodule if directory is empty
git submodule update --init --recursive

# Setup dev environment
cd ray
python python/ray/setup-dev.py -y
```

### 2. Running the Example
The `example.py` script performs sentiment analysis.

```bash
python example.py [options]
```

**Common Options:**
- `--data-path <path>`: Path to the input CSV file (default checks `/data/imdb.csv` then `/data/zwt/imdb.csv`).
- `--model <path>`: Path to the vLLM compatible model (default: `/data/zwt/model/models/Qwen/Qwen3-8B`).
- `--limit <int>`: Number of records to process (default: 1000).
- `--cuda-visible-devices <list>`: Comma-separated GPU IDs (e.g., "0,1").

## Development Conventions

- **Type Hinting**: Python code uses standard `typing` module for function signatures (e.g., `def func(a: str) -> None:`).
- **GPU Management**: The code explicitly manages `CUDA_VISIBLE_DEVICES` to control which GPUs are visible to the Ray workers and vLLM engine.
- **Ray Data**: The project uses Ray Data (`ray.data`) for loading and processing datasets, utilizing `map_batches` for efficient GPU inference.
