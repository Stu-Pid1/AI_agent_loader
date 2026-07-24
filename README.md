# AI Agent Loader

A local AI model runner with a browser-based GUI. Browse, download, and run Hugging Face models on your own hardware.

## Features

- **Hub Browser** — Search and explore Hugging Face models with filters for task type, library, and popularity. View model cards, file sizes, and download with one click.
- **Text Generation** — Run LLMs (Llama, Mistral, Phi, etc.) with chat and completion modes. Supports both transformers and quantized GGUF models.
- **Image Generation** — Generate images with Stable Diffusion, SDXL, Flux, and other diffusion models.
- **Classification** — Text classification, sentiment analysis, image classification, and named entity recognition.
- **Speech** — Speech-to-text (Whisper) and text-to-speech.
- **Object Detection** — Detect objects in images with bounding boxes.
- **Summarization & Translation** — Summarize long texts and translate between languages.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- ~2GB disk space for the app + space for downloaded models

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd AI_agent_loader
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install PyTorch with CUDA:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **(Optional) GGUF support with GPU acceleration:**
   ```bash
   set CMAKE_ARGS=-DGGML_CUDA=on
   pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```

6. **(Optional) Set up Hugging Face token for gated models:**
   ```bash
   copy .env.example .env
   # Edit .env and add your HF token
   ```

## Usage

```bash
python app.py
```

Open your browser to `http://localhost:7860`.

### Quick Start

1. Go to the **Hub Browser** tab
2. Search for a model (e.g., "TinyLlama" for text generation)
3. Click a result to see details, then click **Download Model**
4. Switch to the appropriate tab (e.g., **Text Generation**)
5. Click **Refresh**, select the model, click **Load**
6. Enter a prompt and click **Generate**

## Architecture

The app uses a modular runner system. Each AI task type has its own runner that implements a common interface (`load`, `run`, `unload`). Adding support for a new model type requires:

1. Create a new runner in `runners/`
2. Create a UI tab in `ui/`
3. Register the pipeline tag in `config/task_registry.py`
4. Add the tab to `ui/app_builder.py`
