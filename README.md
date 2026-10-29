# EdgeForge AI ⚒️

**Turn one sketch or layout into a labelled computer-vision dataset, on your own GPU.**

## 🚀 What is EdgeForge?

EdgeForge is a **Synthetic Data Platform** built for startups and engineers who need computer-vision training data without the high cost of manual collection.

It transforms a **single input sketch/layout** into **batches of diverse, labelled training images**. The layout's edges constrain an SDXL + ControlNet generator, a prompt engine adds hard conditions (fog, night, rain, rust, motion blur and more), a layout augmenter adds geometric variety, and a YOLOv8 model writes bounding-box labels in YOLO format. You get back a `.zip` with `images/` and `labels/` folders.

![One coarse car layout, its Canny edges, a remixed layout, and the generated image with its YOLO label](docs/pipeline.jpg)

![Eight images from one batch request, each with its own sampled condition and the YOLO boxes written to the dataset](docs/gallery.jpg)

<sub>Real output, not cherry-picked: one `POST /generate_batch` call ("a car on a highway", batch size 8) on a free Colab T4, from the layout in [`docs/layout.png`](docs/layout.png). The boxes are the zip's own YOLO labels, mistakes included (a car in fog labelled "airplane", two merged cars labelled "truck"), which is why the labels need a spot-check before training.</sub>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yashsinha1702/edgeforge-ai/blob/main/notebooks/colab_demo.ipynb) **Try it on a free GPU:** the notebook installs EdgeForge on Colab, starts the engine and generates a small labelled dataset.

### **Key Features**

* **🎨 Memory-efficient high-res decoding:** A custom **Tiled VAE** decodes latents in 512 px tiles ("fractional batches") with Gaussian-weighted blending, so decoder VRAM is set by the tile size, not the image size: measured on a T4, it decodes 4096×4096 with 0.76 GB, where a plain decode runs out of memory ([benchmarks](#-benchmarks--tests)). The generation pipeline produces 1024×1024 images. `tests/benchmark.py` stress-tests the decoder on 4096×4096 latents and reports peak VRAM.
* **🏷️ Automated Annotation:** An integrated **YOLOv8** auto-labeller writes standard YOLO-format (`.txt`) bounding boxes for every generated image.
* **🏭 The Dataset Factory:** Batch mode produces unique variations of one intent. Each image gets a different visibility, weather or damage condition and a different random seed.
* **📐 Layout Augmentation:** Shifts, scales, flips and multiplies the input layout to create geometric diversity (for example, "1 car" → "3 cars in the distance").
* **🧠 The Director:** A rule-based prompt engine that expands simple intents (for example, "a rusty car, hard to see") into detailed photorealistic prompts. It's built as a drop-in point for an LLM later.
* **🌐 API + UI:** A FastAPI backend (with interactive docs at `/docs`) and a React/Vite frontend.

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Input Layout] --> B[Canny Edge Map<br/>1024x1024];
    B --> C[Layout Augmenter<br/>batch mode];
    D[User Intent] --> E[The Director];
    E --> F{SDXL + ControlNet Canny};
    C --> F;
    B -. single-shot .-> F;
    F -- latents --> G[Tiled VAE Decoder];
    G --> H[YOLOv8 Labeler];
    H --> I[ZIP: images/ + labels/];
```

| Stage | Module | What it does |
| --- | --- | --- |
| Edge extraction | `pipeline.py` → `preprocess_canny` | Resizes the upload to 1024×1024 and runs Canny edge detection (thresholds 100/200). |
| Layout augmentation | `layout_engine.py` → `LayoutAugmenter` | Pastes 1–3 copies of the edge map, each scaled 0.4–0.75×, randomly placed and flipped half the time, onto a blank canvas. Works best with a single object on a plain background. |
| Prompting | `prompt_expander.py` → `PromptExpander` | **Single-shot:** matches keywords ("hard to see", "rain", "damage", ...) to failure-mode modifiers. **Batch:** samples a random condition and seed per image. Both append quality boosters and a negative prompt. |
| Generation | `pipeline.py` → `EdgeForgePipeline` | `stabilityai/stable-diffusion-xl-base-1.0` with `diffusers/controlnet-canny-sdxl-1.0` (conditioning scale 0.5), fp16 with model CPU offload, returns **latents**. |
| Decoding | `tiled_vae.py` → `TiledVAEWrapper` | Decodes the latents with the fp32 SDXL VAE in 512 px tiles with 64 px overlap. Each tile is weighted by a 2-D Gaussian mask and accumulated in CPU buffers, then normalised, so tile seams blend away. |
| Labelling | `labeler.py` → `AutoLabeler` | YOLOv8n (COCO-pretrained) detections with confidence > 0.4, written as `class x_center y_center width height` (normalised). |

---

## ⚡ Quick Start (Hybrid Mode)

Use the hosted UI with the engine running on your own GPU.

1.  **Run the Engine (your GPU):** Build and run the Docker image (see [Option A](#option-a-docker-recommended) below). The first start downloads the model weights, so wait for `Factory Ready! 🚀` in the logs.

2.  **Open the Web Interface:** Go to **[https://edgeforge-ai.vercel.app](https://edgeforge-ai.vercel.app)**. Until you generate, it shows real example output. To generate, the page calls the engine at `http://127.0.0.1:8000` directly from your browser. If your browser blocks requests from the hosted page to localhost, run the frontend locally instead (`npm run dev`, see Option B).

3.  **Generate:** Upload a layout, describe the intent, choose a batch size and click **Generate Asset**. The dataset downloads as a `.zip`. You can generate as many datasets as you like for free on your own hardware.

---

## 🛠️ Installation (Local Development)

### **Prerequisites**

* NVIDIA GPU with CUDA (8 GB+ VRAM recommended). The pipeline runs SDXL in fp16, so a CPU-only setup isn't practical. With less than 14 GB of VRAM it uses model CPU offload, which parks the models in system RAM (more than 13 GB of RAM: on a 13 GB Colab VM the offload step gets the process killed); with 14 GB or more, such as Colab's T4, the models stay on the GPU.
* Docker with the NVIDIA Container Toolkit (for Container mode) OR Python 3.10–3.12 (for Source mode). `numpy<2` has no Python 3.13 wheels.
* Disk space for model weights. The first start downloads the fp16 weights of SDXL base and the SDXL Canny ControlNet, plus the SDXL VAE, from Hugging Face (about 10 GB in total), and ultralytics downloads `yolov8n.pt` automatically. No weights are stored in this repo.

### **Option A: Docker (Recommended)**

Build the container yourself to ensure all dependencies are locked.

```bash
# 1. Clone the repository
git clone https://github.com/yashsinha1702/edgeforge-ai.git
cd edgeforge-ai

# 2. Build the image (takes ~10 mins the first time)
docker build -t edgeforge-ai .

# 3. Run it. Mount the Hugging Face cache so model weights download only once.
docker run --gpus all -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  edgeforge-ai
```

The image is based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`, so PyTorch with CUDA 12.1 is preinstalled.

### **Option B: Manual Python Setup**

```bash
# 1. Create an environment and install PyTorch with CUDA first
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.1.0" --index-url https://download.pytorch.org/whl/cu121

# 2. Install the remaining dependencies
pip install -r requirements.txt

# 3. Start the Backend Server (src/ must be on PYTHONPATH, as in the Dockerfile)
PYTHONPATH=src python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload

# 4. Start the Frontend (in a new terminal), then open the URL Vite prints (default http://localhost:5173)
cd frontend
npm install
npm run dev
```

---

## 📚 Usage Guide

### **1. Single-Shot Mode (Prototyping)**

* **Best for:** Testing prompts and layouts.
* **Input:** Upload a coarse layout (for example, a white car on a black background) or a Canny edge map, and set the batch size to 1.
* **Output:** `edgeforge_asset.zip` containing `generated_image.png` (1024×1024), `generated_image.txt` (YOLO labels) and `prompt.json` (the expanded prompt). Single-shot uses a fixed seed (42), so results are repeatable for the same input and prompt.

### **2. Batch Factory Mode (Production)**

* **Best for:** Creating the actual training dataset.
* **Batch Size:** The UI slider goes from 2 to 10. The API accepts 1 to 50.
* **Process:**
  1. **Augment:** The engine randomly shifts, scales, flips and multiplies the input layout for every image.
  2. **Generate:** Each image gets a randomly sampled condition: low visibility (fog, night, blizzard, motion blur), weather (rainstorm, harsh sunlight) or damage (rust, dents, dirt).
  3. **Label:** YOLOv8 detects objects in each new image.
* **Output:** `edgeforge_dataset.zip` containing:
  * `images/train_0000.png`, ... (the generated synthetic data)
  * `labels/train_0000.txt`, ... (YOLO text files)
  * `prompts.json` (the prompt, condition and seed behind each image, so a dataset can be traced and regenerated)

### **3. Using the API directly**

```bash
# Single image
curl -X POST http://127.0.0.1:8000/generate \
  -F "intent=a rusty car on a street, hard to see" \
  -F "control_image=@layout.png" \
  -o edgeforge_asset.zip

# Batch of 10
curl -X POST http://127.0.0.1:8000/generate_batch \
  -F "intent=a car on a highway" \
  -F "batch_size=10" \
  -F "control_image=@layout.png" \
  -o edgeforge_dataset.zip
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

> **Labels note:** The class IDs are COCO indices from the pretrained YOLOv8n model (for example, `2` = car). They're machine-generated pseudo-labels, so spot-check them before training. The zip doesn't include a class-name file.

---

## 🧪 Benchmarks & Tests

```bash
# API tests, CPU only: fake GPU pipeline, real prompt engine and layout augmenter
pip install pytest httpx && pytest tests/test_api.py

# Tiled VAE stress test: decodes random 4096x4096 latents and prints time and peak VRAM (GPU required)
PYTHONPATH=src/efficient_diffusion_loader python tests/benchmark.py

# End-to-end generation from src/input_layout.jpg -> output_director_test.png (GPU required)
cd src && python test_generation.py
```

`tests/test_api.py` checks the zips each endpoint returns, the 400 and 422 errors, that two concurrent requests never run on the GPU at the same time while the server keeps answering, and that no temp files are written. Run it by path: bare `pytest` also collects `src/test_generation.py`, which loads the full SDXL pipeline at import time.

### Measured on a free Colab T4 (15 GB)

October 2026, Python 3.12, torch 2.11, diffusers 0.25.0, the models kept on the GPU:

| Step | Result |
|---|---|
| Load all models (weights already downloaded) | 42 s |
| One 1024×1024 image: 50 SDXL + ControlNet steps, tiled decode, YOLO labels | about 80 s |
| Peak GPU memory while generating | 9.9 GB allocated by PyTorch |

Decoding random SDXL latents with the fp32 VAE, tiles of 512 px with 64 px overlap as in `pipeline.py` (peak memory on top of the VAE's own weights):

| Output size | Plain `vae.decode` | diffusers `enable_tiling()` | EdgeForge tiled decoder |
|---|---|---|---|
| 1024 × 1024 | 2.6 s, 3.0 GB | 2.7 s, 3.0 GB | 5.5 s, 0.76 GB |
| 2048 × 2048 | 18.3 s, 12.0 GB | 17.1 s, 3.0 GB | 16.0 s, 0.76 GB |
| 4096 × 4096 | out of memory | 77.2 s, 3.3 GB | 60.7 s, 0.76 GB |

The tiled decoder's memory stays at 0.76 GB at every size because only one 512 px tile is on the GPU at a time and the blended image is accumulated in system RAM. It pays for that at 1024 px, where a single plain decode is cheaper.

---

## 📂 Project Structure

```text
edgeforge-ai/
├── src/
│   ├── app/main.py                    # FastAPI app: /generate and /generate_batch
│   ├── efficient_diffusion_loader/
│   │   ├── pipeline.py                # SDXL + ControlNet (Canny) pipeline + tiled decode
│   │   ├── tiled_vae.py               # Tiled VAE decoder with Gaussian blending
│   │   ├── prompt_expander.py         # "The Director": rule-based prompt expansion
│   │   ├── layout_engine.py           # Layout augmenter: shift / scale / flip / multiply
│   │   └── labeler.py                 # YOLOv8 auto-labeller -> YOLO .txt
│   ├── test_generation.py             # Manual end-to-end generation script (GPU)
│   └── input_layout.jpg               # Sample input for test_generation.py
├── tests/
│   ├── test_api.py                    # API tests on CPU (fake GPU pipeline)
│   └── benchmark.py                   # Tiled VAE time / VRAM benchmark at 4096x4096 (GPU)
├── notebooks/colab_demo.ipynb         # Run the engine on a free Colab T4
├── docs/                              # README images and the sample layout (layout.png)
├── frontend/                          # React + Vite UI (public/examples: sample outputs for the hosted page)
├── TiledVAEWrapper.py                 # Earlier standalone version of the tiled decoder
├── Dockerfile                         # CUDA 12.1 / PyTorch 2.1 runtime image for the API
├── requirements.txt
└── setup.py                           # Packages efficient_diffusion_loader
```

---

## ⚠️ Limitations & Security

* **Synchronous generation:** Each request waits until its whole batch is generated and zipped in memory, so large batches are slow (about 80 s per image on a T4) and memory-hungry. Requests run one at a time on the GPU and queue behind each other; the server itself keeps answering. A job queue with progress reporting is the natural next step.
* **Output resolution:** The pipeline currently generates 1024×1024 images. The tiled decoder scales further, but larger generation sizes aren't wired up.
* **The Director is rule-based:** It uses keyword matching and random sampling from a fixed ontology. Its structured `constraints` output is informational and not yet consumed downstream.
* **No authentication:** The API allows all CORS origins and has no auth, so any web page you open could send it requests. Keep it bound to `127.0.0.1` (as in the commands above) and don't expose port 8000 on untrusted networks.

---

## 🤝 Contributing

We welcome pull requests! Please make sure you test the **Batch Generation** flow before submitting, as it's the core value proposition.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
