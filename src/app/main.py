import json
import threading
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

# Import all modules
from efficient_diffusion_loader import EdgeForgePipeline, PromptExpander, AutoLabeler, LayoutAugmenter


MAX_BATCH = 50

# Global Variables
forge_pipeline = None
director = None
labeler = None
layout_engine = None

# One GPU job at a time: concurrent requests queue here instead of competing for VRAM.
gpu_lock = threading.Lock()


def load_models():
    global forge_pipeline, director, labeler, layout_engine
    print("Loading EdgeForge Factory...")
    director = PromptExpander()
    labeler = AutoLabeler()
    layout_engine = LayoutAugmenter() # Initialize Remix Engine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forge_pipeline = EdgeForgePipeline(device=device)
    print("Factory Ready! 🚀")


@asynccontextmanager
async def lifespan(app):
    load_models()  # once, before the first request
    yield


app = FastAPI(title="EdgeForge AI API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_layout(upload: UploadFile) -> BytesIO:
    """The uploaded layout, kept in memory (no shared temp file), checked to be an image."""
    data = upload.file.read()
    try:
        Image.open(BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="control_image is not a readable image")
    return BytesIO(data)


# The endpoints are plain `def`: FastAPI runs them in its thread pool, so a long generation
# never blocks the event loop (and /docs keeps answering while the GPU works).
@app.post("/generate")
def generate_endpoint(
    intent: str = Form(...),
    control_image: UploadFile = File(...)
):
    """ Single Shot Endpoint (No layout remixing) """
    layout = read_layout(control_image)

    with gpu_lock:
        processed_edges = forge_pipeline.preprocess_canny(layout)
        directive = director.expand(intent)

        result_image = forge_pipeline.generate(
            prompt=directive['prompt'],
            control_image=processed_edges,
            seed=42
        )

        label_text = labeler.label_image(result_image)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        img_buffer = BytesIO()
        result_image.save(img_buffer, format="PNG")
        zip_file.writestr("generated_image.png", img_buffer.getvalue())
        zip_file.writestr("generated_image.txt", label_text)
        zip_file.writestr("prompt.json", json.dumps({"prompt": directive['prompt'], "seed": 42}, indent=1))

    zip_buffer.seek(0)
    return Response(content=zip_buffer.getvalue(), media_type="application/zip")

@app.post("/generate_batch")
def generate_batch_endpoint(
    intent: str = Form(...),
    control_image: UploadFile = File(...),
    batch_size: int = Form(5)
):
    """ Batch Factory Endpoint (WITH layout remixing) """
    if not 1 <= batch_size <= MAX_BATCH:
        raise HTTPException(status_code=422, detail=f"batch_size must be between 1 and {MAX_BATCH}")

    # 1. Load Base Layout
    layout = read_layout(control_image)

    # 3. Production Loop
    zip_buffer = BytesIO()
    with gpu_lock, zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Get base edges (The single car)
        base_edges = forge_pipeline.preprocess_canny(layout)

        # 2. Director: Get Prompt Variations
        variations = director.generate_variations(intent, count=batch_size)

        print(f"Starting Batch Generation of {batch_size} images...")
        prompts = []

        for idx, var in enumerate(variations):
            print(f"[{idx+1}/{batch_size}] Remixing & Forging...")

            # --- GEOMETRY STEP ---
            # Randomly shift/scale/multiply the car edges
            remixed_edges = layout_engine.augment(base_edges, max_objects=3)

            # --- GENERATION STEP ---
            img = forge_pipeline.generate(
                prompt=var['prompt'],
                control_image=remixed_edges, # Use the Remix!
                seed=var['seed']
            )

            # --- LABELING STEP ---
            label_txt = labeler.label_image(img)

            # --- SAVE STEP ---
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")

            filename = f"train_{idx:04d}"
            zip_file.writestr(f"images/{filename}.png", img_buffer.getvalue())
            zip_file.writestr(f"labels/{filename}.txt", label_txt)
            prompts.append({"image": f"images/{filename}.png", "prompt": var['prompt'], "seed": var['seed'],
                            "condition": var.get('constraints', {}).get('condition', '')})

        # What made each image, so a dataset can be traced back and regenerated
        zip_file.writestr("prompts.json", json.dumps(prompts, indent=1))

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=edgeforge_dataset.zip"}
    )
