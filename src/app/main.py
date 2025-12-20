from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import torch
import zipfile

# Import all modules
from efficient_diffusion_loader import EdgeForgePipeline, PromptExpander, AutoLabeler, LayoutAugmenter

app = FastAPI(title="EdgeForge AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables
forge_pipeline = None
director = None
labeler = None
layout_engine = None

@app.on_event("startup")
def load_models():
    global forge_pipeline, director, labeler, layout_engine
    print("Loading EdgeForge Factory...")
    director = PromptExpander()
    labeler = AutoLabeler()
    layout_engine = LayoutAugmenter() # Initialize Remix Engine
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forge_pipeline = EdgeForgePipeline(device=device)
    print("Factory Ready! 🚀")

@app.post("/generate")
async def generate_endpoint(
    intent: str = Form(...),
    control_image: UploadFile = File(...)
):
    """ Single Shot Endpoint (No layout remixing) """
    image_bytes = await control_image.read()
    input_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_pil.save("temp_input.png")
    
    processed_edges = forge_pipeline.preprocess_canny("temp_input.png")
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
        
    zip_buffer.seek(0)
    return Response(content=zip_buffer.getvalue(), media_type="application/zip")

@app.post("/generate_batch")
async def generate_batch_endpoint(
    intent: str = Form(...),
    control_image: UploadFile = File(...),
    batch_size: int = Form(5)
):
    """ Batch Factory Endpoint (WITH layout remixing) """
    # 1. Load Base Layout
    image_bytes = await control_image.read()
    input_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_pil.save("temp_batch_input.png")
    
    # Get base edges (The single car)
    base_edges = forge_pipeline.preprocess_canny("temp_batch_input.png")

    # 2. Director: Get Prompt Variations
    variations = director.generate_variations(intent, count=batch_size)
    
    # 3. Production Loop
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        
        print(f"Starting Batch Generation of {batch_size} images...")
        
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
            
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(), 
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=edgeforge_dataset.zip"}
    )