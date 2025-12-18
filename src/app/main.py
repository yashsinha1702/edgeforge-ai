from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import torch
import zipfile  # Required for ZIP export

# Import your custom modules
from efficient_diffusion_loader import EdgeForgePipeline, PromptExpander, AutoLabeler

app = FastAPI(title="EdgeForge AI API", version="0.1.0")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------

# --- GLOBAL VARIABLES ---
# These must be defined at the top level so all functions can see them
forge_pipeline = None
director = None
labeler = None

@app.on_event("startup")
def load_models():
    """
    Load models once on startup.
    """
    # We use 'global' to write to the variables defined above
    global forge_pipeline, director, labeler
    
    print("Loading EdgeForge Engine... (This may take a minute)")
    
    # 1. Initialize The Director (Prompt Logic)
    director = PromptExpander()
    
    # 2. Initialize The Artist (Diffusion + Tiled VAE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forge_pipeline = EdgeForgePipeline(device=device)

    # 3. Initialize The Labeler (YOLO)
    labeler = AutoLabeler()
    
    print("EdgeForge Engine Ready! 🚀")

@app.post("/generate")
async def generate_endpoint(
    intent: str = Form(...),
    control_image: UploadFile = File(...)
):
    """
    Generates an image, detects objects, and returns a ZIP file with both.
    """
    # 1. Process Input Image
    image_bytes = await control_image.read()
    input_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    input_pil.save("temp_input_debug.png")
    
    # Preprocess edges
    processed_edges = forge_pipeline.preprocess_canny("temp_input_debug.png")

    # 2. Run The Director
    directive = director.expand(intent)
    print(f"Executing Directive: {directive['prompt']}")

    # 3. Run The Artist
    result_image = forge_pipeline.generate(
        prompt=directive['prompt'],
        control_image=processed_edges,
        seed=42
    )

    # 4. Run The Labeler (Auto-Annotation)
    # The 'labeler' variable is now guaranteed to exist from the global scope
    label_text = labeler.label_image(result_image)
    print(f"Generated Labels: {label_text}")

    # 5. Create ZIP (Image + Label)
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Add Image
        img_buffer = BytesIO()
        result_image.save(img_buffer, format="PNG")
        zip_file.writestr("generated_image.png", img_buffer.getvalue())
        
        # Add Label (YOLO Format)
        zip_file.writestr("generated_image.txt", label_text)
        
        # Add Metadata
        zip_file.writestr("metadata.json", str(directive))

    # Return the ZIP
    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(), 
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=edgeforge_asset.zip"}
    )

@app.get("/health")
def health_check():
    return {"status": "online", "gpu": torch.cuda.get_device_name(0)}