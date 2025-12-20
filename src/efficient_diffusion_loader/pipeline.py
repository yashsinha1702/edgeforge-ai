import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from .tiled_vae import TiledVAEWrapper

class EdgeForgePipeline:
    def __init__(self, device="cuda"):
        self.device = device
        print(f"Loading EdgeForge Pipeline on {device}...")

        # 1. Load ControlNet (The 'Structure' enforcer)
        # Using 'Canny' (Edge Detection) allows us to sketch a scene and have AI fill it.
        controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_id, 
            torch_dtype=torch.float16
        ).to(device)

        # 2. Load the VAE (The 'Memory' bottleneck)
        # 2. Load the VAE (The 'Memory' bottleneck)
        # CRITICAL FIX: Always load SDXL VAE in float32 to prevent black images/NaNs
        vae_id = "stabilityai/sdxl-vae"
        self.vae = AutoencoderKL.from_pretrained(
            vae_id, 
            torch_dtype=torch.float32 
        ).to(device)

        # 3. Apply your OPTIMIZATION (Module 4)
        # We wrap the VAE immediately so the pipeline uses fractional decoding
        self.tiled_vae = TiledVAEWrapper(self.vae, tile_size=512, overlap=64)

        # 4. Load Main Pipeline (SDXL)
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_id,
            controlnet=self.controlnet,
            vae=self.vae, # Pass our VAE (which we will hijack during decode)
            torch_dtype=torch.float16,
        ).to(device)
        
        # Optimize for speed
        self.pipe.enable_model_cpu_offload() # Standard Diffusers optimization

    def preprocess_canny(self, image_path):
        """
        Converts a real image into a 1024x1024 Canny edge map.
        """
        image_pil = Image.open(image_path).convert("RGB")
        
        # --- CRITICAL FIX 2: RESIZE ---
        # SDXL works best at 1024x1024. We resize to ensure tensor alignment.
        image_pil = image_pil.resize((1024, 1024), Image.LANCZOS)
        
        image = np.array(image_pil)
        
        # Detect edges
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        
        return Image.fromarray(image)

    def generate(self, prompt, control_image, seed=None):
        if seed:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        # 1. Run Diffusion
        # We need a strong negative prompt for SDXL to look realistic
        negative_prompt = "cartoon, drawing, anime, low quality, blur, distortion, grid, messy"
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt, # Added Negative Prompt
            image=control_image,
            controlnet_conditioning_scale=0.5,
            output_type="latent",
            generator=generator
        )
        latents = output.images 

        # --- CRITICAL FIX 1: SCALING ---
        # The VAE expects latents to be scaled up. 
        # Standard value is 0.13025, so we divide by it (effectively multiplying).
        if hasattr(self.vae.config, "scaling_factor"):
            latents = latents / self.vae.config.scaling_factor
        else:
            latents = latents / 0.13025 # Fallback for SDXL default
        # -------------------------------

        # 2. Decode
        print("Decoding with Fractional Batches...")
        self.vae.to(self.device) 
        final_image = self.tiled_vae.decode_with_blending(latents)
        
        # 3. Post-process
        final_image = (final_image / 2 + 0.5).clamp(0, 1)
        final_image = final_image.cpu().permute(0, 2, 3, 1).float().numpy()
        final_image = (final_image * 255).round().astype("uint8")
        
        return Image.fromarray(final_image[0])