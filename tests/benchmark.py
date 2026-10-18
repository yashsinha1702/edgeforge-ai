import torch
import gc
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import time


from tiled_vae import TiledVAEWrapper 



def measure_vram(func, *args, **kwargs):
    """
    Helper to measure Peak VRAM usage of a function.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    peak_mem = torch.cuda.max_memory_allocated()
    used_mem = peak_mem - start_mem
    
    return result, used_mem / (1024**3), end_time - start_time # Returns GB

def run_benchmark():
    print("--- EdgeForge AI: Module 4 Benchmark ---")
    
    # 1. Setup: Load a standard SDXL VAE
    print("Loading VAE model (SDXL)...")
    model_id = "stabilityai/sdxl-vae"
    try:
        # float32, like the main pipeline: TiledVAEWrapper casts each latent tile to
        # float32 (an fp16 VAE would fail with a dtype mismatch, and the SDXL VAE
        # produces NaN / black tiles in fp16).
        vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32).to("cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have internet access or the model cached.")
        return

    # 2. Simulate a Massive Image (4K Resolution)
    # 4096 x 4096 pixels -> Latents are 1/8th size -> 512 x 512
    # Batch size 1, 4 channels (standard for SDXL)
    target_res = 4096
    latent_res = target_res // 8
    print(f"Simulating Generation of {target_res}x{target_res} image...")
    
    # Create random noise latents (simulating the output of the Diffusion Unet)
    latents = torch.randn((1, 4, latent_res, latent_res), dtype=torch.float16).to("cuda")
    
    # 3. Test 1: Standard Decode (DANGEROUS - Might OOM)
    # We will skip the actual run to save your GPU, but theoretically calculate it.
    # A 4K float16 tensor takes ~250MB. But the intermediate states in VAE attention 
    # blocks can balloon to 20GB+.
    print(f"\n[Theoretical] Standard Decode estimate: >24 GB VRAM required.")
    
    # 4. Test 2: EdgeForge Tiled Decode
    print(f"\n[Actual] Running Tiled Decode...")
    
    # Initialize your wrapper
    tiled_wrapper = TiledVAEWrapper(vae, tile_size=512, overlap=32)
    
    # Run and measure
    decoded_image, vram_usage, duration = measure_vram(tiled_wrapper.decode_with_blending, latents)
    
    print(f"Success!")
    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Peak VRAM Added: {vram_usage:.2f} GB")
    
    # 5. Verification: Save the output
    print("\nSaving debug image...")
    # Convert from [-1, 1] range to [0, 255] uint8
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
    decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()
    decoded_image = (decoded_image * 255).round().astype("uint8")
    
    img = Image.fromarray(decoded_image[0])
    img.save("benchmark_4k_output.png")
    print("Saved to 'benchmark_4k_output.png'. Check this file for grid seams.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_benchmark()
    else:
        print("Benchmarks require a GPU (CUDA).")