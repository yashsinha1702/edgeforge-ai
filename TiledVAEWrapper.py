import torch
from diffusers import AutoencoderKL
from tqdm import tqdm

class TiledVAEWrapper:
    def __init__(self, vae: AutoencoderKL, tile_size=512, overlap=32):
        self.vae = vae
        self.tile_size = tile_size  # Pixel size of the tile (e.g., 512)
        self.overlap = overlap      # Pixel overlap to prevent seams
        

        self.scale_factor = 8 
        self.latent_tile_size = tile_size // self.scale_factor
        self.latent_overlap = overlap // self.scale_factor

    def decode(self, latents: torch.Tensor):
        """
        Decodes large latents by splitting them into tiles to save VRAM.
        """
        # 1. Initialize the output buffer on CPU (System RAM) to save VRAM
        # "Moving data between VRAM and System RAM"
        batch, channels, height, width = latents.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        
        # Buffer on CPU
        decoded_buffer = torch.zeros(
            (batch, 3, output_height, output_width), 
            device='cpu' 
        )
        
        # Count buffer for averaging overlaps
        count_buffer = torch.zeros(
            (batch, 3, output_height, output_width), 
            device='cpu'
        )

        # 2. Grid Calculation
        # We define a sliding window over the latent space
        grid_rows = self._get_grid(height, self.latent_tile_size, self.latent_overlap)
        grid_cols = self._get_grid(width, self.latent_tile_size, self.latent_overlap)

        # 3. Fractional Batch Loop 
        print(f"Starting Tiled Decode: {len(grid_rows) * len(grid_cols)} tiles...")
        
        for h_start in tqdm(grid_rows):
            for w_start in grid_cols:
                # Define latent crop coordinates
                h_end = min(h_start + self.latent_tile_size, height)
                w_end = min(w_start + self.latent_tile_size, width)
                
                # Adjust start if we hit the edge to keep tile size constant
                h_start = max(0, h_end - self.latent_tile_size)
                w_start = max(0, w_end - self.latent_tile_size)

                # Crop Latent (Fractional Batch)
                latent_tile = latents[:, :, h_start:h_end, w_start:w_end]
                
                # Move to GPU for inference
                latent_tile = latent_tile.to(self.vae.device)

                # Decode the specific tile
                with torch.no_grad():
                    decoded_tile = self.vae.decode(latent_tile).sample

                # Move back to CPU for storage 
                decoded_tile = decoded_tile.to('cpu')

                # Calculate Pixel Coordinates for placement
                px_h_start = h_start * self.scale_factor
                px_h_end = h_end * self.scale_factor
                px_w_start = w_start * self.scale_factor
                px_w_end = w_end * self.scale_factor

                # Accumulate into buffer (Simple addition for now, blending comes next)
                decoded_buffer[:, :, px_h_start:px_h_end, px_w_start:px_w_end] += decoded_tile
                count_buffer[:, :, px_h_start:px_h_end, px_w_start:px_w_end] += 1

        # 4. Normalize by overlap count to blend seams
        final_image = decoded_buffer / count_buffer
        return final_image

    def _get_grid(self, dim_size, tile_size, overlap):
        """Helper to generate start indices"""
        grid = []
        pos = 0
        while pos < dim_size:
            grid.append(pos)
            pos += (tile_size - overlap)
        return grid

    def decode_with_blending(self, latents: torch.Tensor):
        batch, channels, height, width = latents.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        
        # Initialize buffers on CPU
        decoded_buffer = torch.zeros((batch, 3, output_height, output_width), device='cpu')
        weight_buffer = torch.zeros((batch, 3, output_height, output_width), device='cpu')

        # Pre-calculate the Gaussian mask for the TILE size (e.g., 512x512)
        # We do this ONCE to save compute
        tile_mask = self._get_gaussian_mask(self.tile_size, self.tile_size)

        grid_rows = self._get_grid(height, self.latent_tile_size, self.latent_overlap)
        grid_cols = self._get_grid(width, self.latent_tile_size, self.latent_overlap)
        
        print(f"Orchestrating Tiled Decode with Gaussian Blending...")

        for h_start in tqdm(grid_rows):
            for w_start in grid_cols:
                # ... [Coordinate logic remains the same as previous step] ...
                h_end = min(h_start + self.latent_tile_size, height)
                w_end = min(w_start + self.latent_tile_size, width)
                h_start = max(0, h_end - self.latent_tile_size)
                w_start = max(0, w_end - self.latent_tile_size)

                # Crop Latent
                latent_tile = latents[:, :, h_start:h_end, w_start:w_end].to(self.vae.device)

                # Decode
                with torch.no_grad():
                    decoded_tile = self.vae.decode(latent_tile).sample
                
                # --- CRITICAL CHANGE ---
                # Apply the weight mask to the decoded tile
                weighted_tile = decoded_tile * tile_mask

                # Move to CPU
                weighted_tile = weighted_tile.to('cpu')
                mask_cpu = tile_mask.to('cpu')

                # Calculate Pixel Coordinates
                px_h_start = h_start * self.scale_factor
                px_w_start = w_start * self.scale_factor
                px_h_end = px_h_start + self.tile_size
                px_w_end = px_w_start + self.tile_size

                # Accumulate Weighted Tile
                decoded_buffer[:, :, px_h_start:px_h_end, px_w_start:px_w_end] += weighted_tile
                
                # Accumulate Weights (so we can normalize later)
                weight_buffer[:, :, px_h_start:px_h_end, px_w_start:px_w_end] += mask_cpu

        # Normalize: (Tile A * Weight A + Tile B * Weight B) / (Weight A + Weight B)
        # Add small epsilon to avoid division by zero
        final_image = decoded_buffer / (weight_buffer + 1e-7)
        
        return final_image

    def _get_gaussian_mask(self, height, width):
        """
        Creates a 2D Gaussian mask to weight center pixels higher than edge pixels.
        This prevents 'seams' or grid artifacts in the final image.
        """
        # Create a 1D gaussian distribution
        def get_1d_gaussian(size):
            x = torch.arange(size, dtype=torch.float32)
            center = size / 2.0
            sigma = size / 3.0 # Controls the 'spread'. 3.0 is standard.
            return torch.exp(-(x - center)**2 / (2 * sigma**2))

        gauss_h = get_1d_gaussian(height).unsqueeze(1) # Column vector
        gauss_w = get_1d_gaussian(width).unsqueeze(0)  # Row vector
        
        # Outer product to create 2D Gaussian surface
        mask = gauss_h @ gauss_w 
        
        # Reshape to match channel dims (C, H, W)
        return mask.unsqueeze(0).to(self.vae.device)


