from efficient_diffusion_loader import EdgeForgePipeline

# Initialize the Engine
forge = EdgeForgePipeline()

# 1. Input: The "Constraint" (Layout)
# Load your reference image to extract edges
control_image = forge.preprocess_canny("194.jpg")
control_image.save("debug_edges.png") # Save to see what the AI 'sees'

# 2. Input: The "Intent" (Prompt)
# This is what the LLM Agent (Module 1) will eventually generate.
prompt = "photorealistic, cinematic lighting, 8k, rusty ruined texture, moss growing, post-apocalyptic atmosphere"
negative_prompt = "cartoon, low quality, blur, watermark"

# 3. Generate
result = forge.generate(prompt, control_image, seed=42)

# 4. Save
result.save("output_edgeforge.png")
print("Generation Complete. Check 'output_edgeforge.png'")