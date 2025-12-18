# from efficient_diffusion_loader import EdgeForgePipeline

# # Initialize the Engine
# forge = EdgeForgePipeline()

# # 1. Input: The "Constraint" (Layout)
# # Load your reference image to extract edges
# control_image = forge.preprocess_canny("194.jpg")
# control_image.save("debug_edges.png") # Save to see what the AI 'sees'

# # 2. Input: The "Intent" (Prompt)
# # This is what the LLM Agent (Module 1) will eventually generate.
# prompt = "photorealistic, cinematic lighting, 8k, rusty ruined texture, moss growing, post-apocalyptic atmosphere"
# negative_prompt = "cartoon, low quality, blur, watermark"

# # 3. Generate
# result = forge.generate(prompt, control_image, seed=42)

# # 4. Save
# result.save("output_edgeforge.png")
# print("Generation Complete. Check 'output_edgeforge.png'")


from efficient_diffusion_loader import EdgeForgePipeline, PromptExpander

# 1. Initialize Components
director = PromptExpander()
forge = EdgeForgePipeline()

# 2. The User Input (Vague)
user_request = "a rusty car, hard to see"

# 3. The Director's Interpretation (Module 1)
directive = director.expand(user_request)
print(f"\nGenerated Prompt: {directive['prompt']}")
print(f"Constraints: {directive['constraints']}")

# 4. The Artist's Execution (Module 2 + 4)
control_image = forge.preprocess_canny("input_layout.jpg")

result = forge.generate(
    prompt=directive['prompt'], # Use the expanded prompt
    control_image=control_image, 
    seed=42
)

result.save("output_director_test.png")
print("Saved 'output_director_test.png'")