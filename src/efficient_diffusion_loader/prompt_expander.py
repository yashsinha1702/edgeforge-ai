import json
import random

class PromptExpander:
    def __init__(self):
        # In a production version, this would connect to Llama-3 or GPT-4.
        # For now, we use a "Rule-Based Ontology" as defined in your plan.
        self.failure_modes = {
            "hard to see": [
                "heavy fog, low visibility, dense mist, atmospheric haze",
                "night time, low light, underexposed, dark shadows",
                "blizzard, heavy snow, whiteout conditions",
                "motion blur, camera shake, blurry image"
            ],
            "weather": [
                "rainstorm, wet surfaces, puddles, overcast sky",
                "harsh sunlight, high contrast, lens flare, glint"
            ],
            "damage": [
                "rusty, corroded surface, peeling paint, metallic texture",
                "dented, scratched, damaged bodywork, broken parts",
                "dirty, muddy, covered in dust, grunge aesthetic"
            ]
        }
        
        # Base quality boosters to ensure photorealism
        self.quality_boosters = "photorealistic, 8k, highly detailed, cinematic lighting, raw photo"



    def generate_variations(self, base_intent, count=5):
        """
        Generates 'count' unique variations of the user's intent.
        """
        variations = []
        print(f"Director: Brainstorming {count} scenarios for '{base_intent}'...")
        
        for i in range(count):
            # Randomly mix and match failure modes
            mode_keys = list(self.failure_modes.keys())
            chosen_category = random.choice(mode_keys)
            chosen_condition = random.choice(self.failure_modes[chosen_category])
            
            # Construct variation
            prompt = f"{base_intent}, {chosen_condition}, {self.quality_boosters}"
            
            # Simplified constraint logic for variation
            constraints = {"condition": chosen_condition}
            
            variations.append({
                "prompt": prompt,
                "constraints": constraints,
                "seed": random.randint(1000, 99999) # Crucial: Different seed for each
            })
            
        return variations
        
    def expand(self, user_intent):
        """
        Translates a vague user intent into a structured EdgeForge directive.
        """
        print(f"Director: Analyzing intent '{user_intent}'...")
        
        # 1. Decompose Intent (Simple Keyword Matching for MVP)
        active_modifiers = []
        
        if "hard to see" in user_intent or "hidden" in user_intent:
            # Pick a random specific challenge (e.g., fog vs night) to ensure diversity
            active_modifiers.append(random.choice(self.failure_modes["hard to see"]))
            
        if "weather" in user_intent or "rain" in user_intent:
            active_modifiers.append(random.choice(self.failure_modes["weather"]))
            
        if "damage" in user_intent or "broken" in user_intent or "old" in user_intent:
            active_modifiers.append(random.choice(self.failure_modes["damage"]))

        # 2. Construct the Final Prompt
        # Combine user's core object + selected failure modes + quality boosters
        # If no specific mode matched, we just pass the intent through (fallback)
        if not active_modifiers:
             scene_description = user_intent
        else:
             scene_description = f"{user_intent}, {', '.join(active_modifiers)}"

        final_prompt = f"{scene_description}, {self.quality_boosters}"

        # 3. Generate Structured Constraints (The 'JSON' output from your plan)
        # These would theoretically control post-processing or camera parameters.
        constraints = {
            "luminance_target": "low" if "night" in final_prompt else "normal",
            "blur_kernel": "high" if "blur" in final_prompt else "none",
            "weather_condition": "rain" if "rain" in final_prompt else "clear"
        }

        return {
            "prompt": final_prompt,
            "constraints": constraints
        }

# Example Usage for testing
if __name__ == "__main__":
    director = PromptExpander()
    directive = director.expand("a car on the street, hard to see")
    print(json.dumps(directive, indent=2))