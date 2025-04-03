import random
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Set device (multi-GPU support)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load model with GPU acceleration
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 for lower VRAM
    device_map="cuda",  # ✅ Auto-distributes across multiple GPUs
    trust_remote_code=True
).to(device)


templet = {
  "Physical_Appearance": {
    "Gender": ["Male", "Female", "Transgender"],
    "Age_Group": ["Child", "Teen", "Young Adult", "Middle-aged", "Elderly"],
    "Body_Type": ["Slim", "Athletic", "Average", "Heavy-built", "Stocky", "Chubby", "Slender", "Lanky"],
    "Height": ["Short", "Medium", "Tall"],
    "Skin_Tone": ["Fair", "Light", "Medium", "Tan", "Brown", "Dark"],
    "Hair_Length": ["Bald", "Short", "Medium", "Long"],
    "Hair_Type": ["Straight", "Wavy", "Curly", "Frizzy"],
    "Hair_Color": ["Black", "Brown", "Blonde", "Red", "Grey", "White"],
    "Hairstyle": ["Ponytail", "Bun", "Braided", "Side-parted", "Messy"],
    "Facial_Hair": ["Clean-shaven", "Light stubble", "Full beard", "Goatee", "Moustache"],
    "Eyewear": ["None", "Glasses", "Sunglasses"],
    "Facial_Features": ["High cheekbones", "Strong jawline", "Round face", "Oval face", "Defined chin"],
    "Distinctive_Features": ["Scar", "Mole", "Tattoo", "Birthmark", "Freckles", "Piercing"],
    "Posture": ["Upright", "Slouching", "Confident stance", "Hands in pockets"]
  },
  "Clothing_Attributes": {
    "Upper_Body": {
      "Clothing_Type": ["T-shirt", "Shirt", "Jacket", "Hoodie", "Blazer"],
      "Color": ["Black", "White", "Blue", "Red", "Grey", "Green", "Yellow"],
      "Style": ["Striped", "Plain", "Patterned", "Graphic Print"]
    },
    "Lower_Body": {
      "Clothing_Type": ["Jeans", "Pants", "Shorts", "Trousers", "Skirt"],
      "Color": ["Black", "Blue", "Grey", "Beige", "White"],
      "Fit": ["Skinny", "Regular", "Loose"]
    }
  },
  "Accessories": {
    "Headwear": ["Cap", "Hat", "Hood", "Hijab"],
    "Eyewear": ["Glasses", "Sunglasses"],
    "Jewelry": ["Necklace", "Bracelet", "Ring", "Earrings", "Wristwatch"],
    "Bags": ["Backpack", "Handbag", "Shoulder Bag", "Crossbody Bag", "Fanny Pack", "Briefcase"],
    "Gloves": ["None", "Leather Gloves", "Wool Gloves"],
    "Footwear": ["Sneakers", "Boots", "Sandals", "Loafers", "Heels", "Slippers"]
  },
  "Occlusions": {
    "Face_Occlusion": ["None", "Mask", "Scarf", "Helmet", "Veil", "Sunglasses"],
    "Body_Occlusion": ["None", "Seated", "Crouched", "Covered by other person"]
  }
}


# text = f"""Analyze the given image and generate a concise description of a person's overall appearance in a structured format. Follow these steps:
# 1. Detect if a person is present. If not, return "None".
# 2. Extract only the visible visual attributes: {att[0]}, {att[1]}, {att[2]}, {att[3]}, and {att[4]}.
# 3. Select the best-matching template from the provided list: "{templates}".
# 4. Generate a structured response of up to 77 words, ensuring all details are strictly derived from the image.
# 5. If an attribute is not visible, ignore it and do not generate imagined content."""

# ✅ Function to generate rewrites on GPU
def generate_rewrites_deepseek(text, num_variants=random.randint(4, 12)):
#     prompt = f"""Rewrite the following sentence in different ways while preserving its meaning:
# Original: {text}
    prompt = f"""Rewrite the given caption: '{text}' while increasing its lexical diversity and structuring it effectively.  

**Guidelines:**  
1. Rephrase the sentence while maintaining the original meaning.  
2. Ensure the new caption is **concise (max 77 tokens)** and **rich in descriptive vocabulary**.  
3. Incorporate person attributes such as **physical appearance, clothing, and accessories**.  
4. If the original caption **starts with 'A,' modify its structure** while keeping the meaning intact.  
5. **Output only the rephrased caption—do NOT include instructions or explanations.**  
"""  

#     prompt = f"""Rewrite the following sentence {text} in diffrenct way while taking the refrence form provided {templet} is in json format and use reid attributes such as 
#     Physical_Appearance, Clothing_Attributes, Accessories, etc and also ensures that the rewritten upto 77 tokens should be unique token and last make atleast 2 complete caption.: 

# Rewrites:
# 1."""
  

    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # ✅ Move tensors to GPU

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=77,  # ✅ Reduce max tokens for speed
            temperature=0.99,  # ✅ Lower temp for faster inference
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Extract rewrites
    rewrites = []
    lines = generated_text.split('\n')
    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            rewrite = line.strip().split(".", 1)[1].strip()
            rewrites.append(rewrite)
            if len(rewrites) >= num_variants:
                break
    
    return rewrites

# Read captions from captions.json
# /home/dslab/Documents/s24015/MLLM4Text-ReID/data/RSTPReid/data_captions.json
# /home/dslab/Documents/s24015/MLLM4Text-ReID/data/ICFG-PEDES/ICFG-PEDES.json
with open('/home/dslab/Documents/s24015/MLLM4Text-ReID/ICFG.json', 'r') as f:
    captions_data = json.load(f)
    
try:
    with open('./ICFG_rewritten_captions_r.json', 'r') as f:
        rewritten_captions = json.load(f)
except FileNotFoundError:
    print("No existing rewritten_captions.json found. Creating a new one.")
    rewritten_captions = {}
# Process each caption and generate rewrites
for image_path, captions in captions_data.items():
    if image_path in rewritten_captions:
        print(f"Skipping image '{image_path}' as rewrites already exist")
        continue
    rewritten_captions[image_path] = captions
    try:
        rewrites = generate_rewrites_deepseek(text =captions[0],num_variants= random.randint(3, 13))
        
        rewritten_captions[image_path].extend(rewrites)
        with open('./ICGF_rewritten_captions_r.json', 'w') as f:
            json.dump(rewritten_captions, f, indent=4)
    except Exception as e:
        print(f"Error generating rewrites for caption '{captions}': {str(e)}")

# Save rewritten captions to a new JSON file
# with open('ICFG_rewritten_captions.json', 'w') as f:
#     json.dump(rewritten_captions, f, indent=4)

print("Rewritten captions saved to rewritten_captions.json")
