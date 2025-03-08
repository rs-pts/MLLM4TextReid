import os
import json
import time 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

import os

os.chdir("/storage/avinash/ReId/MLLM4Text-ReID")
print("Current Working Directory:", os.getcwd())
# Set random seed for reproducibility
torch.manual_seed(1234)

# Load model & tokenizer
model_path = 'Qwen/Qwen-VL-Chat'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# use cuda device
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:1", trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:4", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda:6", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()

model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)


templates = [
    "Wearing [clothing description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Sporting [hair description], the [person/woman/man] is dressed in [clothing description] and is carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "In [clothing description] and [footwear description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] and is also carrying [belongings description].",
    "Carrying [belongings description], the [person/woman/man] is dressed in [clothing description] and [footwear description].",
    "In [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "Carrying [belongings description], the [person/woman/man] is wearing [clothing description] and [footwear description].",
    "In [clothing description] and [accessory description], the [person/woman/man] is also carrying [belongings description].",
    "With [hair description], the [person/woman/man] is dressed in [clothing description] and [accessory description].",
    "Sporting [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description] and [accessory description].",
    "With [hair description], the [person/woman/man] is wearing [clothing description] with [accessory description].",
    "In [clothing description] and [accessory description], the [person/woman/man] also has [hair description].",
    "In [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "With [accessory description], the [person/woman/man] also has [hair description] and is carrying [belongings description].",
    "Wearing [clothing description] and [footwear description], the [person/woman/man] also has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [footwear description], the [person/woman/man] is wearing [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] sports [hair description] and is dressed in [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Wearing [footwear description], [accessory description], [clothing description], the [person/woman/man] is also carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "Dressed in [footwear description], [accessory description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is attired in [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description].",
    "In [accessory description], [footwear description], [clothing description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] is seen wearing [clothing description], [footwear description], [accessory description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "Sporting [hair description], the [person/woman/man] is wearing [footwear description], [clothing description], [accessory description], and carrying [belongings description].",
    "The [person/woman/man] is seen in [footwear description], [accessory description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] can be spotted wearing [accessory description], [footwear description], [clothing description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] has [hair description] and is dressed in [accessory description], [footwear description], [clothing description], and carrying [belongings description].",
    "The [person/woman/man] is attired in [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "With [hair description], the [person/woman/man] is wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description].",
    "Dressed in [accessory description], [clothing description], [footwear description], and carrying [belongings description], the [person/woman/man] has [hair description].",
    "The [person/woman/man] can be seen wearing [accessory description], [clothing description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is dressed in [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description].",
    "The [person/woman/man] is wearing [clothing description], [accessory description], [footwear description], and carrying [belongings description]. The [person/woman/man] has [hair description]."
] 
# Define attributes and text template
att = ['clothing', 'shoes', 'hairstyle', 'gender', 'belongings']

# text = f'Write a description about the overall appearance of the person in the image, including the attributions: {att[0]}, {att[1]}, {att[2]}, {att[3]} and {att[4]}. If any attribute is not visible, you can ignore. Do not imagine any contents that are not in the image.'
# text = f"""Generate a description about the overall appearance of the person, must include the {att[0]}, {att[1]}, {att[2]}, {att[3]} and {att[4]}, in a style similar to template:""{templates}"". select only one best matching template and generate response upto 77 words only. If some requirements in the template are not visible, you must ignore. Do not imagine any contents that are not in the image."""
#text_template = f'Generate a description about the overall appearance of the person, including {", ".join(att)}, in a style similar to the given templates. If some details are not visible, you can ignore them. Do not imagine details not in the image.'

# text = f"""Generate a concise description of a person's overall appearance using a structured format. Ensure the description include only visual attributes {att[0]}, {att[1]}, {att[2]}, {att[3]} and {att[4]}. Select the best-matching template from the provided list of template :""{templates}"" and generate a response of up to 77 words. If any attribute is not visible in the image, you must ignore. Do not imagine any contents that are not in the image . if person is not visible in the image return "None" as response"""

text = f"""Analyze the given image and generate a concise description of a person's overall appearance in a structured format. Follow these steps:
1. Detect if a person is present. If not, return "None".
2. If an attribute is not visible, ignore it and do not generate imagined content.
3. Extract only the visible visual attributes: {att[0]}, {att[1]}, {att[2]}, {att[3]}, and {att[4]}.
4. Select the best-matching template from the provided list: "{templates}" based on detected attributes on the image.
5. Generate a structured response of up to 77 words, ensuring all details are strictly derived from the image."""
# Image directory & output file
image_folder = "/storage/public_datasets/LUPerson/imgss/china"
output_file = "china_test.json"

# Load existing captions if the file exists
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        try:
            captions_dict = json.load(f)
        except json.JSONDecodeError:
            captions_dict = {}  # Handle corrupted JSON files
else:
    captions_dict = {}

# Walk through all files in the directory
idx = 0 
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image extensions
            image_path = os.path.join(root, file)
            # if idx ==5: 
            #     break
            # idx += 1

            # Skip already processed images
            if image_path in captions_dict:
                print(f"skiping, file exist.")
                continue  

            print(f"Processing: {image_path}")

            try:
                # Prepare input for the model
                query = tokenizer.from_list_format([
                    {'image': image_path},
                    {'text': text},
                ])

                # Generate caption
                start_time = time.time()
                caption, _ = model.chat(tokenizer, query=query, history=None)
                end_time = time.time()
                print(f"time taken by generating caption delta t : {start_time - end_time}")

                # Store in dictionary
                captions_dict[image_path] = caption

                # **Append to JSON file after each new caption** (avoids loss in case of interruption)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(captions_dict, f, indent=4, ensure_ascii=False)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

print(f"\n✅ Captions updated in {output_file}")