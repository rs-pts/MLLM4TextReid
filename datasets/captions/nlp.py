import json
import torch
import random
import nltk
from nltk.corpus import wordnet
from transformers import pipeline

# Download NLTK WordNet (if not already downloaded)
nltk.download('wordnet')

# Load the paraphrasing model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
paraphrase = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base", device=0 if device == "cuda" else -1)

# Function to replace common words with synonyms
def replace_synonyms(sentence):
    words = sentence.split()
    new_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace("_", " ")
            if synonym.lower() != word.lower():  # Avoid replacing with itself
                new_sentence.append(synonym)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    return " ".join(new_sentence)

# Function to generate paraphrased variations with diverse tokens
def augment_caption(caption, num_variations=4):
    variations = []
    for _ in range(num_variations):
        result = paraphrase(caption, max_length=50, num_return_sequences=1, temperature=0.99, top_k=50)
        new_caption = result[0]['generated_text']
        new_caption = replace_synonyms(new_caption)  # Further increase uniqueness
        variations.append(new_caption)
    return variations

# Load the JSON file
with open("ICFG.json", "r") as f:
    data = json.load(f)

# Process each image and augment captions
for image_path, captions in data.items():
    original_caption = captions[0]  # Keep the first caption unchanged
    augmented_captions = augment_caption(original_caption, num_variations=4)
    data[image_path].extend(augmented_captions)  # Append new variations

# Save the updated JSON file
with open("augmented_captions.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ Data augmentation completed with diverse tokens! Saved in 'augmented_captions.json'.")
