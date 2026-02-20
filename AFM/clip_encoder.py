# pip install torch torchvision transformers pillow

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

def clip_encode(image_path, text_list):
    # 1. Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 2. Prepare the image and text
    image = Image.open("path_to_your_image.jpg")
    text = ["a photo of a dog", "a photo of a cat"]

    # 3. Process inputs
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    # 4. Forward pass to get embeddings (encoding)
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Extract features
    image_features = outputs.image_embeds    # Shape: [1, 512]
    text_features = outputs.text_embeds      # Shape: [2, 512]

    # 6. Calculate similarity scores (logits)
    logits_per_image = outputs.logits_per_image # Similarity score
    probs = logits_per_image.softmax(dim=1)    # Softmax to get probabilities
    print(probs)

    # 이 probs가 vector 형태로 들어가는건가??? 훔. 