from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

folder_path = "images"

for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    
    image = Image.open(img_path).convert('RGB')
    
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"{img_name} → {caption}")