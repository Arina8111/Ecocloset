from flask import Flask, render_template, request, jsonify
import base64
import io
import pandas as pd
from suggestion import get_recommendation
import torch
from PIL import Image, UnidentifiedImageError
from transformers import ViTForImageClassification, ViTImageProcessor
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from io import BytesIO
import time
app = Flask(__name__)

model_fabric = models.resnet18()
model_fabric.fc = nn.Linear(model_fabric.fc.in_features, 24)
model_fabric.load_state_dict(torch.load("model/fabric.pth", map_location="cpu"))
model_fabric.eval()

model_name = "google/vit-base-patch16-224"
model_cloth = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=6,
    ignore_mismatched_sizes=True,
)

model_cloth.load_state_dict(torch.load("model/Eshita_model_new.pth", map_location="cpu"))
model_cloth.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

processor = ViTImageProcessor.from_pretrained(model_name)

labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Shirt"
]

classes = [
    'Wool', 'Viscose', 'Uclassified', 'Utilities', 'Velvet',
    'Terrycloth', 'Silk', 'Suede', 'Polyester', 'Nylon', 'Linen', 'Lut', 'Leather',
    'Fleece', 'Felt', 'Denim', 'Crepe', 'Cotton', 'Corduroy', 'Chenille', 'Blended',
    'Acrylic', 'Artificial fur', 'Artificial leather'
]


def predict(image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model_cloth(**inputs)

    logits = outputs.logits
    pred = logits.argmax(-1).item()

    return labels[pred]


@app.route("/", methods=['POST'])
def form_post():
    score = 0
    df = pd.read_csv('data.csv')

    if request.method == 'POST':
        url = request.form.get('image_url')
        uses = request.form.get('no_of_uses')

        clothing_type = ""
        fabric_type = ""
        uploaded_image_src = None

        def cloth_api(image):
            return predict(image)

        def fabric_api(image):
            img = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model_fabric(img)
                pred = torch.argmax(output, dim=1).item()
            return classes[pred]

        file = request.files.get('file')
        try:
            if not file or not file.filename:
                raise ValueError("No file uploaded")
            image_bytes = file.read()
            if not image_bytes:
                raise ValueError("Empty file")
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
            uploaded_image_src = f"data:{file.mimetype or 'image/jpeg'};base64,{encoded_image}"

            fabric_type = fabric_api(image)
            clothing_type = cloth_api(image)
        except (AttributeError, UnidentifiedImageError, OSError, ValueError):
            if url and url.strip():
                try:
                    target_url = url.strip()
                    image_bytes = None
                    content_type = "image/jpeg"

                    try:
                        resp = requests.get(target_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                        if resp.status_code == 200 and 'image' in resp.headers.get('Content-Type', '').lower():
                            image_bytes = resp.content
                            content_type = resp.headers.get('Content-Type', 'image/jpeg')
                    except Exception:
                        pass

                    if not image_bytes:
                        def get_image(target):
                            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
                            try:
                                driver.get(target)
                                time.sleep(3)
                                img = driver.find_element(By.ID, "landingImage")
                                return img.get_attribute("src")
                            except Exception as e:
                                print("Error:", e)
                                return None
                            finally:
                                driver.quit()

                        img_src = get_image(target_url)
                        if img_src:
                            resp = requests.get(img_src, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                            if resp.status_code == 200:
                                image_bytes = resp.content
                                content_type = resp.headers.get('Content-Type', 'image/jpeg')

                    if image_bytes:
                        image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        fabric_type = fabric_api(image)
                        clothing_type = cloth_api(image)
                        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                        uploaded_image_src = f"data:{content_type};base64,{encoded_image}"
                except Exception as e:
                    print("Error loading image from URL:", e)

        display_fabric = fabric_type.strip() if fabric_type else "Unclassified"
        display_clothing = clothing_type.strip() if clothing_type else "Unclassified"

        scorerow = df[df['fabric'].str.strip().str.lower() == display_fabric.lower()]
        if scorerow.empty:
            norm_target = display_fabric.lower().replace(" ", "_").replace("-", "_")
            scorerow = df[df['fabric'].str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_") == norm_target]

        if scorerow.empty and display_fabric.lower() in ['uclassified', 'utilities']:
            scorerow = df[df['fabric'].str.strip().str.lower() == 'unclassified']

        if scorerow.empty:
            scorerow = df[df['fabric'].str.strip().str.lower() == 'unclassified']

        if not scorerow.empty:
            if 'total_score' in scorerow.columns:
                score = int(scorerow['total_score'].iloc[0])
            elif len(scorerow.columns) > 6:
                score = int(scorerow.iat[0, 6])
            else:
                score = int(scorerow.iloc[0, -1])
            durability = int(scorerow['durability'].iloc[0]) if 'durability' in scorerow.columns else (int(scorerow.iat[0, 5]) if len(scorerow.columns) > 5 else 50)
        else:
            score = 50
            durability = 50

        advice = get_recommendation(display_fabric, score, uses)
        return render_template(
            "result.html",
            FabricType=display_fabric,
            Clothingtype=display_clothing,
            s_score=score,
            durability=durability,
            suggest=advice,
            uploaded_image_src=uploaded_image_src,
        )

