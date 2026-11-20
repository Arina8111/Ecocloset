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