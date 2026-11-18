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