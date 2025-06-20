from django.shortcuts import render
from django.conf import settings
import os
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .forms import UploadImageForm

# Load CLIP model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Precompute cartoon embeddings once
CARTOON_DIR = os.path.join(settings.MEDIA_ROOT, 'cartoon_images')
cartoon_names = []
cartoon_embeddings = []

# Load all cartoon images and store embeddings
for filename in os.listdir(CARTOON_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(CARTOON_DIR, filename)
        try:
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img).cpu().numpy()[0]  # (512,)
            cartoon_embeddings.append(emb)
            cartoon_names.append(filename)
        except Exception as e:
            print(f"Failed to process cartoon image {filename}: {e}")

# Convert list to a single NumPy array for fast cosine similarity
cartoon_embeddings = np.stack(cartoon_embeddings)  # shape: (N, 512)

def index(request):
    result = None
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']

            # Save uploaded image
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, image_file.name)
            with open(image_path, 'wb+') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Load and process uploaded image
            try:
                input_img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    input_embedding = model.encode_image(input_img).cpu().numpy()[0]  # shape: (512,)
            except Exception as e:
                return render(request, 'match/index.html', {
                    'form': form,
                    'error': f"Failed to process uploaded image: {e}"
                })

            # Compute cosine similarities using NumPy
            input_norm = input_embedding / np.linalg.norm(input_embedding)
            cartoon_norms = cartoon_embeddings / np.linalg.norm(cartoon_embeddings, axis=1, keepdims=True)
            similarities = np.dot(cartoon_norms, input_norm)  # shape: (N,)
            best_index = np.argmax(similarities)

            result = {
                'input_img': image_file.name,
                'match_img': cartoon_names[best_index],
                'similarity': round(similarities[best_index] * 100, 2)
            }

    else:
        form = UploadImageForm()

    return render(request, 'match/index.html', {
        'form': form,
        'result': result,
        'MEDIA_URL': settings.MEDIA_URL  # optional for template use
    })
