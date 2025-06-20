# 🎭 Find My Cartoon Twin

A fun and intelligent web application that finds your cartoon look-alike using image similarity powered by OpenAI CLIP and Django.

## 🚀 Features

- 📷 Upload a human photo and find the closest matching cartoon character
- 🧠 Uses OpenAI CLIP model to match facial features and aesthetics
- 🤖 Automatic face detection and preprocessing
- 🖼️ Handles single and multiple faces in an image
- 📁 Simple and clean web interface with Bootstrap
- 🗃️ Admin panel and user image history (optional future scope)

## 🧰 Tech Stack

- **Backend**: Django (Python)
- **Machine Learning**: CLIP (Contrastive Language–Image Pretraining by OpenAI)
- **Frontend**: HTML, CSS, Bootstrap
- **Database**: MySQL (or SQLite for local dev)
- **Tools**: PIL, Torch, sklearn, Django Forms

## 📷 How It Works

1. User uploads a photo.
2. CLIP model generates embeddings for the input and cartoon dataset.
3. Cosine similarity is computed to find the closest cartoon match.
4. Matched result is displayed on the screen.

## 🔧 Setup Instructions

### Prerequisites

- Python 3.9+
- pip
- virtualenv (recommended)
