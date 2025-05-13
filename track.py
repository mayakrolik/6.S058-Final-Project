import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.decomposition import PCA
import os
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import umap
import clip
import dlib

# Function to load and preprocess image
def preprocess_image(image_path, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Function to extract embeddings from ResNet18
def get_image_embeddings(image_paths):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = resnet18(pretrained=True)
    model.to(device)
    model.eval()
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer

    embeddings = []
    for path in image_paths:
        img_tensor = preprocess_image(path).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().cpu().numpy()
            embeddings.append(embedding)
    # print(embeddings)
    return np.array(embeddings)

def get_dlib_face_embeddings(image_paths):
    face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")

    embeddings = []
    for path in image_paths:
        img = dlib.load_rgb_image(path)
        dets = detector(img)
        if dets:
            shape = sp(img, dets[0])
            face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
            embeddings.append(np.array(face_descriptor))
        else:
            embeddings.append(np.zeros(128))  # no face found
    return embeddings

def get_clip_embeddings(image_paths):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    embeddings = []
    for path in image_paths:
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).squeeze().cpu().numpy()
            embeddings.append(embedding)
    return embeddings

# Function to visualize embeddings with image thumbnails
def plot_embeddings_with_thumbnails(embeddings, image_paths):
    n_samples = len(embeddings)
    # print(n_samples)
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity)
    reduced = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("t-SNE of Image Embeddings (FaceNet)")

    for i, (x, y) in enumerate(reduced):
        img = Image.open(image_paths[i]).resize((30, 30))
        ax.imshow(img, extent=(x - 1, x + 1, y - 1, y + 1), zorder=1)

    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.0)  # anchor points
    plt.show()

def plot_embeddings_umap(embeddings, image_paths):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("UMAP of Image Embeddings (FaceNet)")

    for i, (x, y) in enumerate(reduced):
        img = Image.open(image_paths[i]).resize((30, 30))
        ax.imshow(img, extent=(x - 1, x + 1, y - 1, y + 1), zorder=1)

    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.0)
    plt.show()

def plot_embeddings_pca(embeddings, image_paths):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("PCA of Image Embeddings")

    for i, (x, y) in enumerate(reduced):
        img = Image.open(image_paths[i]).resize((30, 30))
        ax.imshow(img, extent=(x - 1, x + 1, y - 1, y + 1), zorder=1)

    ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.0)
    plt.show()

if __name__ == "__main__":
    final_dir = "processed_llm_videos"
    image_paths = glob.glob(f"from_playlist/detected_faces_scrapped/clip_*_frame_face0.png")
    image_paths.append("oranges.png")
    # print(image_paths)
    embeddings = get_dlib_face_embeddings(image_paths)
    plot_embeddings_umap(embeddings, image_paths)