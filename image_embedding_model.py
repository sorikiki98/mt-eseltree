import torch
import clip
from PIL import Image


class CLIPImageEncoder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._preprocess = clip.load("ViT-L/14@336px", device=device)
        self._device = device

    def embed_images(self, image_batch):
        image_features = self._model.encode_image(image_batch).detach().cpu()  # [bs, 1024]
        norm_factors = image_features.norm(dim=-1, keepdim=True)
        image_features /= norm_factors
        image_features.to(self._device)

        return image_features

    def preprocess_images(self, image_urls):
        if type(image_urls) is not list:
            images = [Image.open(image_urls)]
        else:
            images = [Image.open(url) for url in image_urls if str(url).split(".")[-1] != "json"]
        image_tensors = torch.stack([self._preprocess(img) for img in images]).to(self._device)

        return image_tensors  # [bs, 3, 336, 336]

    def get_embedding_dimension(self):
        return self._model.visual.output_dim
