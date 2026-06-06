from django.shortcuts import render
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import torch
from .models import CaptionHistory, SearchHistory


# ---------- LOAD MODELS ONCE (BLIP + CLIP) ----------

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# ---------- HELPER: GET ALL IMAGE PATHS ----------

def get_all_image_paths(root_folder):
    """
    Recursively collect all image paths under root_folder.
    Supports jpg / jpeg / png / webp / bmp.
    """
    image_paths = []
    for root_dir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                full_path = os.path.join(root_dir, file)
                image_paths.append(full_path)
    return image_paths


# ---------- UPDATED CLIP SEARCH WITH FILTERING ----------

def clip_search(text_query):
    """
    Hybrid search:
      1. Read user query (case-insensitive).
      2. Optionally filter search space based on keywords
         (cars / animals / food, model names, colors).
      3. Within filtered images, use CLIP to pick best match.
    """

    root_folder = "media/search_images"
    all_image_paths = get_all_image_paths(root_folder)

    if not all_image_paths:
        return None

    # Normalize query to lowercase for parsing
    q = (text_query or "").lower().strip()
    if not q:
        return None

    # Start with all images as candidates
    candidates = all_image_paths

    def apply_filter(paths, substrings):
        """
        Keep only paths that contain ANY of the given substrings (case-insensitive).
        If filtering gives empty result, return original paths (fail-safe).
        """
        filtered = []
        for p in paths:
            lp = p.lower()
            if any(sub in lp for sub in substrings):
                filtered.append(p)
        return filtered if filtered else paths

    # --- CATEGORY FILTERS (cars / animals / food) ---

    # cars related words
    if any(word in q for word in ["car", "cars", "bmw", "porsche", "m2", "m3", "m5", "911", "macan"]):
        candidates = apply_filter(candidates, ["cars"])

    # animals related words
    if any(word in q for word in ["animal", "animals", "cat", "cats", "dog", "dogs",
                                  "persian", "sphinx", "husky", "shih", "retriever"]):
        candidates = apply_filter(candidates, ["animals"])

    # food related words
    if any(word in q for word in ["food", "biryani", "pizza", "cake", "bread", "idli"]):
        candidates = apply_filter(candidates, ["food"])

    # --- MODEL / SUB-FOLDER FILTERS (BMW models & Porsche models) ---

    if "m2" in q:
        candidates = apply_filter(candidates, ["m2"])
    if "m3" in q:
        candidates = apply_filter(candidates, ["m3"])
    if "m5" in q:
        candidates = apply_filter(candidates, ["m5"])
    if "911" in q:
        candidates = apply_filter(candidates, ["911"])
    if "macan" in q:
        candidates = apply_filter(candidates, ["macan"])
    if "bmw" in q:
        candidates = apply_filter(candidates, ["bmw"])
    if "porsche" in q:
        candidates = apply_filter(candidates, ["porsche"])

    # --- COLOR FILTERS (filename or path contains color keyword) ---
    color_words = ["blue", "red", "green", "yellow", "black", "white", "orange"]
    for color in color_words:
        if color in q:
            candidates = apply_filter(candidates, [color])

    # At this point, "candidates" is a narrowed set of image paths
    # Now we let CLIP decide the best one among them.

    # Convert text query to embedding
    text_inputs = clip_processor(text=[q], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    best_sim = None
    best_path = None

    for img_path in candidates:
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue  # skip any broken/unreadable image

        image_inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sim = torch.nn.functional.cosine_similarity(text_features, image_features).item()

        if (best_sim is None) or (sim > best_sim):
            best_sim = sim
            best_path = img_path

    if best_path is None:
        return None

    # Normalize path to use forward slashes for URLs
    best_path = best_path.replace("\\", "/")  # Windows safety
    return best_path  # e.g. "media/search_images/cars/bmw/M2/blue.webp"


# ---------- MAIN VIEW ----------

def home(request):
    image_url = None
    caption = None
    search_result = None

    if request.method == "POST":

        # 1️⃣ IMAGE UPLOAD + CAPTION (BLIP)
        if "image" in request.FILES:
            img = request.FILES["image"]
            img_name = img.name

            # Ensure media/ exists
            media_path = "media"
            if not os.path.exists(media_path):
                os.makedirs(media_path)

            save_path = os.path.join(media_path, img_name)
            with open(save_path, "wb+") as f:
                for chunk in img.chunks():
                    f.write(chunk)

            # Show uploaded image
            image_url = "/media/" + img_name

            # Generate caption using BLIP
            pil_image = Image.open(save_path).convert("RGB")
            inputs = blip_processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                output = blip_model.generate(**inputs, max_length=20)
            caption_text = blip_processor.decode(output[0], skip_special_tokens=True)
            caption = caption_text

            try:
                img.seek(0)  # important because you already read chunks earlier
                CaptionHistory.objects.create(image=img, caption=caption)
            except Exception as e:
                print("DB save error (CaptionHistory):", e)

            

        # 2️⃣ TEXT SEARCH USING CLIP
        if "query" in request.POST:
            query = request.POST.get("query", "")
            if query.strip() != "":
                result_path = clip_search(query)

                try:
                    SearchHistory.objects.create(query=query, result_image_path=result_path or "")
                except Exception as e:
                    print("DB save error (SearchHistory):", e) 
                       
                if result_path is not None:
                    # result_path like "media/search_images/...."
                    # browser expects "/media/..."
                    if result_path.startswith("media/"):
                        search_result = "/" + result_path
                    else:
                        # fall back just in case
                        search_result = "/" + result_path

    return render(request, "index.html", {
        "image_url": image_url,
        "caption": caption,
        "search_result": search_result,
    })