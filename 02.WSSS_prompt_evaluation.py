# 02_prompt_evaluation.py

import torch
import torch.nn.functional as F
import numpy as np
import json
import yaml
from tqdm import tqdm
import argparse
import os
import pickle
import io
from functools import partial

# Import PIL Image and torchvision transforms
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision import datasets

# Import CONCH model utilities
from conch_helper import set_seed, load_precomputed_features, accuracy, generate_weights
from conch.open_clip_custom.factory import create_model_from_pretrained
from conch.open_clip_custom.custom_tokenizer import tokenize, get_tokenizer

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept the loading of torch storage objects
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_prompts(prompts_file):
    """
    Load prompts from a JSON file.

    Args:
        prompts_file (str): Path to the prompts JSON file.

    Returns:
        dict: Dictionary with class names as keys and lists of prompts as values.
    """
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    return prompts

def compute_accuracy(preds, labels):
    """
    Compute classification accuracy.

    Args:
        preds (torch.Tensor): Predicted class indices.
        labels (torch.Tensor): True class indices.

    Returns:
        float: Accuracy in percentage.
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    accuracy = (correct / total) * 100
    return accuracy

def compute_text_embedding(prompt, model, tokenizer_func, device):
    """
    Compute the normalized text embedding for a given prompt.

    Args:
        prompt (str): The text prompt.
        model: The pre-trained CONCH model.
        tokenizer_func: The tokenize function (partial function with tokenizer object).
        device (torch.device): The computation device.

    Returns:
        torch.Tensor: Normalized text embedding.
    """
    # Tokenize the prompt
    text_tokens = tokenizer_func([prompt]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, dim=-1)
    return text_embedding.squeeze(0)  # Shape: [embed_dim]

def filter_prompts_top_k(performances, top_k):
    """
    Select the top_k prompts based on accuracy for each class.

    Args:
        performances (dict): Dictionary with class names as keys and lists of dicts with 'prompt' and 'accuracy'.
        top_k (int): Number of top prompts to select per class.

    Returns:
        dict: Filtered prompts with class names as keys and lists of top_k prompts as values.
    """
    filtered_prompts = {}
    for classname, prompts in performances.items():
        # Sort the prompts by accuracy in descending order
        sorted_prompts = sorted(prompts, key=lambda x: x['accuracy'], reverse=True)
        # Select the top_k prompts
        top_prompts = [prompt['prompt'] for prompt in sorted_prompts[:top_k]]
        filtered_prompts[classname] = top_prompts
    return filtered_prompts

def main(config_path):
    # Load configuration
    with open(config_path, 'r') as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]
#    dataset_name = hparams.get("dataset", "bracs7class")   ############################################# DATASET
#    classnames = hparams.get("class_names", ['Normal Tissue', 'Pathological Benign', 'Usual Ductal Hyperplasia', 'Flat Epithelial Atypia', 'Atypical Ductal Hyperplasia', 'Ductal Carcinoma In Situ', 'Invasive Carcinoma'])   ############################################ Class name
    
#    dataset_name = hparams.get("dataset", "bracs")   ############################################# DATASET
#    classnames = hparams.get("class_names", ["Invasive", "Non invasive"])
    

#    dataset_name = hparams.get("dataset", "private")   ############################################# DATASET
#    classnames = hparams.get("class_names", ["Adipose", "Background", "Debris", "Lymphocytes", "Mucus", "Smooth muscle", "Normal colon mucosa", "Cancer-associated stroma", "Colorectal adenocarcinoma epithelium"])  

#    dataset_name = hparams.get("dataset", "ebhi")   ############################################# DATASET
#    classnames = hparams.get("class_names", ["Benign", "Malignant"])
    
    dataset_name = hparams.get("dataset", "wsss")   ############################################# DATASET
    classnames = hparams.get("class_names", ["Normal", "Stroma", "Tumor"])

    methods = hparams["methods"]

    prompt_eval_results = {}
    filtered_prompts_all_methods = {}

    # Initialize the tokenizer object
    print("Initializing tokenizer...")
    try:
        tokenizer_obj = get_tokenizer()  # Initialize the tokenizer object
    except Exception as e:
        print(f"Failed to initialize tokenizer: {e}")
        raise

    # Create a partial function for tokenization with the tokenizer object
    tokenize_func = partial(tokenize, tokenizer_obj)

    # Load CONCH model and tokenizer
    print(f"Loading {model_size}...")
    model_path = hparams.get(
        "model_path", "/g/data/nk53/mr3328/wca/conch/CONCH/checkpoints/conch/pytorch_model.bin"  # Adjust path as needed
    )
    try:
        model, preprocess = create_model_from_pretrained(
            model_size,
            model_path,
            hf_auth_token="USE YOUR OWN TOKEN",  # Replace with your actual Hugging Face token
        )
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
        """Randomly crops an image within a size range determined by alpha and the image dimensions.

        Args:
            image (Image): The input image to crop.
            alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.

        Returns:
            PIL Image: Cropped image.
        """
        # Get the width and height of the original image
        w, h = image.size
        # Determine the size of the crop based on alpha and the smallest dimension
        n_px = np.random.uniform(low=alpha, high=0.9) * min(h, w)
        # Perform the crop
        cropped = T.RandomCrop(int(n_px))(image)

        return cropped

    def custom_loader(path: str) -> torch.Tensor:
        """Loads an image, applies multi-resolution patch extraction, and returns augmented versions.

        Args:
            path (str): The path to the image file.

        Returns:
            torch.Tensor: A tensor stack of the processed image and its augmented patches.
        """
        # Load the image using the default loader
        img = datasets.folder.default_loader(path)
        # Get scales and number of patches per scale from config
        scales = hparams.get("patch_scales", [0.5, 0.25, .75])
        n_patches_per_scale = hparams.get("n_patches_per_scale", [5, 5, 5])

        all_patches = []

        for scale, n_patches in zip(scales, n_patches_per_scale):
            for _ in range(n_patches):
                # Resize image according to scale
                scaled_size = [int(dim * scale) for dim in img.size]
                scaled_img = img.resize(scaled_size, Image.Resampling.LANCZOS)

                # Random crop from scaled image
                cropped_img = random_crop(scaled_img, alpha=alpha)
                # Process the patch using the model's preprocess
                processed_patch = preprocess(cropped_img)
                all_patches.append(processed_patch)

        # Also include the original image as a patch
        processed_img = preprocess(img)
        all_patches.append(processed_img)

        # Stack all patches into a tensor
        return torch.stack(all_patches)

    # Define num_workers by fetching from hparams or setting a default
    num_workers = hparams.get("num_workers", 4)  # Default to 4 if not specified

    # Pre-compute image features from dataset
    precomputed_features, target, image_features_tensor = load_precomputed_features(
        model=model,
        dataset_name=dataset_name,
        model_size=model_size,
        alpha=alpha,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,  # Use the defined num_workers
        data_path=data_path,
        custom_loader=custom_loader,
        device=device
    )

    # Assign image_features directly without stacking
    image_features = image_features_tensor.to(device)  # Shape: [num_images, embed_dim]

    # Optional: Verify data shapes
    print("Precomputed features shape:", precomputed_features.shape)
    print("Image features shape:", image_features.shape)
    print("Labels shape:", target.shape)  # Corrected variable name

    # Iterate over each method
    for method in methods:
        # Extract method details based on the unique key
        method_key = list(method.keys())[0]
        method_info = method[method_key]
        method_name = method_info["name"]
        enabled = method_info["enabled"]
        prompts_file = method_info.get("prompts_file", None)

        print(f"Method: {method_name}, Enabled: {enabled}, Prompts File: {prompts_file}")

        if not enabled:
            print(f"Method '{method_name}' is disabled. Skipping...")
            continue

        if not prompts_file:
            print(f"No 'prompts_file' specified for method '{method_name}'. Skipping...")
            continue

        if not os.path.exists(prompts_file):
            print(f"Prompts file '{prompts_file}' not found. Skipping method '{method_name}'.")
            continue

        print(f"\nEvaluating method: {method_name}")
        prompts_dict = load_prompts(prompts_file)
        print(f"Loaded prompts for method '{method_name}'. Classes found: {list(prompts_dict.keys())}")

        prompt_eval_results[method_name] = {}

        for classname in classnames:
            if classname not in prompts_dict:
                print(f"Class '{classname}' not found in prompts for method '{method_name}'. Skipping class.")
                continue

            class_prompts = prompts_dict[classname]
            prompt_eval_results[method_name][classname] = []

            for prompt in tqdm(class_prompts, desc=f"Evaluating prompts for class '{classname}'"):
                # Prepare text inputs for all classes
                text_embeddings = []
                for cls in classnames:
                    if cls == classname:
                        formatted_prompt = prompt.format(cls) if '{}' in prompt else prompt
                    else:
                        # Use a default prompt for other classes
                        formatted_prompt = f"A photo of a {cls}."
                    # Compute text embedding using tokenize_func
                    text_embedding = compute_text_embedding(formatted_prompt, model, tokenize_func, device)
                    text_embeddings.append(text_embedding)

                # Stack text embeddings: [num_classes, embed_dim]
                text_embeddings = torch.stack(text_embeddings)

                # Compute logits: [num_images, num_classes]
                logits = image_features @ text_embeddings.T

                # Apply weighted cosine similarity if method is 'ours'
                if method_name == "ours":
                    weights = torch.tensor([0.7, 0.2, 0.1]).to(device)  # Example weights for ['Normal', 'Stroma', 'Tumor']
                    logits = logits * weights  # Element-wise multiplication

                # Compute predictions
                preds = logits.argmax(dim=1)

                # Compute accuracy
                acc = (preds == target).float().mean().item() * 100
                prompt_eval_results[method_name][classname].append({
                    "prompt": prompt,
                    "accuracy": acc
                })

                # Optional: Print or log the result
                print(f"Class '{classname}', Prompt: '{prompt}' - Accuracy: {acc:.2f}%")

        # After evaluating all prompts for the method, select top_k prompts per class
        top_k = hparams.get("top_k", 30)  # Number of top prompts to select per class #######################################################################
        print(f"\nSelecting top {top_k} prompts for method: {method_name}")
        filtered_prompts = filter_prompts_top_k(prompt_eval_results[method_name], top_k)
        filtered_prompts_all_methods[method_name] = filtered_prompts

        # Save the filtered prompts to a new JSON file
        os.makedirs(f"prompts/{dataset_name}/filtered", exist_ok=True)
        with open(f"prompts/{dataset_name}/filtered/{method_name}_filtered.json", "w") as f:
            json.dump(filtered_prompts, f, indent=4)

    # Save evaluation results
    prompt_eval_file = hparams.get("prompt_eval_file", "prompt_evaluation_results.json")
    with open(prompt_eval_file, "w") as f:
        json.dump(prompt_eval_results, f, indent=4)

    # Optionally, save filtered prompts for use in conch_main.py
    with open("filtered_prompts.json", "w") as f:
        json.dump(filtered_prompts_all_methods, f, indent=4)

    # Print a summary
    for method in methods:
        method_key = list(method.keys())[0]
        method_info = method[method_key]
        method_name = method_info["name"]
        if method_name in filtered_prompts_all_methods:
            print(f"\nSummary for method: {method_name}")
            filtered_prompts = filtered_prompts_all_methods[method_name]
            for classname in classnames:
                total_prompts = len(prompt_eval_results[method_name].get(classname, []))
                kept_prompts = len(filtered_prompts.get(classname, []))
                print(f"Class '{classname}': {kept_prompts}/{total_prompts} prompts kept (Top {top_k} prompts).")
        else:
            print(f"\nNo results found for method: {method_name}")

    print(f"\nPrompt evaluation completed. Results saved to '{prompt_eval_file}'.")
    print(f"Filtered prompts saved to 'filtered_prompts.json' and 'prompts/{dataset_name}/filtered/{method_name}_filtered.json'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Evaluation for Weighted Cosine Similarity Classification")
    parser.add_argument('--config', type=str, default='cfgs/conch_wsss_evaluation.yaml', help='Path to the configuration YAML file') ################################################################################ .yaml file
    args = parser.parse_args()
    main(args.config)
