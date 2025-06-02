import fire
import numpy as np
import torch
import yaml
from MRPHE_helper import (
    accuracy,
    generate_weights,
    load_precomputed_features,
    set_seed,
)
from torchvision import datasets, transforms
from conch.open_clip_custom.factory import create_model_from_pretrained
from conch.open_clip_custom.custom_tokenizer import tokenize, get_tokenizer
from torchvision.transforms import v2 as T
from torchvision import datasets
from torch.nn import functional as F
from PIL import Image
#from sklearn.metrics import confusion_matrix
import random
from torchvision.transforms import functional as TF
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

def print_confusion_matrix(cm, class_names):
    """Print the confusion matrix in a readable format."""
    print("Confusion Matrix:")
    print("Classes:", class_names)
    for i, row in enumerate(cm):
        print(f"{class_names[i]}: {row}")


def main(
    config: str = "cfgs/conch_heghids.yaml",
    num_workers: int = 4,
    seed: int = 42,
    device: str = "cuda",
):
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    # Load config file
    with open(config) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = hparams.get("dataset", "heghids")

    set_seed(seed)

    # Load hyperparameters from config file
    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    # Load model
    print(f"Loading {model_size}")
    model_path = hparams.get(
        "model_path", "/g/data/nk53/mr3328/wca/conch/CONCH/checkpoints/conch/pytorch_model.bin"
    )
    try:
        model, preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            model_path,
            hf_auth_token="HF_TOKEN",
        )
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        raise
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)




    def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
        """
        Randomly crops an image within a size range determined by alpha and the image dimensions.
    
        Args:
            image (Image.Image): The input image to crop.
            alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.
    
        Returns:
            Image.Image: Cropped image.
        """
        # Get the width and height of the original image
        w, h = image.size
        
        # Set the lower and upper bounds for the crop size based on the alpha parameter
        lower_bound = alpha * min(w, h)   # Minimum crop size
        upper_bound = 0.9 * min(w, h)     # Maximum crop size (90% of the smallest dimension)
        
        # Randomly choose a crop size between lower and upper bounds
        crop_size = int(np.random.uniform(low=lower_bound, high=upper_bound))
        
        # Generate random cropping parameters (top-left corner)
        crop_params = transforms.RandomCrop.get_params(image, (crop_size, crop_size))
        
        # Perform the crop using torchvision's functional API
        cropped_image = TF.crop(image, *crop_params)
    
        return cropped_image

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
        scales = hparams.get("patch_scales", [0.25, 0.50, 0.75])
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

    # Pre-compute image features from dataset
    # Pre-compute image features from dataset (only once)
    precomputed_features, target, image_features_tensor = load_precomputed_features(
        model=model,
        dataset_name=dataset_name,
        model_size=model_size,
        alpha=alpha,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        data_path=data_path,
        custom_loader=custom_loader,
        device=device
    )

    # Assign image_features directly without stacking
    image_features = image_features_tensor.to(device)  # Shape: [batch_size, embed_dim]

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            text_scale = (
                torch.exp(torch.tensor(method["text_scale"])).to(device)
                if "text_scale" in method
                else None
            )
            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            if method_enabled:
                zeroshot_weights = generate_weights(
                    method_name,
                    model=model,
                    dataset_name=dataset_name,
                    tt_scale=text_scale,
                    device=device,
                )
                # Set zero-shot weights to the same dtype as image features
                zeroshot_weights = zeroshot_weights.to(image_features.dtype)
                # Normalize zeroshot_weights
                zeroshot_weights_norm = F.normalize(zeroshot_weights, dim=0)
            else:
                continue

            # Normalize image features
            image_features_norm = F.normalize(image_features, dim=1)
            
            
            
            
            # Initialize lists to store metrics added
            acc_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            cm_list = []
            
            
            # Baseline
            cosine_similarities = F.cosine_similarity(
                image_features_norm.unsqueeze(1),
                zeroshot_weights_norm.T.unsqueeze(0),
                dim=2
            )  # [batch_size, num_classes]

            # Apply temperature scaling
            temperature = hparams.get("temperature", 1.50)
            cosine_similarities = cosine_similarities / temperature

            # Convert similarities to probabilities
            probabilities = torch.softmax(cosine_similarities, dim=1)

            # Predict labels
            predicted_labels = probabilities.argmax(dim=1).cpu()

            # Compute accuracy
            baseline_acc = (predicted_labels == target.cpu()).float().mean().item() * 100

#            if method_name != "ours":
#                print(f"{method_name}: {baseline_acc:.2f}\n")
#                results[method_name] = round(baseline_acc, 2)

            
            if method_name != "ours":
                # Compute precision, recall, and F1-score
                precision = precision_score(
                    target.cpu(), predicted_labels, average='macro', zero_division=0
                ) * 100
                recall = recall_score(
                    target.cpu(), predicted_labels, average='macro', zero_division=0
                ) * 100
                f1 = f1_score(
                    target.cpu(), predicted_labels, average='macro', zero_division=0
                ) * 100
            
                # Print metrics
                print(f"{method_name}: {baseline_acc:.4f}%")
                print(f"Precision: {precision:.4f}%")
                print(f"Recall: {recall:.4f}%")
                print(f"F1 Score: {f1:.4f}%\n")
            
                # Optionally, print classification report and confusion matrix
                class_names = hparams.get("class_names", [])
                print("Classification Report:")
                print(classification_report(
                    target.cpu(), predicted_labels, target_names=class_names, zero_division=0
                ))
                print("Confusion Matrix:")
                cm = confusion_matrix(target.cpu(), predicted_labels)
                print_confusion_matrix(cm, class_names)
                print("\n")
            
                # Store the metrics
                results[method_name] = {
                    'Accuracy': round(baseline_acc, 2),
                    'Precision': round(precision, 2),
                    'Recall': round(recall, 2),
                    'F1 Score': round(f1, 2)
                }



            if method_name == "ours":
                acc_list = []
                hybrid_weight = hparams.get("hybrid_weight", .5)
                n_run = hparams.get("n_run", 10)
                print(f"n_run: {n_run}")
            
                for i in range(n_run):
                    current_seed = seed + i
                    set_seed(current_seed)
                    print(f"Current seed: {current_seed}")

                    patch_embeds_list = []
                    for idx, patch_features in enumerate(precomputed_features):
                        num_patches_to_sample = hparams.get("num_patches_to_sample", 10)
                        total_patches = patch_features.size(0)
                        if total_patches > num_patches_to_sample:
                            indices = torch.randperm(total_patches)[:num_patches_to_sample]
                            #print(f"Run {i}, indices: {indices}")
                            
                        else:
                            indices = torch.arange(total_patches)
                        sampled_patches = patch_features[indices]
            
                        if sampled_patches.size(0) > 0:
                            # Extract embeddings
                            embeddings = sampled_patches[:, :-1]  # [num_patches, embed_dim]
                            embeddings = F.normalize(embeddings, dim=-1)
            
                            # Compute similarities with class embeddings
                            similarities = embeddings @ zeroshot_weights  # [num_patches, num_classes]
            
                            # Compute attention weights
                            attention_weights, _ = similarities.max(dim=1)  # [num_patches]
                            attention_weights = attention_weights.exp()
                            attention_weights = attention_weights / attention_weights.sum()
            
                            # Compute weighted sum of embeddings
                            weighted_embeddings = embeddings * attention_weights.unsqueeze(1)
                            patch_embeds = weighted_embeddings.sum(dim=0)
                            patch_embeds = F.normalize(patch_embeds, dim=-1)
                        else:
                            patch_embeds = torch.zeros_like(image_features[0]).to(device)
                        patch_embeds_list.append(patch_embeds)
            
                    # Stack patch embeddings
                    patch_embeds_tensor = torch.stack(patch_embeds_list).to(device)  # [batch_size, embed_dim]
            
                    # Normalize global embeddings
                    global_embeds = F.normalize(image_features, dim=-1)
            
                    # Combine with global embeddings
                    hybrid_embeds = hybrid_weight * patch_embeds_tensor + (1 - hybrid_weight) * global_embeds
                    hybrid_embeds_norm = F.normalize(hybrid_embeds, dim=1)
            
                    # Compute cosine similarities
                    cosine_similarities = F.cosine_similarity(
                        hybrid_embeds_norm.unsqueeze(1),
                        zeroshot_weights_norm.T.unsqueeze(0),
                        dim=2
                    )  # [batch_size, num_classes]

                    # Apply temperature scaling
                    cosine_similarities = cosine_similarities / temperature

                    # Convert similarities to probabilities
                    probabilities = torch.softmax(cosine_similarities, dim=1)

                    # Predict labels
                    predicted_labels = probabilities.argmax(dim=1).cpu()

                    # Compute accuracy
                    acc = (predicted_labels == target.cpu()).float().mean().item() * 100
                    acc_list.append(acc)
                    
                    
                    
                    
                    precision = precision_score(
                        target.cpu(), predicted_labels, average='macro', zero_division=0
                    )
                    recall = recall_score(
                        target.cpu(), predicted_labels, average='macro', zero_division=0
                    )
                    f1 = f1_score(
                        target.cpu(), predicted_labels, average='macro', zero_division=0
                    )
                    precision_list.append(precision * 100)
                    recall_list.append(recall * 100)
                    f1_list.append(f1 * 100)
    
                # Compute mean and std of metrics
                mean_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                mean_precision = np.mean(precision_list)
                std_precision = np.std(precision_list)
                mean_recall = np.mean(recall_list)
                std_recall = np.std(recall_list)
                mean_f1 = np.mean(f1_list)
                std_f1 = np.std(f1_list)
    
                print(f"{method_name} Metrics over {n_run} run(s):")
                print(f"Accuracy: {mean_acc:.2f}+/-{std_acc:.4f}")
                print(f"Precision: {mean_precision:.4f}+/-{std_precision:.4f}")
                print(f"Recall: {mean_recall:.4f}+/-{std_recall:.4f}")
                print(f"F1 Score: {mean_f1:.4f}+/-{std_f1:.4f}")
    
                # For the last run, you can print detailed reports
                if i == n_run - 1:
                    class_names = hparams.get("class_names", [])
                    print(f"Classification Report for {method_name} (last run):")
                    print(classification_report(
                        target.cpu(), predicted_labels, target_names=class_names, zero_division=0
                    ))
    
                    print(f"Confusion Matrix for {method_name} (last run):")
                    cm = confusion_matrix(target.cpu(), predicted_labels)
                    print(cm)
    
                #results[method_name] = round(mean_acc, 2)  
                            # Store all metrics in results (MODIFIED HERE)
                results[method_name] = {
                    'Accuracy': round(mean_acc, 2),
                    'Precision': round(mean_precision, 2),
                    'Recall': round(mean_recall, 2),
                    'F1 Score': round(mean_f1, 2)
                }                  

                # Compute mean and std of accuracies
                mean_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                print(f"{method_name}: {mean_acc:.4f}")
                print(acc_list)
#                results[method_name] = round(mean_acc, 2)

    # Print final results
    print("\nFinal Results:")
    for method_name, acc in results.items():
        print(f"{method_name}: {acc}%")

if __name__ == "__main__":
    fire.Fire(main)