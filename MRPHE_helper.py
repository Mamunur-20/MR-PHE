import os
import random
import numpy as np
import torch
import json
import pickle
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from conch.open_clip_custom.custom_tokenizer import get_tokenizer, tokenize
from conch.open_clip_custom.factory import create_model_from_pretrained
from torchvision.datasets import ImageNet, ImageFolder, Places365
#from my_datasets.ebhi_loader import EBHIDataset
#from my_datasets.mhist_loader import MHISTDataset
#from my_datasets.breakhis_loader import BREAKHISDataset
#from my_datasets.private_loader import PRIVATEDataset
from my_datasets.heghids_loader import HEGHIDSDataset
from my_datasets.wsss_loader import WSSSDataset
from my_datasets.bracs7class_loader import BRACSDataset
from my_datasets.bracs_loader import BRACSDataset
from my_datasets.crc_loader import CRCDataset
from my_datasets import *


def load_json(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dataset_name, custom_loader):
    data_path = data_path
      
    if dataset_name == "wsss":  # Add your custom dataset
       dataset = WSSSDataset(
           root=data_path,
           transform=None,
           loader=custom_loader,
           split="train",  # or "test" depending on your use case
       )
#       
#       
    if dataset_name == "heghids":  # Add your custom dataset
       dataset = HEGHIDSDataset(
           root=data_path,
           transform=None,
           loader=custom_loader,
           split="train",  # or "test" depending on your use case
       )
#       
#       
#    if dataset_name == "private":  # Add your custom dataset
#       dataset = PRIVATEDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )

       
       

#    if dataset_name == "breakhis":  # Add your custom dataset
#       dataset = BREAKHISDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )   
       
#    if dataset_name == "ebhi":  # Add your custom dataset
#       dataset = EBHIDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )
       
#    if dataset_name == "bracs":  # Add your custom dataset
#       dataset = BRACSDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )
       
       
       
#    if dataset_name == "bracs7class":  # Add your custom dataset
#       dataset = BRACSDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )
              
# 
#    if dataset_name == "crc":  # Add your custom dataset
#       dataset = CRCDataset(
#           root=data_path,
#           transform=None,
#           loader=custom_loader,
#           split="train",  # or "test" depending on your use case
#       )
 
        
               
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset


#def wordify(string):
#    word = string.replace("_", " ")
#    return word

def wordify(string):
    if isinstance(string, list):
        return [s.replace("_", " ") for s in string]
    return string.replace("_", " ")



def load_classes(dataset_name):
    with open(
        f"features/{dataset_name}/{dataset_name}.json",
        "r",
    ) as f:
        classes = json.load(f)

    wordify_classes = []
    for c in classes:
        wordify_classes.append(wordify(c))

    return wordify_classes


def generate_weights(
    method,
    model,
    dataset_name,
    tt_scale=None,
    device=None,
):
    templates = None
    make_sentence = False
    is_template = True

    # if dataset start with imagenet
#    if dataset_name.startswith(MyDataset.ImageNet):
#        classes = (
#            openai_imagenet_classes
#            if method in ["clip-d", "waffle"]
#            else imagenet_classes
#        )
#    else:
    classes = load_classes(dataset_name)

    print(f"Creating {method} text embeddings...")

    if method != "clip":
        if method == "ours":
            load_file = "ours"
        elif method == "cupl":
            load_file = "cupl"
        elif method == "waffle":
            load_file = "clip-d"
        else:
            load_file = method

        with open(f"prompts/{dataset_name}/{load_file}.json") as f:
            templates = json.load(f)

        if method in ["waffle", "clip-d", "cupl", "ours"]:
            is_template = False

        if method == "clip-d":
            make_sentence = True

        if method == "waffle":
            templates = construct_random(templates)

    zeroshot_weights = zeroshot_classifier(
        model,
        classes,
        templates,
        is_template,
        make_sentence,
        tt_scale,
        device,
    )

    return zeroshot_weights


def load_precomputed_features(
    model,
    dataset_name: str,
    model_size: str,
    alpha: float,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
):
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"features/{dataset_name}"

    # if save_root not exist, create it
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root, f"{save_file}-{alpha}-{n_samples}.pkl")

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        load_res = pickle.load(open(filename, "rb"))
    else:
        print(f"File {filename} not found, precomputing features...")
        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


        precomputed_features = []
        image_features_tensor = []
        target = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, labels = [p.to(device) for p in batch]
    
                b, ns = images.shape[:2]
                images = images.flatten(0, 1)
    
                image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                image_features = image_features.view(b, ns, -1)  # b, ns, d
    
                # Correct separation of image and patch features
                patch_features = image_features[:, :-1]  # Patches
                image_features = image_features[:, -1:]  # Original images
    
                # Compute weights
                weight_image = (image_features * patch_features).sum(dim=-1, keepdim=True)
    
                # Concatenate patch features with weights
                patch_with_weights = torch.cat([patch_features, weight_image], -1)
    
                precomputed_features.append(patch_with_weights)
                target.append(labels)
                image_features_tensor.append(image_features.squeeze(1))



#        with torch.no_grad():
#            for batch in tqdm(dataloader):
#                images, labels = [p.to(device) for p in batch]
#
#                b, ns = images.shape[:2]
#                images = images.flatten(0, 1)
#
#                image_features = model.encode_image(images)
#                image_features = F.normalize(image_features)
#                image_features = image_features.view(b, ns, -1)  # b,ns,d
#
#                patch_features = image_features[:, 1:]
#                image_features = image_features[:, :1]
#
#                weight_image = (image_features * patch_features).sum(
#                    dim=-1, keepdim=True
#                )
#
#                patch_with_weights = torch.cat([patch_features, weight_image], -1)
#
#                precomputed_features.append(patch_with_weights)
#                target.append(labels)
#                image_features_tensor.append(image_features.squeeze(1))

        load_res = {
            "patches": torch.cat(precomputed_features, dim=0),
            "images": torch.cat(image_features_tensor, dim=0),
            "labels": torch.cat(target, dim=0),
        }

        os.makedirs(save_root, exist_ok=True)
        pickle.dump(load_res, open(filename, "wb"))

    precomputed_features = load_res["patches"].to(device)
    target = load_res["labels"].to(device)
    image_features_tensor = load_res["images"].to(device)

    return precomputed_features, target, image_features_tensor


def make_descriptor_sentence(descriptor):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"




def zeroshot_classifier(
    model,
    textnames,
    templates=None,
    is_template=True,
    make_sentence=False,
    tt_scale=None,
    device=None,
):
    with torch.no_grad():
        zeroshot_weights = []
        # Get the tokenizer object from get_tokenizer
        tokenizer = get_tokenizer()
        
        for i in tqdm(range(len(textnames))):
            textname = textnames[i][0]  # Extract the string from the list
            if not is_template:
                texts = []
                for t in templates[textname]:  # Use the extracted string
                    if make_sentence:
                        desc_sen = make_descriptor_sentence(t)
                        texts.append(f"{textname}, {desc_sen}")
                    else:
                        texts.append(t)
            elif templates:
                texts = [template.format(textname) for template in templates]  # Use the extracted string
            else:
                texts = [f"a photo of a {textname}."]  # Use the extracted string
            
            # Log the generated texts
            print(f"Generated texts for {textname}: {texts}")

            if tt_scale is not None:
                label = f"a photo of a {textname}."  # Use the extracted string
                label_tokens = tokenize(tokenizer, [label]).to(device)  # Pass tokenizer and text as a list
                label_embeddings = model.encode_text(label_tokens)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

            texts_tensor = tokenize(tokenizer, texts).to(device)  # Pass tokenizer and texts
            class_embeddings = model.encode_text(texts_tensor)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if tt_scale is not None:
                weight = class_embeddings @ label_embeddings.T
                weight = (weight * tt_scale).softmax(dim=0)
                class_embedding = (class_embeddings * weight).sum(dim=0)
                class_embedding /= class_embedding.norm()
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def construct_random(gpt3_prompts):
    """
    Custom Waffle generator for histopathology.
    """
    key_list = list(gpt3_prompts.keys())

    # Histopathology-related words
    histopath_words = ["cell", "tissue", "nucleus", "cytoplasm", "mitosis", "stain", "lesion", "atypia", "necrosis", "pleomorphism"]

    def structured_descriptor_builder(cls, word):
        return f"A photo of a {cls}, which has {word}."

    gpt3_prompts = {key: [] for key in gpt3_prompts.keys()}

    for key in key_list:
        for _ in range(30):  # Generate 30 random descriptors per class
            base_word = np.random.choice(histopath_words)
            gpt3_prompts[key].append(structured_descriptor_builder(key, base_word))

    return gpt3_prompts





def accuracy(output, target, n, dataset_name):
    # Get index of the maximum value as prediction
#    if dataset_name.startswith(MyDataset.ImageNetA):
#        _, pred = output[:, imagenet_a_lt].max(1)
#    elif dataset_name.startswith(MyDataset.ImageNetR):
#        _, pred = output[:, imagenet_r_lt].max(1)
#    else:
    _, pred = output.max(1)
    # Compare prediction with target
    correct = pred.eq(target)
    # Calculate top-1 accuracy
    return float(correct.float().sum().cpu().numpy()) / n * 100