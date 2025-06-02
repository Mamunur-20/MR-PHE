# generate_prompts_wsss_short.py

import json
import os

# Define your class names and their synonyms along with clinical significance
classnames_with_clinical_significance = {
    "Normal": {
        "synonyms": [
            "normal",
            "uninvolved tissue",
            "healthy tissue",
            "benign tissue",
            "non-malignant tissue",
            "standard tissue"
        ],
        "clinical_significance": [
            "Normal tissue maintains organ function.",
            "Healthy cells provide a baseline for comparison.",
            "Normal epithelium lacks cancerous changes.",
            "Uninvolved tissue indicates absence of disease.",
            "Histopathology normal tissue is non-cancerous.",
            "Normal cellular architecture ensures proper tissue function."
        ]
    },
    "Stroma": {
        "synonyms": [
            "stroma",
            "stromal tissue",
            "supportive tissue",
            "connective tissue",
            "fibrous tissue",
            "mesenchymal tissue"
        ],
        "clinical_significance": [
            "Stromal cells support tumor growth.",
            "Stroma provides structural framework for tissues.",
            "Cancer-associated stroma influences tumor progression.",
            "Stromal environment affects cancer cell behavior.",
            "Stroma contributes to the tumor microenvironment.",
            "Stroma plays a role in angiogenesis within tumors."
        ]
    },
    "Tumor": {
        "synonyms": [
            "tumor",
            "malignant cells",
            "cancerous tissue",
            "adenocarcinoma",
            "malignant epithelium",
            "carcinoma"
        ],
        "clinical_significance": [
            "Tumor cells exhibit uncontrolled proliferation.",
            "Malignant tumors can invade surrounding tissues.",
            "Tumors can metastasize to distant organs.",
            "Tumor size and grade indicate prognosis.",
            "Adenocarcinoma is a common type of tumor.",
            "Tumor heterogeneity can impact treatment response."
        ]
    }
}

# Define the templates
templates = [
    "CLASSNAME.",
    "A photomicrograph showing CLASSNAME.",
    "A photomicrograph of CLASSNAME.",
    "An image of CLASSNAME.",
    "An image showing CLASSNAME.",
    "An example of CLASSNAME.",
    "CLASSNAME is shown.",
    "This is CLASSNAME.",
    "There is CLASSNAME.",
    "A histopathological image showing CLASSNAME.",
    "A histopathological image of CLASSNAME.",
    "A histopathological photograph of CLASSNAME.",
    "A histopathological photograph showing CLASSNAME.",
    "Shows CLASSNAME.",
    "Presence of CLASSNAME.",
    "CLASSNAME is present.",
    "An H&E stained image of CLASSNAME.",
    "An H&E stained image showing CLASSNAME.",
    "An H&E image showing CLASSNAME.",
    "An H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E."
]

def generate_prompts(classnames_with_clinical_significance, templates):
    """
    Generate prompts for each class by combining synonyms, templates, and clinical significance statements.

    Args:
        classnames_with_clinical_significance (dict): Dictionary containing class names, their synonyms, and clinical significance.
        templates (list): List of template strings with 'CLASSNAME' as a placeholder.

    Returns:
        dict: Dictionary with class names as keys and lists of generated prompts as values.
    """
    prompts = {}
    for class_name, data in classnames_with_clinical_significance.items():
        prompts[class_name] = []
        
        # Generate prompts using synonyms and templates
        for synonym in data["synonyms"]:
            for template in templates:
                prompt = template.replace("CLASSNAME", synonym)
                prompts[class_name].append(prompt)
        
        # Add clinical significance statements as additional prompts
        for significance in data["clinical_significance"]:
            prompts[class_name].append(significance)
    
    return prompts

def main(output_folder="prompts/wsss/", filename="ours_filtered.json"):
    """
    Main function to generate prompts and save them to a JSON file.

    Args:
        output_folder (str): Path to the output directory where the JSON file will be saved.
        filename (str): Name of the JSON file to save the prompts.
    """
    # Generate the prompts
    prompts = generate_prompts(classnames_with_clinical_significance, templates)
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the prompts to a JSON file
    json_path = os.path.join(output_folder, filename)
    with open(json_path, "w") as json_file:
        json.dump(prompts, json_file, indent=4)
    
    print(f"Prompt generation completed. Prompts saved to '{json_path}'.")

if __name__ == "__main__":
    main()
