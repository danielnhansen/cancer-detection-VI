from datasets import load_dataset
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="ODELIA-AI/ODELIA-Challenge-2025",
    repo_type="dataset",
    revision="refs/convert/parquet",
    allow_patterns=["unilateral/**"],
    local_dir="./odelia_unilateral",  
    local_dir_use_symlinks=False, 
)

dataset = load_dataset(
    "parquet",
    data_files={
        "train": "/cluster/projects/vc/data/mic/open/Prostate/odelia2025/odelia_unilateral/unilateral/partial-train/*.parquet",
        "validation": "/cluster/projects/vc/data/mic/open/Prostate/odelia2025/odelia_unilateral/unilateral/partial-val/*.parquet",
        "test": "/cluster/projects/vc/data/mic/open/Prostate/odelia2025/odelia_unilateral/unilateral/partial-test/*.parquet",
    }
)


from tqdm import tqdm
from pathlib import Path
import torchio as tio 
import numpy as np 
import pandas as pd
from tqdm import tqdm

config = "unilateral"  # "default" or "unilateral"
output_root = Path("./data")

dir_config = {
    "default": {
        "data": "data",
        "metadata": "metadata",
    },
    "unilateral": {
        "data": "data_unilateral",
        "metadata": "metadata_unilateral",
    },
}

# Process dataset
metadata = []
for split, split_dataset in dataset.items():
    print("-------- Start Processing - Split: ", split, " --------")
    for item in tqdm(split_dataset, desc="Processing"):  # Processing data one-by-one
        uid = item["UID"]
        institution = item["Institution"]
        img_names = [name.split("Image_")[1] for name in item.keys() if name.startswith("Image")]
        
        # Create output folder
        path_folder = output_root / institution / dir_config[config]["data"] / uid
        path_folder.mkdir(parents=True, exist_ok=True)

        for img_name in img_names:
            img_data = item.pop(f"Image_{img_name}")
            img_affine = item.pop(f"Affine_{img_name}")

            # Skip if image data is None
            if img_data is None:
                continue

            # Extract image data and affine matrix
            img_data = np.array(img_data, dtype=np.int16) 
            img_affine = np.array(img_affine, dtype=np.float64)
            img = tio.ScalarImage(tensor=img_data, affine=img_affine)
            
            # Save image
            img.save(path_folder / f"{img_name}.nii.gz")

        # Store metadata
        metadata.append(item)  


# Convert metadata to DataFrame
df = pd.DataFrame(metadata)

for institution in df["Institution"].unique():
    # Load metadata
    df_inst = df[df["Institution"] == institution]

    # Save metadata to CSV files
    path_metadata = output_root / institution / dir_config[config]["metadata"]
    path_metadata.mkdir(parents=True, exist_ok=True)

    df_anno = df_inst.drop(columns=["Institution", "Split", "Fold", "__index_level_0__"])
    df_anno.to_csv(path_metadata / "annotation.csv", index=False)

    df_split = df_inst[["UID", "Split", "Fold"]]
    df_split.reset_index(drop=True)
    df_split.to_csv(path_metadata / "split.csv", index=False) 


print("Dataset downloaded and saved successfully!")


# Downloading the evaluation method
local_dir = snapshot_download(
    repo_id="ODELIA-AI/ODELIA-Challenge-2025",
    repo_type="dataset",
    revision="main",            # or another branch/tag if needed
    allow_patterns=["evaluation-method/**"],   # only files in that folder
    local_dir="./evaluation_method",

)

