import os
import json
import argparse
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import monai
from monai.transforms import Compose, ToTensord
from monai.networks.nets import resnet as monai_resnet


# Dataset for processed .npz unilateral files
class OdeliaNPZDataset(Dataset):
    """
    Items: list of dicts with keys:
      - 'npz': path to npz file containing key 'arr' shaped (C, Z, Y, X)
      - 'label': optional int in {0,1,2} (0=normal,1=benign,2=malignant)
      - 'study_id': optional string identifier
    """

    def __init__(self, items, transforms=None, has_labels=True):
        self.items = items
        self.transforms = transforms
        self.has_labels = has_labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = np.load(item["npz"], allow_pickle=True)
        arr = data["arr"]  # (C, Z, Y, X)

        label_val = np.int64(item.get("label", -1)) if self.has_labels else np.int64(-1)
        sample = {"image": arr.astype(np.float32)}
        if self.has_labels:
            sample["label"] = label_val

        if self.transforms:
            sample = self.transforms(sample)

        # Return a consistent tuple
        if self.has_labels:
            return sample["image"], sample["label"]
        else:
            return sample["image"], label_val


# Model wrapper (aligned with training)
class ResNet3DClassifier(torch.nn.Module):
    def __init__(self, in_channels, n_classes=3, bigger=True):
        super().__init__()
        if bigger:
            self.backbone = monai_resnet.resnet152(
                spatial_dims=3, n_input_channels=in_channels, num_classes=n_classes
            )
        else:
            self.backbone = monai_resnet.resnet18(
                spatial_dims=3, n_input_channels=in_channels, num_classes=n_classes
            )

    def forward(self, x):
        return self.backbone(x)


def load_items_from_manifest(manifest_path: str):
    with open(manifest_path, "r") as f:
        items = json.load(f)
    return items


def build_val_transforms(has_labels: bool):
    if has_labels:
        return Compose([ToTensord(keys=["image", "label"])])
    else:
        return Compose([ToTensord(keys=["image"])])


def compute_auc(labels, probs):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")

    y_true = np.array(labels)
    if y_true.ndim == 0 or len(y_true) == 0:
        return float("nan")
    # malignant vs non-malignant
    y_true_bin = (y_true == 2).astype(int)
    if np.unique(y_true_bin).size < 2:
        return float("nan")
    scores = np.array(probs)[:, 2]
    return float(roc_auc_score(y_true_bin, scores))


def run_inference(
    checkpoint: str,
    manifest: str,
    output_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load items and detect label availability
    items = load_items_from_manifest(manifest)
    if not items:
        raise ValueError("Manifest contains no items.")
    has_labels = "label" in items[0]

    # Determine in_channels from first item
    sample_arr = np.load(items[0]["npz"]) ["arr"]
    in_channels = int(sample_arr.shape[0])

    # Load checkpoint to know architecture size (bigger or not)
    ckpt = torch.load(checkpoint, map_location=device)
    bigger_model = bool(ckpt.get("bigger_model", True))

    # Build model and load weights
    model = ResNet3DClassifier(in_channels=in_channels, n_classes=3, bigger=bigger_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dataloader
    val_trans = build_val_transforms(has_labels=has_labels)
    dataset = OdeliaNPZDataset(items, transforms=val_trans, has_labels=has_labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_probs = []
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader, desc="inference", leave=False)):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    # Flatten batches
    if all_probs:
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_probs = np.empty((0, 3), dtype=np.float32)
        all_preds = np.empty((0,), dtype=np.int64)
        all_labels = np.empty((0,), dtype=np.int64)

    # Build ids from filenames (or study_id if available)
    for it in items:
        sid = it.get("study_id") or os.path.splitext(os.path.basename(it["npz"]))[0]
        all_ids.append(sid)
    all_ids = np.array(all_ids)

    # Metrics (if labels present)
    metrics = {}
    if has_labels and all_labels.size > 0:
        acc = float((all_preds == all_labels).mean())
        auc = compute_auc(all_labels, all_probs)
        metrics = {"accuracy": acc, "auc_malignant_vs_rest": auc}

    # Save outputs
    pred_file = os.path.join(output_dir, "predictions.npz")
    np.savez_compressed(
        pred_file,
        ids=all_ids,
        probs=all_probs.astype(np.float32),
        preds=all_preds.astype(np.int64),
        labels=all_labels.astype(np.int64),
    )

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved predictions to: {pred_file}")
    if metrics:
        print(f"Metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Inference for 3D ResNet classifier")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt (e.g., outputs/all_resnet152/best_model.pt)")
    parser.add_argument("--manifest", required=True, help="JSON list of items with 'npz' and optional 'label'")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default=None, help="Where to save predictions/metrics; defaults to outputs/inference/<timestamp>")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(
        "./outputs/inference", datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    run_inference(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
