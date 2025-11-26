"""
For ODELIA unilateral folders

Example usage:
  uv run preprocess.py --input ./CAM/data_unilateral --output ./data/processed_CAM
"""

import os, argparse, json
from glob import glob
import numpy as np
import SimpleITK as sitk

def read_sitk(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
    return arr, img

def resample_to_spacing(img, spacing=(0.7,0.7,3.0)):
    orig_sp = img.GetSpacing()
    orig_sz = img.GetSize()
    new_sz = [int(round(orig_sz[i] * orig_sp[i]/spacing[i])) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_sz)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(img)

def normalize_zscore(vol):
    m, s = vol.mean(), vol.std()
    if s == 0: s = 1
    return (vol - m) / s

def build_channels(folder):
    paths = {
        'pre': os.path.join(folder, 'Pre.nii.gz'),
        'post1': os.path.join(folder, 'Post_1.nii.gz'),
        'post2': os.path.join(folder, 'Post_2.nii.gz'),
        'sub1': os.path.join(folder, 'Sub_1.nii.gz'),
        't2': os.path.join(folder, 'T2.nii.gz'),
    }
    imgs = {}
    for k,p in paths.items():
        if os.path.exists(p):
            arr,img = read_sitk(p)
            imgs[k] = (arr,img)
    if 'pre' not in imgs:
        raise FileNotFoundError(f"No Pre.nii.gz in {folder}")

    # resample all to pre spacing for consistency (and working with model)
    target_spacing = (0.7,0.7,3.0)
    resampled = {}
    for k,(arr,img) in imgs.items():
        img_r = resample_to_spacing(img, spacing=target_spacing)
        resampled[k] = sitk.GetArrayFromImage(img_r)

    # build enhancement maps
    pre = resampled['pre']
    post1 = resampled.get('post1', pre)
    diff = post1 - pre
    rel = diff / (pre + 1e-6)
    channels = [pre, post1, diff, rel]
    
    if 't2' in resampled:
        # Append T2 as additional channel if available
        channels.append(resampled['t2'])
    out = np.stack(channels, axis=0)  # (C,Z,Y,X)
    out = normalize_zscore(out)
    return out, {'channels': list(resampled.keys()), 'spacing': target_spacing}

def process_all(input_root, output_root):
    folders = sorted([f for f in glob(os.path.join(input_root, '*')) if os.path.isdir(f)])
    os.makedirs(output_root, exist_ok=True)
    for fold in folders:
        uid = os.path.basename(fold)
        try:
            arr,meta = build_channels(fold)
            out_path = os.path.join(output_root, uid + '.npz')
            np.savez_compressed(out_path, arr=arr.astype(np.float32), meta=json.dumps(meta))
            print(f"[OK] {uid}")
        except Exception as e:
            print(f"[FAIL] {uid}: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    process_all(args.input, args.output)
