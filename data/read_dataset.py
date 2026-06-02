import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
from skimage import io
from torchvision import transforms

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    if num_workers > 0 and torch.multiprocessing.get_start_method() != 'spawn':
        try:
            torch.multiprocessing.set_start_method('spawn')
        except RuntimeError:
            num_workers = 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1 if num_workers > 0 else None,
        # drop_last=True
    )


import imageio.v2 as iio

class dirsUnifiedCTDataset(Dataset):
    def __init__(self, input_dirs, target_dir, mode, transform=None,
                 get_pseudogt=False, sino_dirs=None):
        self.input_dirs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
        self.target_dir = target_dir
        self.transform = transform
        self.get_pseudo = get_pseudogt
        self.mode = mode
        self.data_pairs = []
        self.sino_dirs = sino_dirs

        if not os.path.exists(self.target_dir):
            raise ValueError(f"Target directory {self.target_dir} does not exist!")
        self.target_files_set = set(
            f for f in os.listdir(self.target_dir)
            if f.lower().endswith(('.png', '.tif', '.tiff'))
        )

        def get_gt_filename(input_fname):
            if "_nsr" in input_fname:
                base_name = input_fname.split("_nsr")[0]
                return base_name + ".tif"
            else:
                return input_fname

        total_inputs = 0
        for input_dir in self.input_dirs:
            if not os.path.exists(input_dir):
                print(f"Warning: Input directory {input_dir} does not exist, skipping!")
                continue

            files = sorted([
                f for f in os.listdir(input_dir)
                if f.lower().endswith(('.png', '.tif', '.tiff'))
            ])

            for f in files:
                gt_name = get_gt_filename(f)

                if gt_name in self.target_files_set:
                    self.data_pairs.append({
                        "input": os.path.join(input_dir, f),
                        "target": os.path.join(self.target_dir, gt_name)
                    })
                else:

                    found = False
                    base_gt = os.path.splitext(gt_name)[0]
                    for ext in ['.tif', '.tiff', '.png']:
                        if (base_gt + ext) in self.target_files_set:
                            self.data_pairs.append({
                                "input": os.path.join(input_dir, f),
                                "target": os.path.join(self.target_dir, base_gt + ext)
                            })
                            found = True
                            break
                    if not found and total_inputs < 5:
                        print(f"No corresponding GT found: {f} -> Expected GT: {gt_name}")

            total_inputs += len(files)

        print(f"Mode [{mode}]:")
        print(f"  Input source folders: {len(self.input_dirs)}")
        print(f"  Input files scanned: {total_inputs}")
        print(f"  Successfully paired data: {len(self.data_pairs)} pairs")

        if len(self.data_pairs) == 0:
            raise ValueError("No matching image pairs found. Please check the filename matching logic!")

    def _ensure_grayscale(self, image):
        if image.ndim == 3:
            if image.shape[2] == 3:  # RGB
                image = np.dot(image, [0.2989, 0.5870, 0.1140])
            elif image.shape[2] == 1:
                image = image[:, :, 0]

        return image

    def _normalize_image(self, image):
        image = image.astype(np.float32)
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 1e-8:
            image = (image - min_val) / (max_val - min_val)
        else:
            image = np.zeros_like(image)
        return image

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]
        input_path = pair["input"]
        target_path = pair["target"]

        try:
            input_image = iio.imread(input_path)
            target_image = iio.imread(target_path)

            input_image = np.array(input_image)
            target_image = np.array(target_image)

            input_image = self._ensure_grayscale(input_image)
            target_image = self._ensure_grayscale(target_image)

            input_image = self._normalize_image(input_image)
            target_image = self._normalize_image(target_image)

            input_image = torch.from_numpy(input_image).float()
            target_image = torch.from_numpy(target_image).float()

            if input_image.dim() == 2:
                input_image = input_image.unsqueeze(0)
            if target_image.dim() == 2:
                target_image = target_image.unsqueeze(0)

            return input_image, target_image

        except Exception as e:
            print(f"Error loading: {input_path}")
            raise e

