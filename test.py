import os
import time
import torch
import numpy as np
from skimage import io
from os.path import isdir
import sys
from torchvision import transforms

from models.VVBPConvNet import VVBPConvNet_PatchAggregator


def ensure_grayscale(image):
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            image = np.dot(image, [0.2989, 0.5870, 0.1140])
        else:
            image = image[:, :, 0]
    return image


def normalize_image(image, method='percentile'):
    image_min = image.min()
    image_max = image.max()

    if method == 'percentile':
        p_low = np.percentile(image, 1)
        p_high = np.percentile(image, 99)
        output_clipped = np.clip(image, p_low, p_high)
        normalized = (output_clipped - p_low) / (p_high - p_low + 1e-10)
    elif method == 'adaptive':
        from skimage import exposure
        normalized = exposure.equalize_adapthist(
            image.squeeze(),
            clip_limit=0.03
        )
        normalized = normalized[np.newaxis, np.newaxis, ...]
    else:
        if image_max - image_min > 1e-8:
            normalized = (image - image_min) / (image_max - image_min)
        else:
            normalized = np.zeros_like(image)
    return normalized

def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU memory usage: {allocated:.2f} GB / {reserved:.2f} GB")



def test_VVBPConvNet(model, input_dir_test, save_dir_test,
                     weight, views=60):
    try:
        checkpoint = torch.load(weight,
                                map_location=device,
                                weights_only=True)
    except Exception as e:
        checkpoint = torch.load(weight, map_location=device)
    model = model(
        views=views,
        ct_resolution=(181, 181),
        window_size=64).to(device)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    if isdir(input_dir_test):
        files = os.listdir(input_dir_test)
        for file in files:
            if file.endswith(('.png', '.tif', '.tiff')):
                imname, ext = os.path.splitext(file)
                fname = os.path.join(input_dir_test, file)

                input_image = np.array(io.imread(fname))
                input_image = ensure_grayscale(input_image)
                input_image = normalize_image(input_image, method=None)

                transf = transforms.ToTensor()
                re_sino = transf(input_image).to(device)
                re_sino = torch.unsqueeze(re_sino, 0)

                slice = model(re_sino)
                slice = torch.squeeze(slice)
                slice = slice.detach().cpu().numpy()
                slice = normalize_image(slice, method=None)
                slice = slice.clip(0, 1)

                os.makedirs(save_dir_test, exist_ok=True)
                save_path = os.path.join(save_dir_test, imname+'.tif')
                io.imsave(save_path, slice.squeeze(), check_contrast=False)


import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script configuration parameters")

    parser.add_argument('--input_dir_test', type=str, default=r'data/test/sinogram60views_VVBP',
                        help="Directory for test input data")
    parser.add_argument('--save_dir_test', type=str, default='result/VVBPConvNet60views',
                        help="Directory to save test results")
    parser.add_argument('--weight', type=str, default='weight/best_model.pth',
                        help="Path to the pre-trained weight file")
    parser.add_argument('--views', type=int, default=60,
                        help="Number of views")

    return parser.parse_args()


if __name__ == "__main__":
    clear_gpu_memory()
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VVBPConvNet_PatchAggregator().to(device)

    test_VVBPConvNet(model, args.input_dir_test, args.save_dir_test, args.weight, args.views)