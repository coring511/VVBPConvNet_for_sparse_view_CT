import os
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm
import pandas as pd


def estimate_poisson_parameters(image, target_nsr_percent):
    """
    Estimate Poisson noise parameter I0 based on target NSR.

    Parameters:
        image: input image (raw grayscale values)
        target_nsr_percent: target NSR (percentage, e.g., 5 for 5%)

    Returns:
        I0: estimated Poisson noise intensity
        sigma: recommended Gaussian noise standard deviation (set to 0 or very small)
    """
    # Normalize image
    img = image.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Compute mean intensity of the image
    mean_intensity = np.mean(img_norm)

    # Target NSR (convert to decimal)
    target_nsr = target_nsr_percent / 100.0

    # Derive I0 from Poisson noise characteristics
    # NSR ≈ sqrt(mean_intensity) / sqrt(I0)
    # I0 ≈ mean_intensity / NSR²

    if target_nsr < 0.001:  # NSR close to 0 (noise-free)
        I0 = 1e10           # very large → almost no noise
        sigma = 0
    else:
        I0 = mean_intensity / (target_nsr ** 2)
        # Set Gaussian noise very small (Poisson dominant)
        sigma = target_nsr * 0.1   # about 10% of target NSR

    return I0, sigma


def estimate_gaussian_parameters(image, target_nsr_percent):
    """
    Estimate Gaussian noise standard deviation based on target NSR.

    Parameters:
        image: input image (raw grayscale values)
        target_nsr_percent: target NSR (percentage, e.g., 5 for 5%)

    Returns:
        sigma: recommended Gaussian noise standard deviation
    """
    img = image.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    sigma_signal = np.std(img_norm)

    target_nsr = target_nsr_percent / 100.0

    sigma_noise = target_nsr * sigma_signal

    return sigma_noise


def generate_nsr_range(nsr_min=0, nsr_max=10, n_steps=11, distribution='linear'):
    """
    Generate a sequence of NSR values from nsr_min to nsr_max.

    Parameters:
        nsr_min: minimum NSR (%)
        nsr_max: maximum NSR (%)
        n_steps: number of steps
        distribution: 'linear', 'log', 'quadratic'

    Returns:
        nsr_values: list of NSR values
    """
    if distribution == 'linear':
        # Linear uniform distribution
        nsr_values = np.linspace(nsr_min, nsr_max, n_steps)

    elif distribution == 'log':
        # Logarithmic distribution (denser at low NSR region)
        if nsr_min == 0:
            nsr_min = 0.1      # avoid log(0)
        nsr_values = np.logspace(np.log10(nsr_min), np.log10(nsr_max), n_steps)

    elif distribution == 'quadratic':
        # Quadratic distribution
        t = np.linspace(0, 1, n_steps)
        nsr_values = nsr_min + (nsr_max - nsr_min) * (t ** 2)

    else:
        raise ValueError(f"Unknown distribution type: {distribution}")

    return nsr_values


def add_gaussian_noise_with_nsr(image, target_nsr_percent, sigma=None):
    """
    Add pure Gaussian noise with specified NSR.

    Parameters:
        image: input image
        target_nsr_percent: target NSR (%)
        sigma: if provided, use directly; otherwise estimate automatically

    Returns:
        noisy: noisy image
        actual_sigma: actually used sigma
    """
    img = image.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if sigma is None:
        sigma = estimate_gaussian_parameters(image, target_nsr_percent)

    noise = np.random.randn(*img.shape) * sigma
    noisy = img_norm + noise
    noisy = np.clip(noisy, 0, 1)

    return noisy.astype(np.float32), sigma


def add_quantitative_noise_with_nsr(image, target_nsr_percent, anisotropic=False,
                                    I0=None, sigma=None):
    """
    Add noise with a specified NSR.

    Parameters:
        image: input image
        target_nsr_percent: target NSR (%)
        anisotropic: whether to add anisotropic noise
        I0, sigma: if provided, use directly; otherwise estimate automatically

    Returns:
        noisy: noisy image
        actual_I0: actually used I0
        actual_sigma: actually used sigma
    """
    img = image.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Estimate parameters if not provided
    if I0 is None or sigma is None:
        I0, sigma = estimate_poisson_parameters(image, target_nsr_percent)

    # 1. Poisson noise
    if target_nsr_percent < 0.01:   # almost noise-free
        poisson_noisy = img_norm
    else:
        poisson_noisy = np.random.poisson(img_norm * I0) / I0

    # 2. Gaussian noise (optional)
    if sigma > 1e-6:
        if anisotropic:
            H = img.shape[0]
            noise = np.random.randn(*img.shape) * sigma
            noise[H // 2:, :] *= 2.0
        else:
            noise = np.random.randn(*img.shape) * sigma
        noisy = poisson_noisy + noise
    else:
        noisy = poisson_noisy

    noisy = np.clip(noisy, 0, 1)
    return noisy.astype(np.float32), I0, sigma


def compute_noise_metrics_fast(clean, noisy):
    """Fast computation of noise metrics (simplified version)."""
    clean = clean.astype(np.float32)
    noisy = noisy.astype(np.float32)

    # Normalize both to the same range
    c_min, c_max = clean.min(), clean.max()
    clean_norm = (clean - c_min) / (c_max - c_min + 1e-10)
    noisy_norm = (noisy - c_min) / (c_max - c_min + 1e-10)
    noisy_norm = np.clip(noisy_norm, 0, 1)

    noise = noisy_norm - clean_norm

    # NSR
    nsr_percent = (np.std(noise) / (np.std(clean_norm) + 1e-10)) * 100
    nsr_db = 20 * np.log10(np.std(noise) / (np.std(clean_norm) + 1e-10))

    # PSNR
    mse = np.mean((clean_norm - noisy_norm) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))

    return {
        'NSR_percent': nsr_percent,
        'NSR_dB': nsr_db,
        'PSNR': psnr
    }


def process_folder_gaussian_nsr_list(input_folder, output_folder, nsr_list,
                                     save_metrics=True, verify_nsr=True):
    """
    Add pure Gaussian noise with specific NSR values from a list.

    Parameters:
        input_folder: input folder (noise-free images)
        output_folder: output folder
        nsr_list: list of target NSR percentages, e.g. [1, 3, 6]
        save_metrics: whether to save summary CSV
        verify_nsr: whether to verify actual NSR
    """
    os.makedirs(output_folder, exist_ok=True)

    file_list = sorted([f for f in os.listdir(input_folder)
                        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])
    if not file_list:
        print("No image files found.")
        return []

    first_img = iio.imread(os.path.join(input_folder, file_list[0]))
    if first_img.ndim == 3:
        first_img = np.mean(first_img, axis=2)
    img_norm = (first_img - first_img.min()) / (first_img.max() - first_img.min() + 1e-8)
    sigma_signal = np.std(img_norm)

    all_results = []

    for idx, target_nsr in enumerate(nsr_list, 1):
        sigma_noise = (target_nsr / 100.0) * sigma_signal

        print(f"\n{'=' * 70}")
        print(f"Noise level {idx}: target NSR = {target_nsr}%")
        print(f"Sigma_noise = {sigma_noise:.6f}")
        print(f"{'=' * 70}")

        actual_nsr_list = []
        psnr_list = []

        for fname in tqdm(file_list, desc=f"Processing for NSR={target_nsr}%"):
            img = iio.imread(os.path.join(input_folder, fname))
            if img.ndim == 3:
                img = np.mean(img, axis=2)

            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            noise = np.random.randn(*img.shape) * sigma_noise
            noisy = img_norm + noise
            noisy = np.clip(noisy, 0, 1)

            name, ext = os.path.splitext(fname)
            out_name = f"{name}_gau_nsr{target_nsr}percent{ext}"
            out_path = os.path.join(output_folder, out_name)
            iio.imwrite(out_path, noisy)

            if verify_nsr:
                metrics = compute_noise_metrics_fast(img, noisy)
                actual_nsr_list.append(metrics['NSR_percent'])
                psnr_list.append(metrics['PSNR'])

        if verify_nsr and actual_nsr_list:
            avg_actual_nsr = np.mean(actual_nsr_list)
            std_actual_nsr = np.std(actual_nsr_list)
            avg_psnr = np.mean(psnr_list)
            print(f"   Actual NSR: {avg_actual_nsr:.2f}% ± {std_actual_nsr:.2f}%")
            print(f"   Error: {abs(avg_actual_nsr - target_nsr):.2f}%")
            print(f"   Average PSNR: {avg_psnr:.2f} dB")

            all_results.append({
                'target_NSR_percent': target_nsr,
                'sigma_noise': sigma_noise,
                'actual_NSR_percent': avg_actual_nsr,
                'NSR_error': abs(avg_actual_nsr - target_nsr),
                'PSNR_mean': avg_psnr
            })
        else:
            all_results.append({
                'target_NSR_percent': target_nsr,
                'sigma_noise': sigma_noise
            })

    if save_metrics:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_folder, 'gaussian_nsr_custom_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    return all_results


def process_folder_gaussian_nsr_sweep(input_folder, output_folder,
                                      nsr_min=0, nsr_max=10, n_steps=11,
                                      distribution='linear',
                                      save_metrics=True,
                                      verify_nsr=True):
    """
    Add a series of pure Gaussian noise levels (NSR from nsr_min to nsr_max)
    to all images in a folder.

    Parameters:
        input_folder: input folder (noise-free images)
        output_folder: output folder
        nsr_min: minimum NSR (%)
        nsr_max: maximum NSR (%)
        n_steps: number of noise levels
        distribution: NSR distribution type ('linear', 'log', 'quadratic')
        save_metrics: whether to save metrics
        verify_nsr: whether to verify actual NSR
    """
    os.makedirs(output_folder, exist_ok=True)

    nsr_values = generate_nsr_range(nsr_min, nsr_max, n_steps, distribution)

    print(f"\n{'=' * 70}")
    print(f"Generating Gaussian noise sequence: NSR {nsr_min}% → {nsr_max}% ({n_steps} steps)")
    print(f"{'=' * 70}")
    print(f"NSR values: {[f'{v:.2f}%' for v in nsr_values]}")
    print(f"Distribution: {distribution}")
    print(f"{'=' * 70}\n")

    file_list = sorted([f for f in os.listdir(input_folder)
                        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])
    print(f"Number of files: {len(file_list)}\n")

    first_img = iio.imread(os.path.join(input_folder, file_list[0]))
    if first_img.ndim == 3:
        first_img = np.mean(first_img, axis=2)

    all_results = []

    for idx, target_nsr in enumerate(nsr_values, 1):
        sigma = estimate_gaussian_parameters(first_img, target_nsr)

        sub_folder = output_folder
        os.makedirs(sub_folder, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Noise level {idx}/{n_steps}: target NSR = {target_nsr:.2f}%")
        print(f"{'=' * 70}")
        print(f"Estimated Gaussian sigma = {sigma:.6f}")

        actual_nsr_list = []
        psnr_list = []

        for fname in tqdm(file_list, desc=f"Processing images"):
            img = iio.imread(os.path.join(input_folder, fname))
            if img.ndim == 3:
                img = np.mean(img, axis=2)

            noisy, used_sigma = add_gaussian_noise_with_nsr(img, target_nsr, sigma)

            name, ext = os.path.splitext(fname)
            out_name = f"{name}_gau{idx:02d}_nsr{target_nsr:.1f}{ext}"
            out_path = os.path.join(sub_folder, out_name)
            iio.imwrite(out_path, noisy)

            if verify_nsr:
                metrics = compute_noise_metrics_fast(img, noisy)
                actual_nsr_list.append(metrics['NSR_percent'])
                psnr_list.append(metrics['PSNR'])

        if verify_nsr and actual_nsr_list:
            avg_actual_nsr = np.mean(actual_nsr_list)
            std_actual_nsr = np.std(actual_nsr_list)
            avg_psnr = np.mean(psnr_list)

            print(f"\nVerification results:")
            print(f"   Target NSR: {target_nsr:.2f}%")
            print(f"   Actual NSR: {avg_actual_nsr:.2f}% ± {std_actual_nsr:.2f}%")
            print(f"   Error: {abs(avg_actual_nsr - target_nsr):.2f}%")
            print(f"   Average PSNR: {avg_psnr:.2f} dB")

            result = {
                'index': idx,
                'target_NSR_percent': target_nsr,
                'sigma': sigma,
                'actual_NSR_percent': avg_actual_nsr,
                'NSR_std': std_actual_nsr,
                'NSR_error': abs(avg_actual_nsr - target_nsr),
                'PSNR_mean': avg_psnr
            }
        else:
            result = {
                'index': idx,
                'target_NSR_percent': target_nsr,
                'sigma': sigma
            }

        all_results.append(result)
        print(f"Completed: {sub_folder}")

    if save_metrics:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_folder, 'gaussian_nsr_sweep_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}")
        print(f"Gaussian NSR sweep summary")
        print(f"{'=' * 70}")
        print(df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'=' * 70}\n")

    return all_results


def process_folder_nsr_sweep(input_folder, output_folder,
                             nsr_min=0, nsr_max=10, n_steps=11,
                             distribution='linear',
                             anisotropic=False,
                             save_metrics=True,
                             verify_nsr=True):
    """
    Add a series of noise levels from nsr_min to nsr_max to all images in a folder.

    Parameters:
        input_folder: input folder (noise-free sinograms)
        output_folder: output folder
        nsr_min: minimum NSR (%)
        nsr_max: maximum NSR (%)
        n_steps: number of noise levels
        distribution: NSR distribution type ('linear', 'log', 'quadratic')
        anisotropic: whether to use anisotropic noise
        save_metrics: whether to save metrics
        verify_nsr: whether to verify actual NSR
    """
    os.makedirs(output_folder, exist_ok=True)

    # Generate NSR sequence
    nsr_values = generate_nsr_range(nsr_min, nsr_max, n_steps, distribution)

    print(f"\n{'=' * 70}")
    print(f"Generating noise sequence: NSR {nsr_min}% → {nsr_max}% ({n_steps} steps)")
    print(f"{'=' * 70}")
    print(f"NSR values: {[f'{v:.2f}%' for v in nsr_values]}")
    print(f"Distribution: {distribution}")
    print(f"{'=' * 70}\n")

    # Get file list
    file_list = sorted([f for f in os.listdir(input_folder)
                        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])

    print(f"Number of files: {len(file_list)}\n")

    # Read first image for parameter estimation
    first_img = iio.imread(os.path.join(input_folder, file_list[0]))
    if first_img.ndim == 3:
        first_img = np.mean(first_img, axis=2)

    # Store information for all noise levels
    all_results = []

    for idx, target_nsr in enumerate(nsr_values, 1):
        # Estimate parameters
        I0, sigma = estimate_poisson_parameters(first_img, target_nsr)

        # Create output folder
        folder_name = f"{idx:02d}_NSR_{target_nsr:.2f}percent_I0_{I0:.0f}_sigma_{sigma:.4f}"
        # sub_folder = os.path.join(output_folder, folder_name)
        sub_folder = output_folder
        os.makedirs(sub_folder, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"Noise level {idx}/{n_steps}: target NSR = {target_nsr:.2f}%")
        print(f"{'=' * 70}")
        print(f"Estimated parameters: I0 = {I0:.1f}, σ = {sigma:.6f}")

        # Process all images
        actual_nsr_list = []
        psnr_list = []
        for fname in tqdm(file_list, desc=f"Processing images"):
            # Read image
            img = iio.imread(os.path.join(input_folder, fname))
            if img.ndim == 3:
                img = np.mean(img, axis=2)

            # Add noise
            noisy, actual_I0, actual_sigma = add_quantitative_noise_with_nsr(
                img, target_nsr, anisotropic, I0, sigma
            )

            # Save
            name, ext = os.path.splitext(fname)
            out_name = f"{name}_noi{idx:02d}{ext}"
            out_path = os.path.join(sub_folder, out_name)
            iio.imwrite(out_path, noisy)

            # Verify actual NSR
            if verify_nsr:
                metrics = compute_noise_metrics_fast(img, noisy)
                actual_nsr_list.append(metrics['NSR_percent'])
                psnr_list.append(metrics['PSNR'])

        # Compute average actual NSR
        if verify_nsr and actual_nsr_list:
            avg_actual_nsr = np.mean(actual_nsr_list)
            std_actual_nsr = np.std(actual_nsr_list)
            avg_psnr = np.mean(psnr_list)

            print(f"\nVerification results:")
            print(f"   Target NSR: {target_nsr:.2f}%")
            print(f"   Actual NSR: {avg_actual_nsr:.2f}% ± {std_actual_nsr:.2f}%")
            print(f"   Error: {abs(avg_actual_nsr - target_nsr):.2f}%")
            print(f"   Average PSNR: {avg_psnr:.2f} dB")

            result = {
                'index': idx,
                'folder': folder_name,
                'target_NSR_percent': target_nsr,
                'I0': I0,
                'sigma': sigma,
                'actual_NSR_percent': avg_actual_nsr,
                'NSR_std': std_actual_nsr,
                'NSR_error': abs(avg_actual_nsr - target_nsr),
                'PSNR_mean': avg_psnr
            }
        else:
            result = {
                'index': idx,
                'folder': folder_name,
                'target_NSR_percent': target_nsr,
                'I0': I0,
                'sigma': sigma
            }

        all_results.append(result)
        print(f"Completed: {sub_folder}")

    # Save summary
    if save_metrics:
        df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_folder, 'nsr_sweep_summary.csv')
        df.to_csv(summary_path, index=False)

        print(f"\n{'=' * 70}")
        print(f"NSR sweep summary")
        print(f"{'=' * 70}")
        print(df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")
        print(f"{'=' * 70}\n")

    return all_results


def calibrate_nsr_parameters(input_folder, target_nsr_values,
                             output_folder=None, n_calibration_images=5):
    """
    Calibrate NSR parameters (find optimal I0 and sigma through actual testing).

    Parameters:
        input_folder: input folder
        target_nsr_values: list of target NSR values (e.g., [0, 2, 5, 8, 10])
        output_folder: output folder for calibration results
        n_calibration_images: number of images used for calibration

    Returns:
        calibrated_params: dictionary of calibrated parameters
    """
    print(f"\n{'=' * 70}")
    print(f"NSR parameter calibration")
    print(f"{'=' * 70}\n")

    # Read calibration images
    file_list = sorted([f for f in os.listdir(input_folder)
                        if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg'))])

    calib_files = file_list[:min(n_calibration_images, len(file_list))]

    calibrated_params = []

    for target_nsr in target_nsr_values:
        print(f"Calibrating target NSR = {target_nsr:.2f}%")

        # Initial estimation
        first_img = iio.imread(os.path.join(input_folder, calib_files[0]))
        if first_img.ndim == 3:
            first_img = np.mean(first_img, axis=2)

        I0_init, sigma_init = estimate_poisson_parameters(first_img, target_nsr)

        # Test multiple I0 values
        I0_candidates = [I0_init * k for k in [0.5, 0.75, 1.0, 1.25, 1.5]]

        best_I0 = I0_init
        best_sigma = sigma_init
        best_error = float('inf')

        for I0_test in I0_candidates:
            nsr_list = []

            for fname in calib_files:
                img = iio.imread(os.path.join(input_folder, fname))
                if img.ndim == 3:
                    img = np.mean(img, axis=2)

                noisy, _, _ = add_quantitative_noise_with_nsr(
                    img, target_nsr, False, I0_test, sigma_init
                )

                metrics = compute_noise_metrics_fast(img, noisy)
                nsr_list.append(metrics['NSR_percent'])

            avg_nsr = np.mean(nsr_list)
            error = abs(avg_nsr - target_nsr)

            if error < best_error:
                best_error = error
                best_I0 = I0_test

        print(f"   Optimal I0: {best_I0:.1f}")
        print(f"   Error: {best_error:.2f}%")

        calibrated_params.append({
            'target_NSR': target_nsr,
            'I0': best_I0,
            'sigma': best_sigma
        })

    # Save calibration results
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame(calibrated_params)
        calib_path = os.path.join(output_folder, 'calibrated_nsr_params.csv')
        df.to_csv(calib_path, index=False)

    return calibrated_params


# ==================== Usage example ====================
if __name__ == '__main__':
    input_folder =  './test/sinogram60views'
    output_folder = './test/sinogram60views_noise'   # output folder

    # ==================== Training ====================
    # input_folder = './train/sinogram60views'
    # output_folder = './train/sinogram60views_gaussian_noise'
    #
    # results = process_folder_gaussian_nsr_list(
    #     input_folder=input_folder,
    #     output_folder=output_folder,
    #     nsr_list=[1, 3, 6],
    #     save_metrics=True,
    #     verify_nsr=True
    # )

    # ==================== Inference ====================
    results = process_folder_nsr_sweep(
        input_folder=input_folder,
        output_folder=output_folder,
        nsr_min=0,         # minimum NSR: 0% (noise-free)
        nsr_max=10,        # maximum NSR: 10%
        n_steps=15,        # 15 levels: 0, 1, 2, ..., 10% (actually step = 10/14 ≈0.714)
        distribution='linear',   # 'linear', 'log', 'quadratic'
        anisotropic=False,
        save_metrics=True,
        verify_nsr=True    # verify actual NSR
    )
