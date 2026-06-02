import numpy as np
import os

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from functools import partial
from scipy.fftpack import fft, ifft
import numpy.fft as fftmodule
from skimage import io
from skimage.transform import radon
import time
import tifffile


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)


def _get_fourier_filter(size, filter_name):
    """Construct the Fourier filter.

    This computation lessens artifacts and removes a small bias as
    explained in [1], Chap 3. Equation 61.

    Parameters
    ----------
    size: int
        filter size. Must be even.
    filter_name: str
        Filter used in frequency domain filtering. Filters available:
        ramp, shepp-logan, cosine, hamming, hann. Assign None to use
        no filter.

    Returns
    -------
    fourier_filter: ndarray
        The computed Fourier filter.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.

    """
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))         # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftmodule.fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftmodule.fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftmodule.fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftmodule.fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]

def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=False):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different
        angle. The tomography rotation axis should lie at the pixel
        index ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int, optional
        Number of rows and columns in the reconstruction.
    filter : str, optional
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

        projection_values: the view-by-view back projections (VVBP) Tensor

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter not in filter_types:
        raise ValueError("Unknown filter: %s" % filter)

    img_shape = radon_image.shape[0]
    if output_size is None:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))

    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size))
    bpall = np.zeros((output_size, output_size,angles_count),dtype='float32')
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    count=0
    for col, angle in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        else:
            interpolant = interp1d(x, col, kind=interpolation,
                                   bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)
        bpall[:,:,count]=interpolant(t)
        count+=1

    projection_values = bpall.reshape((int(output_size * output_size), angles_count))

    if circle:
        out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
        reconstructed[out_reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * angles_count), projection_values * np.pi / (2 * angles_count)
    # return projection_values * np.pi / (2 * angles_count)

if __name__ == "__main__":
    ffile = './test/sinogram60views'
    savef = './test/sinogram60views_VVBP'
    savesino = './test/sinogram60views'
    savefFBP = '../result/FBPresults'
    mkdir(savef)
    # mkdir(savesino)
    mkdir(savefFBP)

    fnames = os.listdir(ffile)
    times = []

    for name in fnames:
        imname, ext = os.path.splitext(name)
        fname = os.path.join(ffile, name)
        o_sino = np.array(io.imread(fname), dtype='float32')
        # o_sino = tifffile.imread(fname).astype(np.float32)
        o_sino = (o_sino - o_sino.min()) / (o_sino.max() - o_sino.min() + 1e-8)
        height, width = (256, 60)
        H = int(np.floor(height / 1.414))
        theta = np.linspace(0., 180, width, endpoint=False)
        print("sinogram shape:", o_sino.shape)
        print("theta len:", len(theta))
        # plt.figure(figsize=(10, 6))
        # plt.imshow(o_sino, cmap='gray', aspect='auto')
        # plt.title('B-spline Interpolation for Each Row of the Image')
        # plt.axis('off')
        # plt.show()

        t_start = time.perf_counter()
        # sino = radon(o_sino, theta=theta, circle=False)
        # io.imsave(os.path.join(savesino, imname + '.tif'), sino.astype(np.float32).squeeze())
        reconstructed, projection_values = iradon(o_sino, theta=theta, output_size=None,
                                                  filter="shepp-logan", interpolation="cubic", circle=False)
        projection_values = projection_values.astype(np.float32)
        io.imsave(os.path.join(savef, imname + '.tif'), projection_values.squeeze())
        # reconstructed2 = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)
        io.imsave(os.path.join(savefFBP, imname + '.tif'), reconstructed.astype(np.float32).squeeze())

        # projection_values = (projection_values - projection_values.min()) / (projection_values.max() - projection_values.min() + 1e-8)
        # projection_values = projection_values.clip(0, 1)
        # projection_values = (projection_values * 255).astype(np.uint8)
        # io.imsave(os.path.join(savef, imname + '.png'), projection_values.squeeze())

        reconstructed_vvbp = projection_values.sum(axis=1).reshape(H, H)
        recon_display = reconstructed_vvbp
        # recon_display = (recon_display - recon_display.min()) / (recon_display.max() - recon_display.min() + 1e-8)
        # io.imsave(os.path.join(savefFBP, imname + '.tif'), recon_display.astype(np.float32).squeeze())
        # recon_display = np.clip(recon_display * 255, 0, 255).astype(np.uint8)

        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"{imname}: Time cost of FBP reconstruction: {elapsed:.3f} s")
        times.append(elapsed)

        # error = np.abs(reconstructed - reconstructed2)
        #
        # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # # reconstructed
        # axes[0].imshow(reconstructed, cmap='gray', aspect='auto')
        # axes[0].set_title('reconstructed')
        # axes[0].axis('off')
        # # recon_display
        # axes[1].imshow(recon_display, cmap='gray', aspect='auto')
        # axes[1].set_title('recon_display')
        # axes[1].axis('off')
        # # error
        # im = axes[2].imshow(error, cmap='gray', aspect='auto')
        # axes[2].set_title('error')
        # axes[2].axis('off')
        #
        # fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        #
        # plt.tight_layout()
        # plt.show()

    print(f"\n Average reconstruction time: {np.mean(times):.3f} s/slice, total {len(times)} CT images.")

