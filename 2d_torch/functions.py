import torch
import torch.nn as nn 
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

def set_seed(seed):
    """
    Set the seed for reproducibility in NumPy, PyTorch, and CUDA operations.
    
    This function ensures that all random number generators are seeded with the same value 
    for reproducibility across different runs of the same code. It works for both CPU and 
    GPU (CUDA) computations, ensuring that the results are consistent across multiple runs 
    of the model.

    Args:
        seed (int): The seed value to use for random number generation.

    Side Effects:
        - Sets the random seed for NumPy using `np.random.seed()`.
        - Sets the random seed for PyTorch using `torch.manual_seed()`.
        - If CUDA is available:
            - Sets the random seed for CUDA operations using `torch.cuda.manual_seed()` 
              and `torch.cuda.manual_seed_all()`.
        - Ensures deterministic behavior in PyTorch's backend by setting:
            - `torch.backends.cudnn.deterministic = True`: Ensures deterministic 
              computation in convolution operations.
            - `torch.backends.cudnn.benchmark = False`: Disables the CUDNN 
              auto-tuner that selects the fastest convolution algorithm based on 
              the hardware and input size, to ensure deterministic behavior.

    Notes:
        - Setting `torch.backends.cudnn.deterministic = True` can make the computation 
          slower, but it guarantees reproducibility.
        - `torch.backends.cudnn.benchmark = False` may affect performance by disabling 
          optimizations, but it ensures deterministic results.

    Example:
        >>> set_seed(42)
        # This will set the seed for NumPy, PyTorch (CPU and CUDA) to 42, ensuring 
        # that all random number generation is consistent across runs.
    """
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_window(window_size, channel):
    def gaussian(window_size, sigma):
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, window_size=11, size_average=True):

    # Assuming the image is of shape [N, C, H, W]
    (_, _, channel) = img1.size()

    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)

    # Parameters for SSIM
    C1 = 0.01**2
    C2 = 0.03**2

    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    SSIM_numerator = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    SSIM_denominator = ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    SSIM = SSIM_numerator / SSIM_denominator

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def d_ssim_loss(img1, img2, window_size=11, size_average=True):
    return ssim(img1, img2, window_size, size_average).mean()


# Combined Loss
def combined_loss(pred, target, lambda_param=0.5):
    l1loss = nn.L1Loss()
    return (1 - lambda_param) * l1loss(pred, target) + lambda_param * d_ssim_loss(pred, target)


def sample_gaussians(gaussian_variables, shape, device, NEGATIVE_GUASSIAN, kernel_size=101):
    # pull the variables
    batch_size = gaussian_variables.shape[0]
    sigma_x = gaussian_variables[:, 0].view(batch_size, 1, 1)
    sigma_y = gaussian_variables[:, 1].view(batch_size, 1, 1)
    rho = gaussian_variables[:, 2].view(batch_size, 1, 1)
    alpha = gaussian_variables[:, 3].view(batch_size, 1)
    colors = gaussian_variables[:, 4:7]
    coords = gaussian_variables[:, 7:9]

    # fix the variables
    sigma_x = F.sigmoid(sigma_x)
    sigma_y = F.sigmoid(sigma_y)
    rho = F.tanh(rho)
    
    if NEGATIVE_GUASSIAN:
        alpha = F.tanh(alpha)  # postive to negative
    else:
        alpha = F.sigmoid(alpha)  # postive to negative
    
    colors = F.sigmoid(colors)
    coords = F.tanh(coords)
    
    # weight the colors values with alpha
    colors = colors * alpha

    # get covariance
    covariance = torch.stack(
        [torch.stack([sigma_x**2, rho*sigma_x*sigma_y], dim=-1),
         torch.stack([rho*sigma_x*sigma_y, sigma_y**2], dim=-1)],
        dim=-2
    )

    determinant = (sigma_x**2) * (sigma_y**2) - (rho * sigma_x * sigma_y)**2
    if (determinant <= 0).any():
        raise ValueError("Covariance matrix must be positive semi-definite")

    inverse_covariance = torch.inverse(covariance)

    xx = torch.linspace(start=-5, end=5, steps=kernel_size, device=device)
    yy = torch.linspace(start=-5, end=5, steps=kernel_size, device=device)
    xx, yy = torch.meshgrid(xx, yy, indexing='ij')
    xx, yy = xx.unsqueeze(0), yy.unsqueeze(0)
    xy = torch.stack((xx, yy), dim=-1)

    z = torch.einsum('b...i,b...ij,b...j->b...', xy, -0.5 * inverse_covariance, xy)
    kernel = torch.exp(z) / (2 * torch.tensor(np.pi, device=device) * torch.sqrt(torch.det(covariance)).view(batch_size, 1, 1))
    
    kernel_max_1, _ = kernel.max(dim=-1, keepdim=True)  # Find max along the last dimension
    kernel_max_2, _ = kernel_max_1.max(dim=-2, keepdim=True)  # Find max along the second-to-last dimension
    kernel_normalized = kernel / kernel_max_2

    kernel_reshaped = kernel_normalized.repeat(1, 3, 1).view(batch_size * 3, kernel_size, kernel_size)
    kernel_rgb = kernel_reshaped.unsqueeze(0).reshape(batch_size, 3, kernel_size, kernel_size)

    pad_h = shape[0] - kernel_size
    pad_w = shape[1] - kernel_size
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Kernel size should be smaller or equal to the image size.")
    
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2,  # padding left and right
               pad_h // 2, pad_h // 2 + pad_h % 2)  # padding top and bottom
    kernel_rgb_padded = torch.nn.functional.pad(kernel_rgb, padding, "constant", 0)  

    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    rgb_values_reshaped = colors.unsqueeze(-1).unsqueeze(-1)

    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)

    indx = alpha > 0
    indx = indx.squeeze()
    pos_img = rgb_values_reshaped[indx] * kernel_rgb_padded_translated[indx]
    final_pos_img = pos_img.sum(dim=0)
    final_pos_img = torch.clamp(final_pos_img, 0, 1)
    final_pos_img = final_pos_img.permute(1, 2, 0)

    indx = alpha < 0
    indx = indx.squeeze()
    neg_img = rgb_values_reshaped[indx] * kernel_rgb_padded_translated[indx]
    final_neg_img = -1*neg_img.sum(dim=0)
    final_neg_img = torch.clamp(final_neg_img, 0, 1)
    final_neg_img = final_neg_img.permute(1, 2, 0)

    return final_image, final_pos_img, final_neg_img


def load_image(image_path, IMAGE_SIZE):
    image = Image.open(image_path)
    image = image.resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    h, w, c = image.shape
    return image, (h, w, c)
