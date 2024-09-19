#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm


#%%

def set_seed(seed):
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


def sample_gaussians(gaussian_variables, kernel_size=101):
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
    alpha = F.tanh(alpha)
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


def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((IMAGE_SIZE[0],IMAGE_SIZE[1]))
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0
    h, w, c = image.shape
    return image, (h, w, c)


set_seed(42)

LEARNING_RATE = 0.01

N_PRIMARY_SAMPLES = 2000
N_BACKUP_SAMPLES = 0

N_SAMPLES = N_PRIMARY_SAMPLES + N_BACKUP_SAMPLES

IMAGE_SIZE = [256, 256, 3]

image, shape = load_image('COLOR_TARGET.jpg')
plt.imshow(image)

device = torch.device('cuda')
image_tensor = torch.tensor(image, dtype=torch.float32, device=device)
image_tensor.shape

coords = np.random.randint(0, [shape[0], shape[1]], size=(N_SAMPLES, 2))
print(coords.min(axis=0))
print(coords.max(axis=0))
# make it a tensor
coords = torch.tensor(coords, device=device)
coords.shape

random_pixels = image_tensor[coords[:, 0], coords[:, 1]]
random_pixels.shape

coords_norm = coords / torch.tensor([shape[0]-1, shape[1]-1], device=device).float()
coords_norm = coords_norm * 2 - 1
coords_norm.shape, coords_norm.min(), coords_norm.max()


colour_values = image_tensor[coords[:, 0], coords[:, 1]]
colour_values.shape, colour_values[:3, ...]

sigma_values = torch.rand(N_SAMPLES, 2, device=device)
rho_values = 2 * torch.rand(N_SAMPLES, 1, device=device) - 1
alpha_values = torch.ones(N_SAMPLES, 1, device=device)*0.8
print(alpha_values[:10])

sigma_values = torch.logit(sigma_values)
rho_values = torch.atanh(rho_values)
alpha_values = torch.atanh(alpha_values)
colour_values = torch.logit(colour_values)
coords_norm = torch.atanh(coords_norm)


W_values = torch.cat([sigma_values, rho_values, alpha_values, colour_values, coords_norm], dim=1)
W_values.shape
sigma_values.shape
print(alpha_values[:10])
print(W_values[:10, 3])


W = nn.Parameter(W_values)
optimizer = torch.optim.Adam([W], lr=LEARNING_RATE)
loss_history = []
W[:10, 3]

EPOCHS = 100
densification_interval = 10


#%%
for epoch in tqdm(range(1000)):

    gaussian_variables = W
    predicted, pos_img, neg_img = sample_gaussians(gaussian_variables)
    loss = combined_loss(predicted, image_tensor, lambda_param=0.2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if (epoch % 10 == 0):
        print(f"Epoch {epoch} -> loss {loss.item()}")

        plt.figure(figsize=(10, 5))

        # First subplot for the predicted image
        plt.subplot(1, 3, 1)
        plt.imshow(predicted.detach().to('cpu').numpy())
        plt.title('Predicted Image')

        # Second subplot for the negative image
        plt.subplot(1, 3, 2)
        plt.imshow(pos_img.detach().to('cpu').numpy())
        plt.title('Positive Image')

        # Second subplot for the negative image
        plt.subplot(1, 3, 3)
        plt.imshow(neg_img.detach().to('cpu').numpy())
        plt.title('Negative Image')
        plt.show()

        

# # Plot placeholders for predicted and neg_img to update later
# img1 = ax[0].imshow(np.zeros((256, 256)))  # assuming the image size is [256, 256]
# ax[0].set_title('Predicted Image')

# img2 = ax[1].imshow(np.zeros((256, 256)))  # assuming the image size is [256, 256]
# ax[1].set_title('Negative Image')



# # Initialize the figure and the axes outside the loop
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# # Plot placeholders for predicted and neg_img to update later
# img1 = ax[0].imshow(np.zeros((256, 256)))  # assuming the image size is [256, 256]
# ax[0].set_title('Predicted Image')

# img2 = ax[1].imshow(np.zeros((256, 256)))  # assuming the image size is [256, 256]
# ax[1].set_title('Negative Image')

# plt.ion()  # Turn on interactive mode

# for epoch in tqdm(range(1000)):

#     gaussian_variables = W
#     predicted, neg_img = sample_gaussians(gaussian_variables)
#     loss = combined_loss(predicted, image_tensor, lambda_param=0.2)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     loss_history.append(loss.item())

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch} -> loss {loss.item()}")
        
#         # Update the images' data without creating new figures
#         img1.set_data(predicted.detach().to('cpu').numpy())
#         img2.set_data(neg_img.detach().to('cpu').numpy())

#         # Redraw the canvas to show updated images
#         fig.canvas.draw()
#         fig.canvas.flush_events()

# plt.ioff()  # Turn off interactive mode when done



# %%
