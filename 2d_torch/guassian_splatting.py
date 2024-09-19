#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from functions import set_seed, load_image, sample_gaussians, combined_loss


set_seed(42)

LEARNING_RATE = 0.01

N_PRIMARY_SAMPLES = 2000
N_BACKUP_SAMPLES = 0

N_SAMPLES = N_PRIMARY_SAMPLES + N_BACKUP_SAMPLES

IMAGE_SIZE = [256, 256, 3]

NEGATIVE_GUASSIAN = False

image, shape = load_image('COLOR_TARGET.jpg', IMAGE_SIZE)
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
if NEGATIVE_GUASSIAN:
    alpha_values = torch.atanh(alpha_values)  # Change from positive to negative
else:
    alpha_values = torch.logit(alpha_values)  # Change from positive to negative

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


def train(gaussian_variables, NEGATIVE_GUASSIAN, shape, device, image_tensor, optimizer, epochs,loss_history):
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            
            predicted, pos_img, neg_img = sample_gaussians(gaussian_variables, shape, device, NEGATIVE_GUASSIAN)
            loss = combined_loss(predicted, image_tensor, lambda_param=0.2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            # Get VRAM usage (allocated memory on the GPU in MB)
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
            
            # Update the progress bar with loss and VRAM info
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'VRAM': f'{vram_allocated:.2f} MB'})
            
            # Update the progress bar
            pbar.update(1)
            if (epoch % 100 == 0):
                print(f"Epoch {epoch} -> loss {loss.item()}")
                print(f"VRAM: {torch.cuda.memory_summary()}")
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

    return gaussian_variables, loss_history

#%%

# Positive training only
W_pos_int, loss_history_int = train(W, False, shape, device, image_tensor, optimizer, 700, loss_history)
torch.cuda.empty_cache() 


# Postive training contineous 
loss_history_pos_final = loss_history_int.copy()
W_pos_final = nn.Parameter(W_pos_int.clone().detach())


# Ensure the optimizer is updated with the new parameter
optimizer_pos = torch.optim.Adam([W_pos_final], lr=LEARNING_RATE)
optimizer_pos.load_state_dict(optimizer.state_dict())

W_pos_final, loss_history_pos_final = train(W_pos_final, False, shape, device, image_tensor, optimizer_pos, 500, loss_history_pos_final)
del W_pos_final
torch.cuda.empty_cache() 

# Negative training contineous ----------------------------------------
loss_history_neg_final = loss_history_int.copy()


W_neg_final = W_pos_int.clone()


alpha = W_neg_final[:, 3]
W_neg_final[:, 3] = torch.atanh(F.sigmoid(alpha))

W_neg_final = nn.Parameter(W_neg_final.detach())

# Ensure the optimizer is updated with the new parameter
optimizer_neg = torch.optim.Adam([W_neg_final], lr=LEARNING_RATE)
optimizer_neg.load_state_dict(optimizer.state_dict())

W_neg_final, loss_history_neg_final = train(W_neg_final, True, shape, device, image_tensor, optimizer_neg, 500, loss_history_neg_final)
del W_neg_final
torch.cuda.empty_cache() 

#%%

plt.plot(loss_history_neg_final, 'b', label='Negative Guassian')  # Red line for negative loss
plt.plot(loss_history_pos_final, 'r', label='Positive Guassian')  # Blue line for positive loss

# Set y-axis to logarithmic scale
plt.yscale('log')


# Manually set y-ticks for more control
ticks = [1, 0.5, 0.1, 0.05, 0.025]  # Adjust based on your data range
plt.yticks(ticks, labels=[f'{tick:g}' for tick in ticks])

# Add axis labels
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')

# Add a legend
plt.legend()

# %%


def get_W(N_SAMPLES, image_tensor):

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
    if NEGATIVE_GUASSIAN:
        alpha_values = torch.atanh(alpha_values)  # Change from positive to negative
    else:
        alpha_values = torch.logit(alpha_values)  # Change from positive to negative

    colour_values = torch.logit(colour_values)
    coords_norm = torch.atanh(coords_norm)
    W_values = torch.cat([sigma_values, rho_values, alpha_values, colour_values, coords_norm], dim=1)
    
    return W_values


W_values = get_W(N_SAMPLES, image_tensor)
W_fully_neg = nn.Parameter(W_values)
optimizer_pos = torch.optim.Adam([W_fully_neg], lr=LEARNING_RATE)

loss_history_full_neg = []

W_pos_final, loss_history_pos_final = train(W_fully_neg, True, shape, device, image_tensor, optimizer_pos, 1000, loss_history_full_neg)
# %%


plt.plot(loss_part_neg, 'b', label='Part Negative Guassian')  # Red line for negative loss
plt.plot(loss_positive, 'r', label='Positive Guassian')  # Blue line for positive loss
plt.plot(loss_history_full_neg, 'g', label='Negative Guassian') 
# Set y-axis to logarithmic scale
plt.yscale('log')


# Manually set y-ticks for more control
ticks = [1, 0.5, 0.1, 0.05, 0.025]  # Adjust based on your data range
plt.yticks(ticks, labels=[f'{tick:g}' for tick in ticks])

# Add axis labels
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')

# Add a legend
plt.legend()
# %%
