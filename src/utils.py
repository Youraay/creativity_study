import torch
from PIL.Image import Image

def latents_to_rgb(latents):
    """
    Convert the SDXL latents (4 channels: Luminance, Cyan, Red, Pattern Structure) to RGB tensors (3 channels).
    Explanation: https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#the-4-channels-of-the-sdxl-latents
    """
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35),
    )

    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255).byte().cpu().numpy().transpose(1, 2, 0)

    return Image.fromarray(image_array)