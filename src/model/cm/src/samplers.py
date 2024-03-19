import numpy as np
import torch

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])

        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model, x, ts):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2

    return x


@torch.no_grad()
def sample_heun(denoiser, x, sigmas, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_euler(denoiser, x, sigmas):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)

        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(denoiser, x, sigmas, s_churn = 0.0, s_tmin = 0.0, s_tmax = float("inf"), s_noise = 1.0):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)

        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_onestep(distiller, x, sigmas):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
        distiller, x, sigmas, ts, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40):

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + torch.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def sample_progdist(denoiser, x, sigmas):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x

### not refactored below

# @torch.no_grad()
# def iterative_colorization(
#     distiller,
#     images,
#     x,
#     ts,
#     t_min=0.002,
#     t_max=80.0,
#     rho=7.0,
#     steps=40,
#     generator=None,
# ):
#     def obtain_orthogonal_matrix():
#         vector = np.asarray([0.2989, 0.5870, 0.1140])
#         vector = vector / np.linalg.norm(vector)
#         matrix = np.eye(3)
#         matrix[:, 0] = vector
#         matrix = np.linalg.qr(matrix)[0]
#         if np.sum(matrix[:, 0]) < 0:
#             matrix = -matrix
#         return matrix

#     Q = torch.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(torch.float32)
#     mask = torch.zeros(*x.shape[1:], device=dist_util.dev())
#     mask[0, ...] = 1.0

#     def replacement(x0, x1):
#         x0 = torch.einsum("bchw,cd->bdhw", x0, Q)
#         x1 = torch.einsum("bchw,cd->bdhw", x1, Q)

#         x_mix = x0 * mask + x1 * (1.0 - mask)
#         x_mix = torch.einsum("bdhw,cd->bchw", x_mix, Q)
#         return x_mix

#     t_max_rho = t_max ** (1 / rho)
#     t_min_rho = t_min ** (1 / rho)
#     s_in = x.new_ones([x.shape[0]])
#     images = replacement(images, torch.zeros_like(images))

#     for i in range(len(ts) - 1):
#         t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         x0 = distiller(x, t * s_in)
#         x0 = torch.clamp(x0, -1.0, 1.0)
#         x0 = replacement(images, x0)
#         next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         next_t = np.clip(next_t, t_min, t_max)
#         x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

#     return x, images


# @torch.no_grad()
# def iterative_inpainting(
#     distiller,
#     images,
#     x,
#     ts,
#     t_min=0.002,
#     t_max=80.0,
#     rho=7.0,
#     steps=40,
#     generator=None,
# ):
#     from PIL import Image, ImageDraw, ImageFont

#     image_size = x.shape[-1]

#     # create a blank image with a white background
#     img = Image.new("RGB", (image_size, image_size), color="white")

#     # get a drawing context for the image
#     draw = ImageDraw.Draw(img)

#     # load a font
#     font = ImageFont.truetype("arial.ttf", 250)

#     # draw the letter "C" in black
#     draw.text((50, 0), "S", font=font, fill=(0, 0, 0))

#     # convert the image to a numpy array
#     img_np = np.array(img)
#     img_np = img_np.transpose(2, 0, 1)
#     img_th = torch.from_numpy(img_np).to(dist_util.dev())

#     mask = torch.zeros(*x.shape, device=dist_util.dev())
#     mask = mask.reshape(-1, 7, 3, image_size, image_size)

#     mask[::2, :, img_th > 0.5] = 1.0
#     mask[1::2, :, img_th < 0.5] = 1.0
#     mask = mask.reshape(-1, 3, image_size, image_size)

#     def replacement(x0, x1):
#         x_mix = x0 * mask + x1 * (1 - mask)
#         return x_mix

#     t_max_rho = t_max ** (1 / rho)
#     t_min_rho = t_min ** (1 / rho)
#     s_in = x.new_ones([x.shape[0]])
#     images = replacement(images, -torch.ones_like(images))

#     for i in range(len(ts) - 1):
#         t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         x0 = distiller(x, t * s_in)
#         x0 = torch.clamp(x0, -1.0, 1.0)
#         x0 = replacement(images, x0)
#         next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         next_t = np.clip(next_t, t_min, t_max)
#         x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

#     return x, images


# @torch.no_grad()
# def iterative_superres(
#     distiller,
#     images,
#     x,
#     ts,
#     t_min=0.002,
#     t_max=80.0,
#     rho=7.0,
#     steps=40,
#     generator=None,
# ):
#     patch_size = 8

#     def obtain_orthogonal_matrix():
#         vector = np.asarray([1] * patch_size**2)
#         vector = vector / np.linalg.norm(vector)
#         matrix = np.eye(patch_size**2)
#         matrix[:, 0] = vector
#         matrix = np.linalg.qr(matrix)[0]
#         if np.sum(matrix[:, 0]) < 0:
#             matrix = -matrix
#         return matrix

#     Q = torch.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(torch.float32)

#     image_size = x.shape[-1]

#     def replacement(x0, x1):
#         x0_flatten = (
#             x0.reshape(-1, 3, image_size, image_size)
#             .reshape(
#                 -1,
#                 3,
#                 image_size // patch_size,
#                 patch_size,
#                 image_size // patch_size,
#                 patch_size,
#             )
#             .permute(0, 1, 2, 4, 3, 5)
#             .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
#         )
#         x1_flatten = (
#             x1.reshape(-1, 3, image_size, image_size)
#             .reshape(
#                 -1,
#                 3,
#                 image_size // patch_size,
#                 patch_size,
#                 image_size // patch_size,
#                 patch_size,
#             )
#             .permute(0, 1, 2, 4, 3, 5)
#             .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
#         )
#         x0 = torch.einsum("bcnd,de->bcne", x0_flatten, Q)
#         x1 = torch.einsum("bcnd,de->bcne", x1_flatten, Q)
#         x_mix = x0.new_zeros(x0.shape)
#         x_mix[..., 0] = x0[..., 0]
#         x_mix[..., 1:] = x1[..., 1:]
#         x_mix = torch.einsum("bcne,de->bcnd", x_mix, Q)
#         x_mix = (
#             x_mix.reshape(
#                 -1,
#                 3,
#                 image_size // patch_size,
#                 image_size // patch_size,
#                 patch_size,
#                 patch_size,
#             )
#             .permute(0, 1, 2, 4, 3, 5)
#             .reshape(-1, 3, image_size, image_size)
#         )
#         return x_mix

#     def average_image_patches(x):
#         x_flatten = (
#             x.reshape(-1, 3, image_size, image_size)
#             .reshape(
#                 -1,
#                 3,
#                 image_size // patch_size,
#                 patch_size,
#                 image_size // patch_size,
#                 patch_size,
#             )
#             .permute(0, 1, 2, 4, 3, 5)
#             .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
#         )
#         x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
#         return (
#             x_flatten.reshape(
#                 -1,
#                 3,
#                 image_size // patch_size,
#                 image_size // patch_size,
#                 patch_size,
#                 patch_size,
#             )
#             .permute(0, 1, 2, 4, 3, 5)
#             .reshape(-1, 3, image_size, image_size)
#         )

#     t_max_rho = t_max ** (1 / rho)
#     t_min_rho = t_min ** (1 / rho)
#     s_in = x.new_ones([x.shape[0]])
#     images = average_image_patches(images)

#     for i in range(len(ts) - 1):
#         t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         x0 = distiller(x, t * s_in)
#         x0 = torch.clamp(x0, -1.0, 1.0)
#         x0 = replacement(images, x0)
#         next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
#         next_t = np.clip(next_t, t_min, t_max)
#         x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

#     return x, images
