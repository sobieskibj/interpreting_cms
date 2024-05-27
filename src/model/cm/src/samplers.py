import numpy as np
import torch
import torchvision

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
def sample_heun(denoiser, x, sigmas, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, fix_noise = False):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        if fix_noise:
            # repeats the same noise for the entire batch
            noise  = torch.randn_like(x[0]).unsqueeze(0).repeat_interleave(x.shape[0], dim = 0)
        else:
            noise = torch.randn_like(x)
        eps = noise * s_noise
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

### CM samplers ###

@torch.no_grad()
def sample_onestep(distiller, x, sigmas, steps = 40):
    """
    Single-step generation from a distilled model.
    
    Modified to change std of x at the sampler level and not on the outside.
    """
    s_in = x.new_ones([x.shape[0]])
    return distiller(x * sigmas[0], sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller, x, sigmas, ts, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False, correction = True):

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (len(ts) - 1)

    if fix_noise:
        noise = x.clone()

    # scale to proper std
    x =  x * sigmas[0]

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        if not fix_noise:
            noise = torch.randn_like(x)

        xs[i] = x0

        if correction:
            x = x0 + noise * np.sqrt(next_t**2 - t_min**2)
        else:
            x = x0 + noise * next_t

    return torch.cat(xs, 0)


@torch.no_grad()
def max_noise_sampler(
    distiller, x, sigmas, k, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):

    s_in = x.new_ones([x.shape[0]])
    xs = []
    
    if fix_noise:
        # fixes the noise to always move along the same trajectory
        noise  = x.clone()

    # scale to proper std
    x =  x * sigmas[0]

    for _ in range(k):
        x0 = distiller(x, t_max * s_in)

        if not fix_noise:
            noise = torch.randn_like(x)

        xs.append((x0 + 1) / 2)
        x = x0 + noise * np.sqrt(t_max ** 2 - t_min ** 2)

    return torch.cat(xs, 0)


@torch.no_grad()
def fixed_noise_scale_sampler(
        distiller, 
        x, 
        sigmas, 
        k, 
        t_step, 
        t_min = 0.002, 
        t_max = 80.0, 
        rho = 7.0, 
        steps = 40, 
        fix_noise = False, 
        correction = True):
    '''
    t_step - timestep indicating the fixed noise scale
    '''
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (k + 1)

    if fix_noise:
        # fixes the noise to always move along the same trajectory
        noise  = x.clone()

    # scale to proper std
    x =  x * sigmas[0]

    # obtain initial x0 from full noise
    x0 = distiller(x, t_max * s_in)
    xs[0] = x0

    # iterate using a fixed noise scale
    t = (t_min_rho + t_step / (steps - 1) * (t_max_rho - t_min_rho)) ** rho

    for k_step in range(1, k + 1):

        if not fix_noise:
            # if trajectory is not fixed, sample a random one
            noise = torch.randn_like(x)

        # either use a corrected scale or not
        if correction:
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        else:
            x = x0 + noise * t

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[k_step] = x0

    return torch.cat(xs, 0)

@torch.no_grad()
def noiser_fixed_scale_sampler(
        distiller, 
        x, 
        sigmas, 
        k, 
        t_step, 
        t_min = 0.002, 
        t_max = 80.0, 
        rho = 7.0, 
        steps = 40, 
        fix_noise = False):
    '''
    t_step - timestep indicating the fixed noise scale
    k - number of iterations on a fixed noise scale
    '''

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (k + 1)

    # save initial x
    xs[0] = x
    x0 = x

    if fix_noise:
        # fixes the noise to always move along the same trajectory
        noise  = torch.randn_like(x0)

    # iterate using a fixed noise scale
    t = (t_min_rho + t_step / (steps - 1) * (t_max_rho - t_min_rho)) ** rho

    for k_step in range(1, k + 1):

        if not fix_noise:
            # if trajectory is not fixed, sample a random one
            noise = torch.randn_like(x0)

        # add noise
        x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[k_step] = x0

    return torch.cat(xs, 0)

@torch.no_grad()
def pc_sampler(
    distiller, x, sigmas, ts, k, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):
    '''
    k - number of corrector steps for each predictor step
    '''

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (len(ts) - 1) * (k + 1)

    if fix_noise:
        noise = x.clone()

    for i in range(len(ts) - 1):

        if not fix_noise:
            noise = torch.randn_like(x)

        # get variance for ith iteration of predictor
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho

        # add properly scaled noise for all iterations except the first one
        # where x comes from standard gaussian and we scale it with highest variance
        if i != 0:
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        else:
            x *= sigmas[0]

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[i * (k + 1)] = x0

        # get variance for k corrector iterations
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        # iterate over corrector steps
        for c_iter in range(k):

            if not fix_noise:
                noise = torch.randn_like(x)

            # scale x0 with fixed noise scale
            x = x0 + noise * np.sqrt(next_t ** 2 - t_min ** 2)

            # denoise and save
            x0 = distiller(x, next_t * s_in)
            xs[i * (k + 1) + c_iter + 1] = x0

    return torch.cat(xs, 0)

@torch.no_grad()
def cf_sampler(
        distiller, 
        x, 
        sigmas, 
        clf,
        step_size,
        k, 
        t_step, 
        t_min = 0.002, 
        t_max = 80.0, 
        rho = 7.0, 
        steps = 40, 
        fix_noise = False):
    '''
    t_step - timestep indicating the fixed noise scale
    k - number of iterations on a fixed noise scale
    '''
    import wandb

    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x

    # as a quick fix, we manually move clf to proper device
    clf = clf.to(x.device)

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (k + 1)
    xs_cf = [None] * (k + 1)
    xs_post_grad = [None] * k
    xs_diff = [None] * k
    xs_diff_post_grad = [None] * k

    # save initial x
    xs[0] = x
    xs_cf[0] = x
    x0 = x
    gt = x.clone()

    if fix_noise:
        # fixes the noise to always move along the same trajectory
        noise  = torch.randn_like(x0)

    # iterate using a fixed noise scale
    t = (t_min_rho + t_step / (steps - 1) * (t_max_rho - t_min_rho)) ** rho

    # get original predictions
    init_pred_class_idx = clf(normalize(x0)).argmax(dim = 1)

    for k_step in range(1, k + 1):

        # add scaled gradient of the classifier to x0
        with torch.enable_grad():
            x0.requires_grad_()

            logits = clf(normalize(x0))

            pred_class_logits = logits.gather(1, logits.argmax(dim = 1).unsqueeze(1))
            grad = torch.autograd.grad(pred_class_logits.sum(), x0)[0]
        
            x0 = x0 - step_size * grad

            x0 = x0.detach()

        if not fix_noise:
            # if trajectory is not fixed, sample a random one
            noise = torch.randn_like(x0)

        # save post gradient step x
        x0_post_grad = x0.clone()
        pred_class_idx = clf(normalize(x0_post_grad)).argmax(dim = 1)
        x0_post_grad[pred_class_idx == init_pred_class_idx] = 0.
        xs_post_grad[k_step - 1] = x0_post_grad

        # save diffs
        diff = (x0 - gt).abs()
        diff[pred_class_idx == init_pred_class_idx] = 0.
        xs_diff_post_grad[k_step - 1] = normalize(diff)

        # TODO: maybe we can make an overlaying mask for the 
        # gradient and replace only the parts within the mask

        # add noise
        x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[k_step] = x0

        # check whether the predictions changed
        pred_class_idx = clf(normalize(x0)).argmax(dim = 1)
        cf = x0.clone()
        cf[pred_class_idx == init_pred_class_idx] = 0.
        xs_cf[k_step] = cf

        diff = (x0 - gt).abs()
        diff[pred_class_idx == init_pred_class_idx] = 0.
        xs_diff[k_step - 1] = normalize(diff)

    flipped = torch.cat(xs_cf, 0)
    post_grad = torch.cat(xs_post_grad, 0)
    diff = torch.cat(xs_diff, 0)

    wandb.log({
        'flipped': wandb.Image(flipped),
        'post_grad': wandb.Image(post_grad),
        'diff': wandb.Image(diff),
        'diff_post_grad': wandb.Image(diff)})

    return torch.cat(xs, 0)

@torch.no_grad()
def classifier_guidance_pc_sampler(
        distiller, x, sigmas, ts, k, clf, target_class_id, grad_at_p, grad_at_c, step_size,
        t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):
    '''
    k - number of corrector steps per predictor step
    '''

    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x

    @torch.enable_grad()
    def get_clf_grad(x):
        x.requires_grad_()
        logits = clf(normalize(x))
        target_class_logits = logits[:, target_class_id].unsqueeze(1)
        grad = torch.autograd.grad(target_class_logits.sum(), x)[0]
        return grad

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = [None] * (len(ts) - 1) * (k + 1)

    if fix_noise:
        noise = x.clone()

    for i in range(len(ts) - 1):

        if not fix_noise:
            noise = torch.randn_like(x)

        # get variance for ith iteration of predictor
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho

        # add properly scaled noise for all iterations except the first one
        # where x comes from standard gaussian and we scale it with highest variance
        if i != 0:
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        else:
            x *= sigmas[0]

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs[i * (k + 1)] = x0

        # optionally move along clf gradient
        if grad_at_p:
            x0 += step_size * get_clf_grad(x0)

        # get variance for k corrector iterations
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        # iterate over corrector steps
        for c_iter in range(k):

            if not fix_noise:
                noise = torch.randn_like(x)

            # scale x0 with fixed noise scale
            x = x0 + noise * np.sqrt(next_t ** 2 - t_min ** 2)

            # denoise and save
            x0 = distiller(x, next_t * s_in)
            xs[i * (k + 1) + c_iter + 1] = x0

            # optionally move along clf gradient
            if grad_at_c:
                x0 += step_size * get_clf_grad(x0)

    return torch.cat(xs, 0)


### Progressive Distillation ###

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
