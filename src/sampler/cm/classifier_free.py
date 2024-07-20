import numpy as np
import torch

import wandb

@torch.no_grad()
def classifier_free(
        distiller, x, sigmas, clf, step_size, k, t_step, t_min = 0.002, 
        t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):
    '''
    t_step - timestep indicating the fixed noise scale
    k - number of iterations on a fixed noise scale
    '''

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