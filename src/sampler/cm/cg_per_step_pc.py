import math
import numpy as np
import torch

import torch.nn.functional as F

@torch.no_grad()
def cg_per_step_pc(
        distiller, x, sigmas, ts, ks, clf, target_class_id, grad_at_p, grad_at_c, step_size,
        scheduler_type, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):
    '''
    ks - number of corrector steps per each predictor step
    '''

    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x

    @torch.enable_grad()
    def get_clf_grad(x):
        x.requires_grad_()
        logits = clf(normalize(x))
        log_probs = F.log_softmax(logits, dim = -1)
        target_class_log_probs = log_probs[:, target_class_id].unsqueeze(1)
        grad = torch.autograd.grad(target_class_log_probs.sum(), x)[0]
        return grad

    def step_size_scheduler(init_step_size, total_step, scheduler_type):
        if scheduler_type == 'cos':
            return lambda step: 0.5 * init_step_size * (1 + math.cos(math.pi * (step / total_step)))
        elif scheduler_type == 'const':
            return lambda step: init_step_size

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = []

    if fix_noise:
        noise = x.clone()

    if grad_at_p and grad_at_c:
        total_step = sum([e + 1 for e in ks])
    elif grad_at_p and not grad_at_c:
        total_step = len(ts)
    elif not grad_at_p and grad_at_c:
        total_step = sum([e for e in ks])

    step = 0
    scheduler = step_size_scheduler(step_size, total_step, scheduler_type)

    for i, k in zip(range(len(ts)), ks):

        if not fix_noise:
            noise = torch.randn_like(x)

        # get variance for ith iteration of predictor
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        t = np.clip(t, t_min, t_max)

        # add properly scaled noise for all iterations except the first one
        # where x comes from standard gaussian and we scale it with highest variance
        if i != 0:
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        else:
            x *= sigmas[0]

        # denoise and save
        x0 = distiller(x, t * s_in)
        xs.append(x0)

        # optionally move along clf gradient
        if grad_at_p:
            step_size_ = scheduler(step)
            step += 1
            grad = get_clf_grad(x0)
            x0 += step_size_ * grad

        # iterate over corrector steps
        for c_iter in range(k):

            if not fix_noise:
                noise = torch.randn_like(x)

            # scale x0 with fixed noise scale
            x = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)

            # denoise and save
            x0 = distiller(x, t * s_in)
            xs.append(x0)

            # optionally move along clf gradient
            if grad_at_c:
                step_size_ = scheduler(step)
                step += 1
                grad = get_clf_grad(x0)
                x0 += step_size_ * grad

    return torch.cat(xs, 0)