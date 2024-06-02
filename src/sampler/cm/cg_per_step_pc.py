import torch
import numpy as np

@torch.no_grad()
def cg_per_step_pc(
        distiller, x, sigmas, ts, ks, clf, target_class_id, grad_at_p, grad_at_c, step_size,
        t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40, fix_noise = False):
    '''
    ks - number of corrector steps per each predictor step
    '''
    import torch.nn.functional as F

    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x

    @torch.enable_grad()
    def get_clf_grad(x):
        x.requires_grad_()
        logits = clf(normalize(x))
        log_probs = F.log_softmax(logits, dim = 0)
        target_class_log_probs = log_probs[:, target_class_id].unsqueeze(1)
        grad = torch.autograd.grad(target_class_log_probs.sum(), x)[0]
        return grad

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    xs = []

    if fix_noise:
        noise = x.clone()

    for i, k in zip(range(len(ts)), ks):

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
        xs.append(x0)

        # optionally move along clf gradient
        if grad_at_p:
            x0 += step_size * get_clf_grad(x0)

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
                x0 += step_size * get_clf_grad(x0)

    return torch.cat(xs, 0)