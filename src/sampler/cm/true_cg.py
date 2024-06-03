import math
import torch
import numpy as np

def true_cg(
        distiller, x, sigmas, ts, clf, target_class_id, step_size,
        scheduler_type, t_min = 0.002, t_max = 80.0, rho = 7.0, steps = 40):
    '''
    ks - number of corrector steps per each predictor step
    '''
    import torch.nn.functional as F

    def normalize(x):
        x = x - x.min()
        x = x / x.max()
        return x

    def get_clf_grad(x0_pre, noise):
        logits = clf(normalize(x0_pre))
        log_probs = F.log_softmax(logits, dim = -1)
        target_class_log_probs = log_probs[:, target_class_id].unsqueeze(1)
        grad = torch.autograd.grad(target_class_log_probs.sum(), noise)[0]
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
    x0 = x
        
    total_step = len(ts)
    step = 0
    scheduler = step_size_scheduler(step_size, total_step, scheduler_type)

    for i in range(len(ts)):

        # sample random noise
        noise = torch.randn_like(x, requires_grad = True)

        # get variance for ith iteration
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        t = np.clip(t, t_min, t_max)

        # add properly scaled noise
        x_t = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)

        # denoise and save pre movement
        x0_pre = distiller(x_t, t * s_in)
        xs.append(x0_pre)

        # move along clf gradient in noise
        clf_grad = get_clf_grad(x0_pre, noise)
        step_size_ = scheduler(step)
        step += 1
        noise = noise + step_size_ * clf_grad

        # denoise and save post movement
        x_t = x0 + noise * np.sqrt(t ** 2 - t_min ** 2)
        x0 = distiller(x_t, t * s_in)
        xs.append(x0)

    return torch.cat(xs, 0)