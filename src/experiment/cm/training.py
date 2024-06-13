import copy
import wandb
import torch
from torch import nn
import functools
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision.utils import make_grid
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import utils
from model.cm.src import logger
from model.cm.src.resample import LossAwareSampler

import logging
log = logging.getLogger(__name__)


def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric


def get_modules(config, fabric):
    # init
    model = instantiate(config.model)
    model_optimizer = instantiate(config.model_optimizer)(params = model.parameters())

    target_model = instantiate(config.target_model)

    # set training mode
    model.train()
    target_model.train()

    #
    target_model.requires_grad_(False) # 290@train_util


    # convert to fp16 NOTE: rather leave it to fabric
    

    # load checkpoints NOTE: could be done after fabric.setup

    # NOTE: here optimizers must also be loaded


    # TODO: load ema parameters: line 86 @ train_util


    # TODO: make sure that we set the correct step, resume_step, global_step


    # TODO: logic from MixedPrecisionTrainer

    # move to fabric
    model, model_optimizer = fabric.setup(model, model_optimizer)
    target_model = fabric.setup(target_model)

    return model, target_model, model_optimizer


def get_objects(config):
    diffusion = instantiate(config.diffusion)
    ema_scale_fn = instantiate(config.ema_scale_fn)
    schedule_sampler = instantiate(config.schedule_sampler)
    
    return diffusion, ema_scale_fn, schedule_sampler

def get_dataloader(config, fabric):
    return fabric.setup_dataloaders(instantiate(config.dataset))


class Trainer(nn.Module):

    def __init__(self,
            config, 
            fabric,
            model, 
            model_optimizer, 
            target_model, 
            dataloader, 
            diffusion,
            ema_scale_fn, 
            schedule_sampler, 
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16,
            fp16_scale_growth,
            lr_anneal_steps,
            weight_decay,
            lr,
            microbatch,
            training_mode,
            step,
            resume_step,
            global_step, 
            lg_loss_scale):
        
        super().__init__()

        logger.configure()

        # 30@train_util
        self.ema_rate = ema_rate
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.lr_anneal_steps = lr_anneal_steps
        self.weight_decay = weight_decay
        self.lr = lr
        self.microbatch = microbatch

        #
        self.step = step
        self.resume_step = resume_step
        self.global_step = global_step
        self.training_mode = training_mode
        self.lg_loss_scale = lg_loss_scale

        # 
        self.config = config
        self.fabric = fabric
        self.model = model
        self.model_optimizer = model_optimizer
        self.target_model = target_model
        self.dataloader = dataloader
        self.diffusion = diffusion
        self.ema_scale_fn = ema_scale_fn
        self.schedule_sampler = schedule_sampler

        # temporarily hardcoded since we only want consistency training for now
        self.teacher_model = None
        self.teacher_diffusion = None

        #
        self.ema_rate = ema_rate
        self.global_batch = self.dataloader.batch_size * fabric.world_size

        #
        self.master_params = list(self.model.parameters())

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(list(self.model.parameters()))
                for _ in range(len(self.ema_rate))
            ]

        self.step = self.resume_step

        self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )


    def run(self):
        saved = False
        # while (
        #     not self.lr_anneal_steps
        #     or self.step < self.lr_anneal_steps
        #     or self.global_step < self.total_training_steps
        # ):
        for batch, cond in self.dataloader:
            
            if not (
                not self.lr_anneal_steps
                or self.step < self.lr_anneal_steps
                or self.global_step < self.total_training_steps):
                break

            # NOTE: for some reason fabric does not move batch
            #       to fp16 so we do that manually
            if batch.dtype != self.model.dtype:
                batch = batch.to(self.model.dtype)

            self.run_step(batch, cond)

            saved = False
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                torch.cuda.empty_cache()

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.optimize()

        if took_step:
            self._update_ema()
            if self.target_model:
                self._update_target_ema()
            if self.training_mode == "progdist":
                self.reset_training_for_progdist()
            self.step += 1
            self.global_step += 1

        self._anneal_lr()
        self.log_step()


    def forward_backward(self, batch, cond):
        self.model_optimizer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch]
            micro_cond = {
                k: v[i : i + self.microbatch]
                for k, v in cond.items()
            }
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0])

            ema, num_scales = self.ema_scale_fn(self.global_step)
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.model,
                        micro,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.model,
                        micro,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=micro_cond,
                    )
            elif self.training_mode == "consistency_distillation":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    model_kwargs=micro_cond,
                )
            elif self.training_mode == "consistency_training":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    model_kwargs=micro_cond,
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            losses = compute_losses()

            # if last_batch or not self.use_ddp:
            #     losses = compute_losses()
            # else:
            #     with self.ddp_model.no_sync():
            #         losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            self.log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # self.mp_trainer.backward(loss)

            if self.use_fp16:
                loss_scale = 2**self.lg_loss_scale
                self.fabric.backward(loss * loss_scale)
            else:
                self.fabric.backward(loss)


    def optimize(self):
        if self.use_fp16:
            return self._optimize_fp16()
        else:
            return self._optimize_normal()


    def _optimize_fp16(self):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        # model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            # zero_master_grads(self.master_params)
            self.model_optimizer.zero_grad()
            return False

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        # for p in self.master_params:
        for p in self.model.parameters():
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))
        # opt.step()
        self.model_optimizer.step()
        self.model_optimizer.zero_grad()
        # zero_master_grads(self.master_params)
        # master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True


    def _optimize_normal(self):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        self.model_optimizer.step()
        # opt.step()
        return True


    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.model.parameters():
            with torch.no_grad():
                param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
        return (grad_norm ** (1 / 2)) / grad_scale, param_norm ** (1 / 2)


    def save(self):
        import blobfile as bf

        step = self.global_step

        def save_checkpoint(rate, params):
            state_dict = self.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            if self.fabric.global_rank == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{step:06d}.pt"
                else:
                    filename = f"ema_{rate}_{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        logger.log("saving optimizer state...")
        if self.fabric.global_rank == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{step:06d}.pt"),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        if self.fabric.global_rank == 0:
            if self.target_model:
                logger.log("saving target model state")
                filename = f"target_model{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(self.target_model.state_dict(), f)
            if self.teacher_model and self.training_mode == "progdist":
                logger.log("saving teacher model state")
                filename = f"teacher_model{step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(self.teacher_model.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        # save_checkpoint(0, self.mp_trainer.master_params)
        save_checkpoint(0, self.model.parameters())
        # dist.barrier()
        self.fabric.barrier()


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.model_optimizer.param_groups:
            param_group["lr"] = lr


    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)


    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with torch.no_grad():
            # update_ema(
            #     self.target_model_master_params,
            #     self.mp_trainer.master_params,
            #     rate=target_ema,
            # )
            update_ema(
                self.target_model.parameters(),
                self.model.parameters(),
                rate=target_ema,
            )
            # master_params_to_model_params(
            #     self.target_model_param_groups_and_shapes,
            #     self.target_model_master_params,
            # )


    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )


    def log_step(self):
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)


    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            logger.logkv_mean(key, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            import pdb; pdb.set_trace()
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def run(config: DictConfig):
    utils.preprocess_config(config)
    utils.setup_wandb(config)

    # make fabric
    fabric = get_fabric(config)

    # automatically move each created tensor to proper device
    with fabric.init_tensor():

        # setup modules, dataloader and trainer
        model, target_model, model_optimizer = get_modules(config, fabric)

        dataloader = get_dataloader(config, fabric)

        # setup trainer by instantiating it and packing with fabric
        trainer = instantiate(config.trainer)(
                            config = config,
                            fabric = fabric, 
                            model = model, 
                            model_optimizer = model_optimizer,
                            target_model = target_model, 
                            dataloader = dataloader)

        trainer = fabric.setup(trainer)
        
        # start training
        trainer.run()
