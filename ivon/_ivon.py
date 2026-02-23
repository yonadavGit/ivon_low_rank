from math import pow
from typing import Callable, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.optim
import torch.distributed as dist
from torch import Tensor


ClosureType = Callable[[], Tensor]


def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(torch.optim.Optimizer):
    hessian_approx_methods = (
        'price',
        'gradsq',
    )

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = 'price',
        clip_radius: float = float("inf"),
        sync: bool = False,
        debias: bool = True,
        rescale_lr: bool = True
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError(
                "Invalid number of MC samples: {}".format(mc_samples)
            )
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError(
                "Invalid Hessian initialization: {}".format(hess_init)
            )
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        # set initial temporary running averages
        self._reset_samples()
        # init all states
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                "Parameters are on different devices: "
                f"{[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                "Parameters are on different dtypes: "
                f"{[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state['count'] = 0
        self.state['avg_grad'] = None
        self.state['avg_nxg'] = None
        self.state['avg_gsq'] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _restore_param_average(
        self, train: bool, param_avg: Tensor, noise: Tensor
    ):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel  # sanity check

        if train:  # collect grad sample for training
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], grad_sample, count
            )
            if self.hess_approx == 'price':
                self.state['avg_nxg'] = _welford_mean(
                    self.state['avg_nxg'], noise * grad_sample, count)
            elif self.hess_approx == 'gradsq':
                self.state['avg_gsq'] = _welford_mean(
                    self.state['avg_gsq'], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():  # explicit sync
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)
        dist.all_reduce(self.state["avg_nxg"])
        self.state["avg_nxg"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (
                    group["ess"] * (group["hess"] + group["weight_decay"])
                ).sqrt()
            )
            noise_samples.append(noise_sample)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]  # sanity check
        assert offset == self._numel  # sanity check

        return torch.cat(param_avgs, 0), torch.cat(noise_samples, 0)

    def _update(self):
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat(
                [p.flatten() for p in group["params"] if p is not None], 0
            )

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state['avg_gsq'],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"]) if self.rescale_lr else lr,
                group["weight_decay"],
                group["clip_radius"],
                1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0,
                group["hess_init"]
            )

            # update params
            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(
                        p.shape
                    )
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]  # sanity check
            offset += group["numel"]
        assert offset == self._numel  # sanity check

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == 'price':
            return avg_nxg[pg_slice] * hess
        elif method == 'gradsq':
            return avg_gsq[pg_slice]
        else:
            raise NotImplementedError(f'unknown hessian approx.: {method}')

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(
        method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd
    ) -> Tensor:
        f = IVON._get_nll_hess(
            method, hess + wd, avg_nxg, avg_gsq, pg_slice
        ) * ess
        return beta2 * hess + (1.0 - beta2) * f + \
            (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )


class IVONLR(torch.optim.Optimizer):
    # IVONLR differs from IVON by adding a low-rank covariance component.
    hessian_approx_methods = (
        'price',
        'gradsq',
    )

    def __init__(
            self,
            params,
            lr: float,
            ess: float,
            hess_init: float = 1.0,
            beta1: float = 0.9,
            beta2: float = 0.99999,
            weight_decay: float = 1e-4,
            mc_samples: int = 1,
            hess_approx: str = "price",
            clip_radius: float = float("inf"),
            sync: bool = False,
            debias: bool = True,
            rescale_lr: bool = True,
            rank: int = 8,
            beta3: float = 0.95,
            eta_u: float = 0.1,
            orth_every: int = 1,
            low_rank_init: float = 1e-6,
            # --- add these (they are referenced below) ---
            s_scale: float = 1.0,
            v_max: float = 1e3,
            alpha_min: float = 1e-8,
            alpha_max: float = 1e2,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 1 <= mc_samples:
            raise ValueError(
                "Invalid number of MC samples: {}".format(mc_samples)
            )
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight decay: {}".format(weight_decay))
        if not 0.0 < hess_init:
            raise ValueError(
                "Invalid Hessian initialization: {}".format(hess_init)
            )
        if not 0.0 < ess:
            raise ValueError("Invalid effective sample size: {}".format(ess))
        if not 0.0 < clip_radius:
            raise ValueError("Invalid clipping radius: {}".format(clip_radius))
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta1))
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta2))
        if not 0.0 <= beta3 <= 1.0:
            raise ValueError("Invalid beta3 parameter: {}".format(beta3))
        if not 0.0 < eta_u:
            raise ValueError("Invalid eta_u parameter: {}".format(eta_u))
        if not 0 <= orth_every:
            raise ValueError("Invalid orth_every parameter: {}".format(orth_every))
        if not 0.0 <= low_rank_init:
            raise ValueError(
                "Invalid low_rank_init parameter: {}".format(low_rank_init)
            )
        if not 0.0 < s_scale:
            raise ValueError("Invalid s_scale parameter: {}".format(s_scale))
        if not 0.0 < v_max:
            raise ValueError("Invalid v_max parameter: {}".format(v_max))
        if not 0.0 <= alpha_min:
            raise ValueError("Invalid alpha_min parameter: {}".format(alpha_min))
        if not alpha_min < alpha_max:
            raise ValueError("Invalid alpha_max parameter: {}".format(alpha_max))
        if rank < 0:
            raise ValueError("Invalid rank parameter: {}".format(rank))
        if hess_approx not in self.hessian_approx_methods:
            raise ValueError("Invalid hess_approx parameter: {}".format(beta2))

        defaults = dict(
            lr=lr,
            mc_samples=mc_samples,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            hess_init=hess_init,
            ess=ess,
            clip_radius=clip_radius,
            rank=rank,
            beta3=beta3,
            eta_u=eta_u,
            orth_every=orth_every,
            low_rank_init=low_rank_init,
            s_scale=s_scale,
            v_max=v_max,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        super().__init__(params, defaults)

        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.sync = sync
        self._numel, self._device, self._dtype = self._get_param_configs()
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr

        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self):
        all_params = []
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
            all_params += [p for p in pg["params"] if p is not None]
        if len(all_params) == 0:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        devices = {p.device for p in all_params}
        if len(devices) > 1:
            raise ValueError(
                "Parameters are on different devices: "
                f"{[str(d) for d in devices]}"
            )
        device = next(iter(devices))
        dtypes = {p.dtype for p in all_params}
        if len(dtypes) > 1:
            raise ValueError(
                "Parameters are on different dtypes: "
                f"{[str(d) for d in dtypes]}"
            )
        dtype = next(iter(dtypes))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None
        for group in self.param_groups:
            group["avg_gw"] = None
            group["avg_gw2"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            hess_init, numel = group["hess_init"], group["numel"]
            rank = min(group["rank"], numel)
            group["rank"] = rank

            group["momentum"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            )
            group["hess"] = torch.zeros(
                numel, device=self._device, dtype=self._dtype
            ).add(torch.as_tensor(hess_init))

            if rank == 0:
                group["U"] = torch.zeros(
                    numel, 0, device=self._device, dtype=self._dtype
                )
                group["s"] = torch.zeros(0, device=self._device, dtype=self._dtype)
                group["avg_gw"] = None
                group["avg_gw2"] = None
                continue

            init = torch.randn(
                numel, rank, device=self._device, dtype=self._dtype
            ) * 0.01
            group["U"] = self._orthonormalize(init)
            group["s"] = torch.full(
                (rank,),
                group["low_rank_init"],
                device=self._device,
                dtype=self._dtype,
            )
            group["avg_gw"] = None
            group["avg_gw2"] = None

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise, diag_noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise, diag_noise)

    def _restore_param_average(
        self, train: bool, param_avg: Tensor, noise: Tensor, diag_noise: Tensor
    ):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                p_slice = slice(offset, offset + p.numel())

                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    if p.requires_grad:
                        param_grads.append(p.grad.flatten())
                    else:
                        param_grads.append(torch.zeros_like(p).flatten())
                offset += p.numel()
        assert offset == self._numel

        if train:
            grad_sample = torch.cat(param_grads, 0)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(
                self.state["avg_grad"], grad_sample, count
            )
            if self.hess_approx == "price":
                # Use diagonal noise only to avoid feedback loop.
                self.state["avg_nxg"] = _welford_mean(
                    self.state["avg_nxg"], diag_noise * grad_sample, count
                )
            elif self.hess_approx == "gradsq":
                self.state["avg_gsq"] = _welford_mean(
                    self.state["avg_gsq"], grad_sample.square(), count
                )

            offset = 0
            for group in self.param_groups:
                gnumel = group["numel"]
                if group["rank"] > 0 and gnumel > 0:
                    g = grad_sample[offset : offset + gnumel]
                    h = (group["hess"] + group["weight_decay"]).clamp_min(1e-12)
                    g_w = g / h.sqrt()
                    group["avg_gw"] = _welford_mean(group["avg_gw"], g_w, count)
                    group["avg_gw2"] = _welford_mean(group["avg_gw2"], g_w.square(), count)
                    var_w = (group["avg_gw2"] - group["avg_gw"].square()).clamp_min(0.0)
                    alpha = (var_w.mean() * group["ess"]).clamp_min(group["alpha_min"]).clamp_max(group["alpha_max"])
                    self._update_low_rank_sample(group, g_w, alpha)
                offset += gnumel
            assert offset == self._numel

    @torch.no_grad()
    def step(self, closure: ClosureType = None) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        if self.sync and dist.is_initialized():
            self._sync_samples()
        self._update()
        self._reset_samples()
        return loss

    def _sync_samples(self):
        world_size = dist.get_world_size()
        dist.all_reduce(self.state["avg_grad"])
        self.state["avg_grad"].div_(world_size)

        if self.hess_approx == "price" and self.state["avg_nxg"] is not None:
            dist.all_reduce(self.state["avg_nxg"])
            self.state["avg_nxg"].div_(world_size)

        if self.hess_approx == "gradsq" and self.state["avg_gsq"] is not None:
            dist.all_reduce(self.state["avg_gsq"])
            self.state["avg_gsq"].div_(world_size)

        for group in self.param_groups:
            if group.get("avg_gw") is not None:
                dist.all_reduce(group["avg_gw"])
                group["avg_gw"].div_(world_size)
            if group.get("avg_gw2") is not None:
                dist.all_reduce(group["avg_gw2"])
                group["avg_gw2"].div_(world_size)

    def _sample_params(self) -> Tuple[Tensor, Tensor, Tensor]:
        noise_samples = []
        diag_noise_samples = []
        param_avgs = []

        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            h = (group["hess"] + group["weight_decay"]).clamp_min(1e-12)
            D = group["ess"] * h
            Dinv_sqrt = D.rsqrt()
            eps = torch.randn(gnumel, device=self._device, dtype=self._dtype)
            diag_noise = eps * Dinv_sqrt

            if group["rank"] > 0:
                U = group["U"]
                s = group["s"].clamp_min(0.0)
                shrink = 1.0 - (1.0 + s).rsqrt()
                y = eps - U @ (shrink * (U.transpose(0, 1) @ eps))
            else:
                y = eps

            noise_sample = Dinv_sqrt * y
            noise_samples.append(noise_sample)
            diag_noise_samples.append(diag_noise)

            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue

                p_avg = p.data.flatten()
                numel = p.numel()
                p_noise = noise_sample[goffset : goffset + numel]

                param_avgs.append(p_avg)
                p.data = (p_avg + p_noise).view(p.shape)
                goffset += numel
                offset += numel
            assert goffset == group["numel"]
        assert offset == self._numel

        return (
            torch.cat(param_avgs, 0),
            torch.cat(noise_samples, 0),
            torch.cat(diag_noise_samples, 0),
        )

    def _update(self):
        self.current_step += 1

        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat(
                [p.flatten() for p in group["params"] if p is not None], 0
            )

            group["momentum"] = self._new_momentum(
                self.state["avg_grad"][pg_slice], group["momentum"], b1
            )

            group["hess"] = self._new_hess(
                self.hess_approx,
                group["hess"],
                self.state["avg_nxg"],
                self.state["avg_gsq"],
                pg_slice,
                group["ess"],
                b2,
                group["weight_decay"],
            )

            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0
            param_avg = self._new_param_averages(
                param_avg,
                group["hess"],
                group["momentum"],
                lr * (group["hess_init"] + group["weight_decay"])
                if self.rescale_lr
                else lr,
                group["weight_decay"],
                group["clip_radius"],
                debias,
                group["hess_init"],
            )

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset : pg_offset + p.numel()].view(
                        p.shape
                    )
                    pg_offset += p.numel()
            assert pg_offset == group["numel"]

            self._update_low_rank(group, debias)

            offset += group["numel"]
        assert offset == self._numel

    @staticmethod
    def _get_nll_hess(method: str, hess, avg_nxg, avg_gsq, pg_slice) -> Tensor:
        if method == "price":
            return avg_nxg[pg_slice] * hess
        if method == "gradsq":
            return avg_gsq[pg_slice]
        raise NotImplementedError(f"unknown hessian approx.: {method}")

    @staticmethod
    def _new_momentum(avg_grad, m, b1) -> Tensor:
        return b1 * m + (1.0 - b1) * avg_grad

    @staticmethod
    def _new_hess(
        method, hess, avg_nxg, avg_gsq, pg_slice, ess, beta2, wd
    ) -> Tensor:
        f = (
            IVONLR._get_nll_hess(method, hess + wd, avg_nxg, avg_gsq, pg_slice)
            * ess
        )
        return (
            beta2 * hess
            + (1.0 - beta2) * f
            + (0.5 * (1 - beta2) ** 2) * (hess - f).square() / (hess + wd)
        )

    @staticmethod
    def _new_param_averages(
        param_avg, hess, momentum, lr, wd, clip_radius, debias, hess_init
    ) -> Tensor:
        return param_avg - lr * torch.clip(
            (momentum / debias + wd * param_avg) / (hess + wd),
            min=-clip_radius,
            max=clip_radius,
        )

    def _update_low_rank(self, group, debias):
        if group["rank"] == 0:
            return

        orth_every = group["orth_every"]
        if orth_every > 0 and self.current_step % orth_every == 0:
            group["U"] = self._orthonormalize(group["U"])

    @torch.no_grad()
    def _update_low_rank_sample(self, group, g_w: Tensor, alpha: Tensor):
        if group["rank"] == 0:
            return

        g_u = g_w / (g_w.norm() + 1e-12)

        U = group["U"]
        a_v = U.t() @ g_w
        a_u = U.t() @ g_u
        a_v2 = a_v.square()
        a_u2 = a_u.square()

        beta3 = group["beta3"]
        denom = a_v2.mean().clamp_min(1e-12)
        scaled = (a_v2 / denom) * alpha * group["s_scale"]
        v = beta3 * group["s"] + (1.0 - beta3) * scaled
        group["s"] = v.clamp_min(1e-12).clamp_max(group["v_max"])

        eta = group["eta_u"]
        U = U + eta * (g_u.unsqueeze(1) * a_u.unsqueeze(0) - U * a_u2.unsqueeze(0))
        orth_every = group["orth_every"]
        if orth_every > 0 and self.state["count"] % orth_every == 0:
            U = self._orthonormalize(U)
        group["U"] = U

    @staticmethod
    def _orthonormalize(U: Tensor) -> Tensor:
        if U.numel() == 0:
            return U
        return torch.linalg.qr(U, mode="reduced").Q
