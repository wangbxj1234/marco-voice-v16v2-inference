# 

import torch
import torch.nn.functional as F
import os, sys

sys.path.append(os.path.abspath('../../third_party/Matcha-TTS'))

from matcha.models.components.flow_matching import BASECFM

class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=384, estimator: torch.nn.Module = None):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,  # make sure  spk_emb_dim=384
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        self.estimator = estimator  # 

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None,
                prompt_len=0, flow_cache=torch.zeros(1, 80, 0, 2),
                encoder_hidden_states=None):
        """Forward diffusion"""
        z = torch.randn_like(mu) * temperature
        cache_size = flow_cache.shape[2]
        if cache_size != 0:
            z[:, :, :cache_size] = flow_cache[:, :, :, 0]
            mu[:, :, :cache_size] = flow_cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        flow_cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond,
                                encoder_hidden_states=encoder_hidden_states), flow_cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, encoder_hidden_states=None):
        """Fixed Euler solver for ODEs."""
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.forward_estimator(x, mask, mu, t, spks, cond, encoder_hidden_states)
            if self.inference_cfg_rate > 0:
                cfg_dphi_dt = self.forward_estimator(
                    x, mask, torch.zeros_like(mu), t,
                    torch.zeros_like(spks) if spks is not None else None,
                    torch.zeros_like(cond),
                    torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None,
                )
                dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt -
                           self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def forward_estimator(self, x, mask, mu, t, spks, cond, encoder_hidden_states=None):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator.forward(x, mask, mu, t, spks, cond)
        else:
            ort_inputs = {
                'x': x.cpu().numpy(),
                'mask': mask.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                't': t.cpu().numpy(),
                'spks': spks.cpu().numpy() if spks is not None else None,
                'cond': cond.cpu().numpy() if cond is not None else None
            }
            output = self.estimator.run(None, ort_inputs)[0]
            return torch.tensor(output, dtype=x.dtype, device=x.device)

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, encoder_hidden_states=None):
        """Computes diffusion loss"""
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1) if spks is not None else None
            cond = cond * cfg_mask.view(-1, 1, 1) if cond is not None else None
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y
