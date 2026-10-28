"""Load a frozen Newt / TD-MPC2 single-task checkpoint on CPU for inference only (route B, stack S5).

Only `agent(obs, ...)` is ever called: no update(), no buffer, no optimizer step. Newt hard-codes CUDA, so this
loads third_party/newt_cpu: an unmodified copy of Newt except two device strings set to "cpu".
"""
import os
import pathlib
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEWT = ROOT / "third_party/newt_cpu/tdmpc2"
MAX_OBS_DIM, MAX_ACTION_DIM = 128, 16


def _import_cpu_tdmpc2():
    if str(NEWT) not in sys.path:
        sys.path.insert(0, str(NEWT))
    from tdmpc2 import TDMPC2  # noqa: E402
    return TDMPC2


def build_cfg(task, action_dim, episode_length):
    sys.path.insert(0, str(NEWT))
    from config import Config, parse_cfg  # noqa: E402  (Newt's own config code)
    cfg = OmegaConf.structured(Config)
    cfg.task, cfg.compile, cfg.num_envs, cfg.rank, cfg.world_size = task, False, 1, 0, 1
    cwd = os.getcwd()
    os.chdir(NEWT)
    try:
        import hydra.utils
        hydra.utils.get_original_cwd = lambda: str(NEWT)   # parse_cfg expects a hydra run
        cfg = OmegaConf.create(parse_cfg(cfg))
    finally:
        os.chdir(cwd)
    cfg.obs_shape = {"state": (MAX_OBS_DIM,)}
    cfg.action_dim = MAX_ACTION_DIM
    cfg.episode_length = episode_length
    cfg.action_dims = [action_dim] * cfg.num_envs
    return cfg


class FrozenNewtAgent:
    def __init__(self, ckpt, task, action_dim, episode_length=25, mpc=True):
        cfg = build_cfg(task, action_dim, episode_length)
        from common.world_model import WorldModel  # noqa: E402
        TDMPC2 = _import_cpu_tdmpc2()
        ckpt_sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        masks = ckpt_sd["_action_masks"]
        # The checkpoint stores one fixed action mask per training env slot; all rows are identical for a
        # single-task agent, so one slot is kept (a structural buffer, not a learned weight).
        assert torch.all(masks == masks[:1]), "action-mask rows differ; cannot reduce to one slot"
        n_masks = 1
        ckpt_sd["_action_masks"] = masks[:1].clone()
        cfg.num_envs = n_masks
        cfg.tasks = [task] * n_masks
        cfg.action_dims = [action_dim] * n_masks
        cfg.discounts = [cfg.discounts[0]] * n_masks
        self.model = WorldModel(cfg)
        self.agent = TDMPC2(self.model, cfg)
        from common.layers import api_model_conversion  # noqa: E402  (what TDMPC2.load does, minus the file read)
        self.model.load_state_dict(api_model_conversion(self.model.state_dict(), ckpt_sd))
        self.agent.eval()
        for p in self.agent.parameters():
            p.requires_grad_(False)              # no-training rule, enforced
        self.cfg, self.action_dim, self.mpc, self.n_slots = cfg, action_dim, mpc, n_masks
        self.task_ids = torch.zeros(n_masks, dtype=torch.long)

    @torch.no_grad()
    def act(self, obs, t0):
        o = np.zeros((self.n_slots, MAX_OBS_DIM), np.float32)
        o[:, : obs.shape[-1]] = obs              # same observation in every slot; slot 0's action is used
        t0v = torch.full((self.n_slots,), bool(t0))
        a = self.agent(torch.from_numpy(o), t0=t0v, eval_mode=True, task=self.task_ids, mpc=self.mpc)
        return a[0, : self.action_dim].cpu().numpy()
