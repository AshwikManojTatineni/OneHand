"""Training-free self-model readout (plan: "Predictive error"), route B.

The frozen Franka TD-MPC2 world model predicts its next latent from the current observation and its own action:
z_hat = next(encode(o_t), a_t), compared with encode(o_{t+1}). Run on trajectories recorded on each body (seen
through the same interface), the ratio error(body) / error(Franka home) is the self-prediction ratio: 1.0 means
the model knows the new body as well as its own. relative_error divides by a no-change baseline
(predict z_{t+1} = z_t), so bodies that barely move are not rewarded. Nothing is fitted.

Usage: python scripts/selfmodel_readout_route_b.py name=path.npz [name=path.npz ...]
"""
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xbody.newt_agent import MAX_ACTION_DIM, MAX_OBS_DIM, FrozenNewtAgent  # noqa: E402


@torch.no_grad()
def one_step_errors(agent, obs_eps, act_eps):
    m = agent.model
    task = torch.zeros(1, dtype=torch.long)
    errs, cos, still = [], [], []
    for obs, act in zip(obs_eps, act_eps):
        obs, act = np.asarray(obs, np.float32), np.asarray(act, np.float32)
        if len(obs) < 2:
            continue
        o = np.zeros((len(obs), MAX_OBS_DIM), np.float32); o[:, : obs.shape[1]] = obs
        a = np.zeros((len(act), MAX_ACTION_DIM), np.float32); a[:, : act.shape[1]] = act
        z = m.encode(torch.from_numpy(o), task.expand(len(o)))
        z_hat = m.next(z[:-1], torch.from_numpy(a[:-1]), task.expand(len(o) - 1))
        z_true = z[1:]
        errs.extend(((z_hat - z_true) ** 2).mean(-1).tolist())
        still.extend(((z[:-1] - z_true) ** 2).mean(-1).tolist())   # no-change baseline
        cos.extend(torch.nn.functional.cosine_similarity(z_hat, z_true, dim=-1).tolist())
    return np.array(errs), np.array(cos), np.array(still)


def main(pairs):
    agent = FrozenNewtAgent(ROOT / "checkpoints/newt/ms-pick-cube.pt", "ms-pick-cube", 4, 25, mpc=False)
    out, base = {}, None
    for name, path in pairs:
        d = np.load(path, allow_pickle=True)
        e, c, s0 = one_step_errors(agent, d["obs"], d["act"])
        out[name] = {"transitions": int(len(e)), "latent_mse_mean": float(e.mean()), "latent_mse_median": float(np.median(e)),
                     "cosine_mean": float(c.mean()), "no_change_mse_mean": float(s0.mean()),
                     "relative_error": float(e.mean() / s0.mean())}
        if base is None:
            base = out[name]["latent_mse_mean"]
        out[name]["self_prediction_ratio"] = out[name]["latent_mse_mean"] / base
        print(name, json.dumps(out[name]))
    (ROOT / "results/phase0/selfmodel_readout_route_b.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main([a.split("=", 1) for a in sys.argv[1:]])
