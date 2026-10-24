import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from pettingzoo.sisl import waterworld_v4

import os
import csv
from datetime import datetime

# -------------------------
# utils
# -------------------------
class CSVLogger:
    def __init__(self, csv_path: str, obs_dim: int, mode: str = "full"):
        assert mode in ("full", "summary")
        self.csv_path = csv_path
        self.obs_dim = obs_dim
        self.mode = mode

        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

        # Définir header
        base_fields = [
            "timestamp",
            "run_name",
            "seed",
            "it",
            "train_ep_avg",
            "rollout_avg",
            "rollout_min",
            "rollout_max",
            "entropy",
            "ent_coef",
            "kl",
            "actorL",
            "criticL",
            "finished",
            "eval_avg",
        ]

        if self.mode == "full":
            mean_fields = [f"obs_mean_{i}" for i in range(obs_dim)]
            std_fields  = [f"obs_std_{i}" for i in range(obs_dim)]
            self.fields = base_fields + mean_fields + std_fields
        else:
            self.fields = base_fields + [
                "obs_mean_global_mean",
                "obs_mean_global_min",
                "obs_mean_global_max",
                "obs_std_global_mean",
                "obs_std_global_min",
                "obs_std_global_max",
            ]

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fields)
                w.writeheader()

    def log(self, row: dict):
        safe_row = {k: row.get(k, "") for k in self.fields}
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            w.writerow(safe_row)

          
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_pursuers(env) -> List[str]:
    return [a for a in env.agents if "pursuer" in a]


def stack_obs(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.stack([obs_dict[a] for a in agents], axis=0)  


def unstack_actions(actions: np.ndarray, agents: List[str]) -> Dict[str, np.ndarray]:
    return {a: actions[i] for i, a in enumerate(agents)}


def stack_rewards_dones(
    rew_dict: Dict[str, float],
    term_dict: Dict[str, bool],
    trunc_dict: Dict[str, bool],
    agents: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    r = np.array([rew_dict[a] for a in agents], dtype=np.float32)
    d = np.array([float(term_dict[a] or trunc_dict[a]) for a in agents], dtype=np.float32)
    return r, d


# -------------------------
# Running mean/std
# -------------------------
class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        # x: [N, obs_dim]
        x = x.astype(np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta**2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip=10.0):
        x = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(x, -clip, clip).astype(np.float32)


# -------------------------
# Networks
# -------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_t: torch.Tensor):
        x = self.net(obs_t)
        mean = self.mean(x)
        std = torch.exp(self.logstd).unsqueeze(0).expand_as(mean)
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs_t: torch.Tensor):
        return self.net(obs_t).squeeze(-1)


# -------------------------
#  Gaussian 
# -------------------------
def atanh(x: torch.Tensor):
    # stable inverse tanh
    x = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def squash_gaussian_sample(mean, std):
    dist = Normal(mean, std)
    u = dist.rsample()
    a = torch.tanh(u)
    # logprob with tanh correction
    logp_u = dist.log_prob(u).sum(-1)
    logp_corr = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    logp = logp_u - logp_corr
    ent = dist.entropy().sum(-1)  # proxy
    return a, logp, ent


def squash_gaussian_logprob(mean, std, a):
    # reconstruct u = atanh(a)
    dist = Normal(mean, std)
    u = atanh(a)
    logp_u = dist.log_prob(u).sum(-1)
    logp_corr = torch.log(1 - a.pow(2) + 1e-6).sum(-1)
    return logp_u - logp_corr


# -------------------------
# Vec of PettingZoo parallel envs
# -------------------------
class VecPZ:
    def __init__(self, num_envs: int, seed: int, n_pursuers=5, n_evaders=5, max_cycles=500, render_mode=None):
        self.envs = []
        for i in range(num_envs):
            env = waterworld_v4.parallel_env(
                n_pursuers=n_pursuers,
                n_evaders=n_evaders,
                max_cycles=max_cycles,
                render_mode=render_mode,
            )
            self.envs.append(env)

        self.num_envs = num_envs
        self.seed = seed
        obs0, _ = self.envs[0].reset(seed=seed)
        self.agents = get_pursuers(self.envs[0])
        assert len(self.agents) == n_pursuers

        self.obs_dim = self.envs[0].observation_space(self.agents[0]).shape[0]
        self.act_dim = self.envs[0].action_space(self.agents[0]).shape[0]

        self.obs = []
        self.ep_return = np.zeros((num_envs, len(self.agents)), dtype=np.float32)
        self.ep_done = np.zeros((num_envs, len(self.agents)), dtype=np.float32)
        self.ep_finished = 0 
        self.reset_all()

    def reset_all(self):
        self.obs = []
        self.ep_return[:] = 0.0
        self.ep_done[:] = 0.0
        self.ep_finished = 0
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=self.seed + i)
            self.obs.append(o)

    def step(self, actions_env: List[Dict[str, np.ndarray]]):
        next_obs = []
        rewards_list, dones_list = [], []
        finished_envs = 0

        for i, env in enumerate(self.envs):
            o2, r, term, trunc, info = env.step(actions_env[i])
            rp, dp = stack_rewards_dones(r, term, trunc, self.agents)
            self.ep_return[i] += rp
            self.ep_done[i] = np.maximum(self.ep_done[i], dp)

            if all((term[a] or trunc[a]) for a in self.agents):
                finished_envs += 1
                o2, _ = env.reset(seed=self.seed + 10000 + i + int(time.time()) % 1000)

            next_obs.append(o2)
            rewards_list.append(rp)
            dones_list.append(dp)

        self.obs = next_obs
        self.ep_finished += finished_envs
        return rewards_list, dones_list

    def close(self):
        for e in self.envs:
            e.close()


# -------------------------
# PPO config
# -------------------------
@dataclass
class PPOConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    num_envs: int = 8
    n_pursuers: int = 5
    n_evaders: int = 5
    max_cycles: int = 500

    T: int = 512
    gamma: float = 0.99
    lam: float = 0.95

    num_updates: int = 500
    epochs: int = 4
    minibatch_size: int = 512

    actor_lr: float = 2e-4
    critic_lr: float = 2e-5

    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.01
    ent_coef_min: float = 0.001

    target_kl: float = 0.015
    max_grad_norm: float = 1.0

    obs_rms_warmup_steps: int = 5000
    obs_clip: float = 10.0

    eval_every: int = 100
    eval_episodes: int = 5


# -------------------------
# Rollout + training
# -------------------------
def linear_ent_coef(cfg: PPOConfig, it: int):
    frac = 1.0 - (it / cfg.num_updates)
    return max(cfg.ent_coef_min, cfg.ent_coef_start * frac)


@torch.no_grad()
def evaluate(actor: Actor, obs_rms: RunningMeanStd, cfg: PPOConfig):
    device = torch.device(cfg.device)  # <-- ajoute ça
    env = waterworld_v4.parallel_env(
        n_pursuers=cfg.n_pursuers,
        n_evaders=cfg.n_evaders,
        max_cycles=cfg.max_cycles,
    )
    obs, _ = env.reset(seed=cfg.seed + 999)
    agents = get_pursuers(env)

    ep_returns = []
    for ep in range(cfg.eval_episodes):
        done_all = False
        ret = np.zeros((len(agents),), dtype=np.float32)
        while not done_all:
            O = stack_obs(obs, agents)
            O = obs_rms.normalize(O, clip=cfg.obs_clip)
            Ot = torch.tensor(O, dtype=torch.float32, device=device)  # <-- ici

            mean, std = actor(Ot)
            a = torch.tanh(mean)
            acts = unstack_actions(a.cpu().numpy(), agents)

            obs2, rewards, term, trunc, info = env.step(acts)
            rp, dp = stack_rewards_dones(rewards, term, trunc, agents)
            ret += rp
            obs = obs2
            done_all = all((term[a] or trunc[a]) for a in agents)

        ep_returns.append(ret.mean().item())
        obs, _ = env.reset(seed=cfg.seed + 999 + ep + 1)

    env.close()
    return float(np.mean(ep_returns))

def obs_stats_to_row(obs_rms: RunningMeanStd, mode: str, obs_dim: int) -> dict:
    mean = obs_rms.mean.astype(np.float64)
    std = np.sqrt(obs_rms.var.astype(np.float64) + 1e-8)

    if mode == "full":
        out = {}
        for i in range(obs_dim):
            out[f"obs_mean_{i}"] = float(mean[i])
        for i in range(obs_dim):
            out[f"obs_std_{i}"] = float(std[i])
        return out
    else:
        return {
            "obs_mean_global_mean": float(mean.mean()),
            "obs_mean_global_min": float(mean.min()),
            "obs_mean_global_max": float(mean.max()),
            "obs_std_global_mean": float(std.mean()),
            "obs_std_global_min": float(std.min()),
            "obs_std_global_max": float(std.max()),
        }
     

def train(
    cfg: PPOConfig,
    save_path: str,
    csv_path: str = "logs_ppo_waterworld_mono.csv",
    run_name: str = "single_agent_patchB",
    obs_log_mode: str = "full",  # "full" ou "summary"
    print_every: int = 20,    
):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    vec = VecPZ(
        num_envs=cfg.num_envs,
        seed=cfg.seed,
        n_pursuers=cfg.n_pursuers,
        n_evaders=cfg.n_evaders,
        max_cycles=cfg.max_cycles,
    )
    agents = vec.agents
    obs_dim, act_dim = vec.obs_dim, vec.act_dim
    print(f"Trainable agents: {agents}")
    print(f"obs_dim: {obs_dim} act_dim: {act_dim}")
    print(f"Patch B: num_envs={cfg.num_envs}, T={cfg.T}, epochs={cfg.epochs}, minibatch={cfg.minibatch_size}")
    print(f"Warmup obs RMS for {cfg.obs_rms_warmup_steps} steps (random policy), then freeze.")

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    obs_rms = RunningMeanStd(shape=(obs_dim,))

    logger = CSVLogger(csv_path=csv_path, obs_dim=obs_dim, mode=obs_log_mode)

    warm_steps = cfg.obs_rms_warmup_steps
    warm_done = 0
    while warm_done < warm_steps:
        actions_env = []
        for e in range(cfg.num_envs):
            a = np.random.uniform(-1, 1, size=(len(agents), act_dim)).astype(np.float32)
            actions_env.append(unstack_actions(a, agents))

            Oe = stack_obs(vec.obs[e], agents)
            obs_rms.update(Oe.reshape(-1, obs_dim))

        vec.step(actions_env)
        warm_done += cfg.num_envs * len(agents)

    E = cfg.num_envs
    B = len(agents)

    for it in range(1, cfg.num_updates + 1):
        ent_coef = linear_ent_coef(cfg, it)

        obs_buf = np.zeros((cfg.T, E, B, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.T, E, B, act_dim), dtype=np.float32)
        logp_old_buf = np.zeros((cfg.T, E, B), dtype=np.float32)
        val_old_buf = np.zeros((cfg.T, E, B), dtype=np.float32)
        rew_buf = np.zeros((cfg.T, E, B), dtype=np.float32)
        done_buf = np.zeros((cfg.T, E, B), dtype=np.float32)

        rollout_returns = np.zeros((E, B), dtype=np.float32)
        start_finished = vec.ep_finished

        # ---------- rollout ----------
        for t in range(cfg.T):
            O = np.stack([stack_obs(vec.obs[e], agents) for e in range(E)], axis=0)
            O = obs_rms.normalize(O.reshape(-1, obs_dim), clip=cfg.obs_clip).reshape(E, B, obs_dim)

            Ot = torch.tensor(O, dtype=torch.float32, device=device).reshape(E * B, obs_dim)
            with torch.no_grad():
                mean, std = actor(Ot)
                a, logp, ent = squash_gaussian_sample(mean, std)
                v = critic(Ot)

            a_np = a.cpu().numpy().reshape(E, B, act_dim)
            logp_np = logp.cpu().numpy().reshape(E, B)
            v_np = v.cpu().numpy().reshape(E, B)

            actions_env = []
            for e in range(E):
                actions_env.append(unstack_actions(a_np[e], agents))
            rewards_list, dones_list = vec.step(actions_env)

            r = np.stack(rewards_list, axis=0)  
            d = np.stack(dones_list, axis=0)    
            rollout_returns += r

            obs_buf[t] = O
            act_buf[t] = a_np
            logp_old_buf[t] = logp_np
            val_old_buf[t] = v_np
            rew_buf[t] = r
            done_buf[t] = d

        finished = vec.ep_finished - start_finished

        O_last = np.stack([stack_obs(vec.obs[e], agents) for e in range(E)], axis=0)
        O_last = obs_rms.normalize(O_last.reshape(-1, obs_dim), clip=cfg.obs_clip).reshape(E, B, obs_dim)
        O_last_t = torch.tensor(O_last, dtype=torch.float32, device=device).reshape(E * B, obs_dim)
        with torch.no_grad():
            v_last = critic(O_last_t).cpu().numpy().reshape(E, B)

        adv = np.zeros((cfg.T, E, B), dtype=np.float32)
        gae = np.zeros((E, B), dtype=np.float32)
        v_next = v_last

        for t in reversed(range(cfg.T)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * not_done * v_next - val_old_buf[t]
            gae = delta + cfg.gamma * cfg.lam * not_done * gae
            adv[t] = gae
            v_next = val_old_buf[t]

        ret = adv + val_old_buf

        obs_flat = torch.tensor(obs_buf.reshape(-1, obs_dim), dtype=torch.float32, device=device)
        act_flat = torch.tensor(act_buf.reshape(-1, act_dim), dtype=torch.float32, device=device)
        logp_old_flat = torch.tensor(logp_old_buf.reshape(-1), dtype=torch.float32, device=device)
        v_old_flat = torch.tensor(val_old_buf.reshape(-1), dtype=torch.float32, device=device)
        adv_flat = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        ret_flat = torch.tensor(ret.reshape(-1), dtype=torch.float32, device=device)

        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        idx = torch.arange(obs_flat.shape[0], device=device)
        last_kl = 0.0
        last_ent = 0.0
        last_actorL = 0.0
        last_criticL = 0.0

        stop_early = False
        for ep in range(cfg.epochs):
            perm = idx[torch.randperm(idx.shape[0])]
            for start in range(0, perm.shape[0], cfg.minibatch_size):
                mb = perm[start:start + cfg.minibatch_size]

                O_mb = obs_flat[mb]
                A_mb = act_flat[mb]
                logp_old_mb = logp_old_flat[mb]
                adv_mb = adv_flat[mb]
                ret_mb = ret_flat[mb]

                mean, std = actor(O_mb)
                logp = squash_gaussian_logprob(mean, std, A_mb)
                ratio = torch.exp(logp - logp_old_mb)

                # clipped policy loss
                unclipped = ratio * adv_mb
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
                policy_loss = -torch.min(unclipped, clipped).mean()

                # value loss (simple MSE)
                v = critic(O_mb)
                value_loss = 0.5 * (ret_mb - v).pow(2).mean()

                # entropy (proxy)
                dist = Normal(mean, std)
                ent = dist.entropy().sum(-1).mean()

                loss_actor = policy_loss - ent_coef * ent
                loss_critic = value_loss * cfg.vf_coef

                # actor step
                opt_actor.zero_grad(set_to_none=True)
                loss_actor.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                opt_actor.step()

                # critic step
                opt_critic.zero_grad(set_to_none=True)
                loss_critic.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                opt_critic.step()

                # KL for early stop + logging
                with torch.no_grad():
                    approx_kl = (logp_old_mb - logp).mean().item()
                    last_kl = approx_kl
                    last_ent = ent.item()
                    last_actorL = policy_loss.item()
                    last_criticL = value_loss.item()

                if last_kl > cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        # ---------- Logging stats ----------
        rollout_avg = float(rollout_returns.mean())
        rollout_min = float(rollout_returns.min())
        rollout_max = float(rollout_returns.max())

        # estimation 
        train_ep_avg = float(np.mean(rollout_returns))


        if it % print_every == 0:
            eval_avg = ""
            if it % cfg.eval_every == 0:
                eval_avg = evaluate(actor, obs_rms, cfg)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_name": run_name,
                "seed": cfg.seed,
                "it": it,
                "train_ep_avg": train_ep_avg,
                "rollout_avg": rollout_avg,
                "rollout_min": rollout_min,
                "rollout_max": rollout_max,
                "entropy": last_ent,
                "ent_coef": ent_coef,
                "kl": last_kl,
                "actorL": last_actorL,
                "criticL": last_criticL,
                "finished": int(finished),
                "eval_avg": eval_avg,
            }
            row.update(obs_stats_to_row(obs_rms, mode=obs_log_mode, obs_dim=obs_dim))
            logger.log(row)

            msg = (
                f"it={it:4d}  "
                f"train_ep_avg={train_ep_avg:8.3f}  "
                f"rollout_avg={rollout_avg:8.3f}  "
                f"[min={rollout_min:8.3f}, max={rollout_max:8.3f}]  "
                f"ent={last_ent:6.3f}  ent_coef={ent_coef:6.4f}  "
                f"kl={last_kl:6.4f}  actorL={last_actorL:6.3f}  criticL={last_criticL:6.3f}  "
                f"finished={finished}"
            )
            if eval_avg != "":
                msg += f"  | eval_avg={float(eval_avg):8.3f}"
            print(msg)

    ckpt = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "obs_mean": obs_rms.mean.astype(np.float32),
        "obs_var": obs_rms.var.astype(np.float32),
        "cfg": cfg.__dict__,
    }
    torch.save(ckpt, save_path)
    print(f"Saved checkpoint to: {save_path}")

    vec.close()

# -------------------------
# Visualisation (play)
# -------------------------
@torch.no_grad()
def play(model_path: str, seed: int = 0, steps: int = 2000, fps: int = 60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(
    model_path,
    map_location=device,
    weights_only=False  
    )

    cfg_dict = ckpt.get("cfg", {})
    obs_dim = int(cfg_dict.get("obs_dim", 242)) if "obs_dim" in cfg_dict else None


    env = waterworld_v4.parallel_env(
        n_pursuers=5,
        n_evaders=5,
        max_cycles=500,
        render_mode="human",
    )
    obs, _ = env.reset(seed=seed)
    agents = get_pursuers(env)
    obs_dim = env.observation_space(agents[0]).shape[0]
    act_dim = env.action_space(agents[0]).shape[0]

    actor = Actor(obs_dim, act_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    obs_rms.mean = ckpt["obs_mean"].astype(np.float64)
    obs_rms.var = ckpt["obs_var"].astype(np.float64)

    print("Playing... (Ctrl+C to stop)")
    dt = 1.0 / fps

    for t in range(steps):
        O = stack_obs(obs, agents) 
        O = obs_rms.normalize(O, clip=10.0)
        Ot = torch.tensor(O, dtype=torch.float32, device=device)

        mean, std = actor(Ot)
        a = torch.tanh(mean) 
        actions = unstack_actions(a.cpu().numpy(), agents)

        obs2, rewards, term, trunc, infos = env.step(actions)

        try:
            env.render()
        except Exception:
            pass

        obs = obs2
        if all((term[a] or trunc[a]) for a in agents):
            obs, _ = env.reset(seed=seed + 1)

        time.sleep(dt)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train PPO Patch B")
    parser.add_argument("--play", action="store_true", help="Visualize a trained policy")
    parser.add_argument("--model", type=str, default="ppo_waterworld.pt", help="Path to save/load model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    
    parser.add_argument("--csv", type=str, default="logs_ppo_waterworld_mono.csv", help="CSV log file path")
    parser.add_argument("--run_name", type=str, default="single_agent_patchB", help="Run name for logging")
    parser.add_argument("--obs_log_mode", type=str, default="full", choices=["full", "summary"], help="Observation logging mode")
    
    args = parser.parse_args()

    cfg = PPOConfig(seed=args.seed)
    
    if args.train:
        train(cfg, save_path=args.model, csv_path=args.csv, run_name=args.run_name, obs_log_mode=args.obs_log_mode)

    if args.play:
        play(args.model, seed=args.seed, steps=args.steps)
if __name__ == "__main__":
    main()