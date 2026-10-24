import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing import Any
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt

from pettingzoo.sisl import waterworld_v4



def make_env(env_kwargs: dict, render_mode=None):
    return waterworld_v4.parallel_env(render_mode=render_mode, **env_kwargs)


def get_agent_list(env, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> List[str]:
    if hasattr(env, "possible_agents") and env.possible_agents is not None:
        return list(env.possible_agents)
    if hasattr(env, "agents") and getattr(env, "agents") is not None:
        return list(env.agents)
    if obs_dict is not None:
        return list(obs_dict.keys())
    raise AttributeError("Could not find agents list on env and no obs_dict provided.")


def get_trainable_agents(env, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> List[str]:
    agents = get_agent_list(env, obs_dict=obs_dict)
    pursuers = [a for a in agents if "pursuer" in a]
    return pursuers if len(pursuers) > 0 else agents


def stack_obs_np(obs_dict: Dict[str, np.ndarray], agents: List[str]) -> np.ndarray:
    return np.stack([obs_dict[a] for a in agents], axis=0)


def unstack_actions_np(actions: np.ndarray, agents: List[str]) -> Dict[str, np.ndarray]:
    return {a: actions[i] for i, a in enumerate(agents)}


def all_done_for_agents(term: Dict[str, bool], trunc: Dict[str, bool], agents: List[str]) -> bool:
    return all((term.get(a, False) or trunc.get(a, False)) for a in agents)


class VecParallel:
    
    def __init__(self, env_kwargs: dict, num_envs: int, seed: int):
        self.env_kwargs = env_kwargs
        self.num_envs = num_envs
        self.seed = seed

        self.envs = [make_env(env_kwargs, render_mode=None) for _ in range(num_envs)]
        self.obs_list: List[Dict[str, np.ndarray]] = []
        self.agents_by_env: List[List[str]] = []

        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=seed + i)
            agents = get_trainable_agents(env, obs_dict=obs)
            self.obs_list.append(obs)
            self.agents_by_env.append(agents)

        agents0 = self.agents_by_env[0]
        self.B = len(agents0)
        a0 = agents0[0]
        self.obs_dim = self.envs[0].observation_space(a0).shape[0]
        self.act_dim = self.envs[0].action_space(a0).shape[0]

    def close(self):
        for env in self.envs:
            env.close()

    def obs_batch_np(self) -> np.ndarray:
        return np.stack(
            [stack_obs_np(self.obs_list[ei], self.agents_by_env[ei]) for ei in range(self.num_envs)],
            axis=0
        )

    def step(self, a_batch_np: np.ndarray, reset_seed_offset: int) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
        
        rew_np = np.zeros((self.num_envs, self.B), dtype=np.float32)
        done_np = np.zeros((self.num_envs, self.B), dtype=np.float32)
        reset_flags = [False] * self.num_envs

        for ei, env in enumerate(self.envs):
            agents = self.agents_by_env[ei]
            act_dict = unstack_actions_np(a_batch_np[ei], agents)
            next_obs, rewards, term, trunc, _ = env.step(act_dict)

            rew_np[ei] = np.array([rewards[a] for a in agents], dtype=np.float32)
            done_np[ei] = np.array([float(term[a] or trunc[a]) for a in agents], dtype=np.float32)

            if all_done_for_agents(term, trunc, agents):
                reset_flags[ei] = True
                obs_reset, _ = env.reset(seed=self.seed + reset_seed_offset + ei)
                self.obs_list[ei] = obs_reset
            else:
                self.obs_list[ei] = next_obs

        return rew_np, done_np, reset_flags



class RunningMeanStd:

    def __init__(self, shape, device, epsilon=1e-4):
        self.device = device
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(epsilon, device=device, dtype=torch.float32)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.to(self.device, dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=self.device, dtype=torch.float32)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = torch.clamp(new_var, 1e-6, 1e6)
        self.count = tot_count

    @torch.no_grad()
    def normalize(self, x: torch.Tensor, clip: float = 10.0):
        x = x.to(self.device, dtype=torch.float32)
        y = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(y, -clip, clip)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd):
        self.mean = sd["mean"]
        self.var = sd["var"]
        self.count = sd["count"]


@torch.no_grad()
def warmup_obs_rms(vec: VecParallel, obs_rms: RunningMeanStd, steps: int, seed: int):
    print(f"Warmup obs RMS for {steps} steps (random policy), then freeze.")
    rng = np.random.RandomState(seed)

    for _ in range(steps):
        a = rng.uniform(low=-1.0, high=1.0, size=(vec.num_envs, vec.B, vec.act_dim)).astype(np.float32)
        vec.step(a, reset_seed_offset=1000)

        O_np = vec.obs_batch_np()
        O = torch.tensor(O_np.reshape(vec.num_envs * vec.B, vec.obs_dim),
                         device=obs_rms.device, dtype=torch.float32)
        obs_rms.update(O)


# PPO 
def tanh_gaussian_logprob(dist: Normal, u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    logp_u = dist.log_prob(u).sum(-1)
    corr = torch.log(1.0 - a.pow(2) + 1e-6).sum(-1)
    return logp_u - corr


def compute_gae(
    rew_buf: torch.Tensor, 
    done_buf: torch.Tensor, 
    val_buf: torch.Tensor, 
    v_last: torch.Tensor, 
    gamma: float,
    lam: float
):
    T, E, B = rew_buf.shape
    adv = torch.zeros((T, E, B), device=rew_buf.device, dtype=torch.float32)
    gae = torch.zeros((E, B), device=rew_buf.device, dtype=torch.float32)
    v_next = v_last

    for t in reversed(range(T)):
        not_done = 1.0 - done_buf[t]
        delta = rew_buf[t] + gamma * not_done * v_next - val_buf[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
        v_next = val_buf[t]

    ret = adv + val_buf
    return adv, ret



@torch.no_grad()
def evaluate_policy(cfg, env_kwargs: dict, actor: nn.Module, obs_rms: RunningMeanStd,
                    device: torch.device, seed: int) -> float:
    env = make_env(env_kwargs, render_mode=None)
    obs, _ = env.reset(seed=seed + 10_000)
    agents = get_trainable_agents(env, obs_dict=obs)

    scores = []
    for k in range(cfg.eval_episodes):
        if k > 0:
            obs, _ = env.reset(seed=seed + 10_000 + k)

        ep = torch.zeros((len(agents),), dtype=torch.float32, device=device)
        done_all = False
        steps = 0
        max_steps = env_kwargs.get("max_cycles", 500) + 5

        while not done_all and steps < max_steps:
            O = torch.tensor(stack_obs_np(obs, agents), dtype=torch.float32, device=device)
            O = obs_rms.normalize(O)
            mean, _ = actor(O)
            a = torch.tanh(mean).cpu().numpy().astype(np.float32)

            next_obs, rewards, term, trunc, _ = env.step(unstack_actions_np(a, agents))
            r = torch.tensor([rewards[a_] for a_ in agents], dtype=torch.float32, device=device)
            ep += r

            done_all = all_done_for_agents(term, trunc, agents)
            obs = next_obs
            steps += 1

        scores.append(ep.mean().item())

    env.close()
    return float(np.mean(scores)) if scores else float("nan")


def save_checkpoint(model_path: str, actor, critic, obs_rms, rew_rms, cfg,
                    env_kwargs: dict, seed: int, obs_dim: int, act_dim: int):
    ckpt = {
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "obs_rms": obs_rms.state_dict(),
        "rew_rms": rew_rms.state_dict(),
        "cfg": getattr(cfg, "__dict__", dict(cfg=cfg)),
        "env_kwargs": env_kwargs,
        "seed": seed,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }
    torch.save(ckpt, model_path)


@torch.no_grad()
def load_actor(model_path: str, actor_cls, rms_cls, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    actor = actor_cls(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    obs_rms = rms_cls(shape=(ckpt["obs_dim"],), device=device)
    obs_rms.load_state_dict(ckpt["obs_rms"])
    return actor, obs_rms, ckpt


@torch.no_grad()
def play_env(env_kwargs: dict, actor, obs_rms: RunningMeanStd, seed: int, steps: int = 3000):
    device = next(actor.parameters()).device
    env = make_env(env_kwargs, render_mode="human")
    obs, _ = env.reset(seed=seed)
    agents = get_trainable_agents(env, obs_dict=obs)

    for t in range(steps):
        O = torch.tensor(stack_obs_np(obs, agents), dtype=torch.float32, device=device)
        O = obs_rms.normalize(O)
        mean, _ = actor(O)
        a = torch.tanh(mean).cpu().numpy().astype(np.float32)

        next_obs, _, term, trunc, _ = env.step(unstack_actions_np(a, agents))
        if all_done_for_agents(term, trunc, agents):
            obs, _ = env.reset(seed=seed + 1234 + t)
        else:
            obs = next_obs

    env.close()


@torch.no_grad()
def record_video_env(env_kwargs: dict, actor, obs_rms: RunningMeanStd, out_mp4: str,
                     seed: int, steps: int = 3000, fps: int = 30):
    import imageio.v2 as imageio

    device = next(actor.parameters()).device
    env = make_env(env_kwargs, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agents = get_trainable_agents(env, obs_dict=obs)

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=fps)

    try:
        for t in range(steps):
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

            O = torch.tensor(stack_obs_np(obs, agents), dtype=torch.float32, device=device)
            O = obs_rms.normalize(O)
            mean, _ = actor(O)
            a = torch.tanh(mean).cpu().numpy().astype(np.float32)

            next_obs, _, term, trunc, _ = env.step(unstack_actions_np(a, agents))
            if all_done_for_agents(term, trunc, agents):
                obs, _ = env.reset(seed=seed + 999 + t)
            else:
                obs = next_obs

        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
    finally:
        writer.close()
        env.close()

    print(f"Saved video to: {out_mp4}")

def knn_neighbors_from_obs(obs_batch: np.ndarray, k: int) -> np.ndarray:
    B = obs_batch.shape[0]
    if B <= 1:
        return np.zeros((B, 0), dtype=np.int64)
    k = min(k, B - 1)
    d2 = ((obs_batch[None, :, :] - obs_batch[:, None, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.argsort(d2, axis=1)[:, :k]


def local_broadcast_mean(msgs: np.ndarray, neigh_idx: np.ndarray) -> np.ndarray:
    B, msg_dim = msgs.shape
    if neigh_idx.shape[1] == 0:
        return np.zeros((B, msg_dim), dtype=np.float32)
    out = np.zeros((B, msg_dim), dtype=np.float32)
    for i in range(B):
        out[i] = msgs[neigh_idx[i]].mean(axis=0)
    return out

@torch.no_grad()
def evaluate_recurrent(cfg, env_kwargs: dict, model,
                       obs_rms: RunningMeanStd, device: torch.device, seed: int) -> float:
    model.eval()
    env = make_env(env_kwargs, render_mode=None)
    obs, _ = env.reset(seed=seed + 10_000)
    agents = get_trainable_agents(env, obs_dict=obs)

    scores = []
    for k in range(cfg.eval_episodes):
        if k > 0:
            obs, _ = env.reset(seed=seed + 10_000 + k)

        B = len(agents)
        h = model.init_hidden(B, device)
        msg_prev = np.zeros((B, cfg.msg_dim), dtype=np.float32)

        ep = torch.zeros((B,), dtype=torch.float32, device=device)
        done_all = False
        steps = 0
        max_steps = env_kwargs.get("max_cycles", 500) + 5

        while not done_all and steps < max_steps:
            obs_np = stack_obs_np(obs, agents)  # (B,obs_dim)
            k_eff = min(cfg.k_neighbors, max(1, B - 1))
            neigh = knn_neighbors_from_obs(obs_np, k=k_eff)
            msg_in_np = local_broadcast_mean(msgs=msg_prev, neigh_idx=neigh)

            O = torch.tensor(obs_np, dtype=torch.float32, device=device)
            O = obs_rms.normalize(O)
            M = torch.tensor(msg_in_np, dtype=torch.float32, device=device)

            mean, std, msg_out, gate, h2 = model.forward_step(O, M, h)

            a = torch.tanh(mean).cpu().numpy().astype(np.float32)
            next_obs, rewards, term, trunc, _ = env.step(unstack_actions_np(a, agents))
            r = torch.tensor([rewards[a_] for a_ in agents], dtype=torch.float32, device=device)
            ep += r

            d = torch.tensor([float(term[a_] or trunc[a_]) for a_ in agents], dtype=torch.float32, device=device)
            h = h2 * (1.0 - d).unsqueeze(-1)
            msg_prev = msg_out.cpu().numpy().astype(np.float32)

            done_all = all_done_for_agents(term, trunc, agents)
            obs = next_obs
            steps += 1

        scores.append(ep.mean().item())

    env.close()
    return float(np.mean(scores)) if scores else float("nan")

@torch.no_grad()
def record_recurrent(cfg: Any, env_kwargs: dict, model: Any, obs_rms: RunningMeanStd,
                     out_mp4: str, seed: int, steps: int = 3000, fps: int = 30):
    import imageio.v2 as imageio

    device = next(model.parameters()).device
    env = make_env(env_kwargs, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agents = get_trainable_agents(env, obs_dict=obs)

    B = len(agents)
    h = model.init_hidden(B, device)
    msg_prev = np.zeros((B, cfg.msg_dim), dtype=np.float32)

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(out_mp4, fps=fps)

    try:
        for t in range(steps):
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)

            obs_np = stack_obs_np(obs, agents)
            k_eff = min(cfg.k_neighbors, max(1, B - 1))
            neigh = knn_neighbors_from_obs(obs_np, k=k_eff)
            msg_in_np = local_broadcast_mean(msg_prev, neigh)

            O = torch.tensor(obs_np, dtype=torch.float32, device=device)
            O = obs_rms.normalize(O)
            M = torch.tensor(msg_in_np, dtype=torch.float32, device=device)

            mean, std, msg_out, gate, h2 = model.forward_step(O, M, h)

            a = torch.tanh(mean).cpu().numpy().astype(np.float32)
            next_obs, _, term, trunc, _ = env.step(unstack_actions_np(a, agents))

            d = torch.tensor([float(term[a_] or trunc[a_]) for a_ in agents], dtype=torch.float32, device=device)
            h = h2 * (1.0 - d).unsqueeze(-1)
            msg_prev = msg_out.cpu().numpy().astype(np.float32)

            if all_done_for_agents(term, trunc, agents):
                obs, _ = env.reset(seed=seed + 999 + t)
                h.zero_()
                msg_prev.fill(0.0)
            else:
                obs = next_obs

        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
    finally:
        writer.close()
        env.close()

    print(f"Saved video to: {out_mp4}")



@torch.no_grad()
def play_recurrent(cfg: Any, env_kwargs: dict, model: Any, obs_rms: RunningMeanStd,
                   seed: int, steps: int = 3000):
    device = next(model.parameters()).device

    env = make_env(env_kwargs, render_mode="human")
    obs, _ = env.reset(seed=seed)
    agents = get_trainable_agents(env, obs_dict=obs)

    B = len(agents)
    h = model.init_hidden(B, device)
    msg_prev = np.zeros((B, cfg.msg_dim), dtype=np.float32)

    for t in range(steps):
        obs_np = stack_obs_np(obs, agents)
        k_eff = min(cfg.k_neighbors, max(1, B - 1))
        neigh = knn_neighbors_from_obs(obs_np, k=k_eff)
        msg_in_np = local_broadcast_mean(msg_prev, neigh)

        O = torch.tensor(obs_np, dtype=torch.float32, device=device)
        O = obs_rms.normalize(O)
        M = torch.tensor(msg_in_np, dtype=torch.float32, device=device)

        mean, std, msg_out, gate, h2 = model.forward_step(O, M, h)

        a = torch.tanh(mean).cpu().numpy().astype(np.float32)
        next_obs, _, term, trunc, _ = env.step(unstack_actions_np(a, agents))

        d = torch.tensor([float(term[a_] or trunc[a_]) for a_ in agents], dtype=torch.float32, device=device)
        h = h2 * (1.0 - d).unsqueeze(-1)
        msg_prev = msg_out.cpu().numpy().astype(np.float32)

        if all_done_for_agents(term, trunc, agents):
            obs, _ = env.reset(seed=seed + 1234 + t)
            h.zero_()
            msg_prev.fill(0.0)
        else:
            obs = next_obs

    env.close()

def plot_key(logger, y_key: str, x_key: str = "env_steps", *, title: str | None = None):

    if not getattr(logger, "rows", None):
        print("Logger is empty.")
        return

    xs, ys = [], []
    for i, r in enumerate(logger.rows):
        if y_key not in r:
            continue
        y = r.get(y_key, None)
        if y is None:
            continue
        try:
            y = float(y)
        except Exception:
            continue
        if np.isnan(y):
            continue

        x = r.get(x_key, i)
        try:
            x = float(x)
        except Exception:
            x = float(i)

        xs.append(x)
        ys.append(y)

    if len(xs) == 0:
        print(f"Missing key or no valid values: {y_key}")
        return

    order = np.argsort(xs)
    xs = np.array(xs, dtype=np.float64)[order]
    ys = np.array(ys, dtype=np.float64)[order]

    plt.figure()
    plt.plot(xs, ys)
    plt.title(title or f"{y_key} vs {x_key}")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.25)
    plt.show()

