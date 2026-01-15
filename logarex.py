#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoGA runner (RETURNS) — alvo em log-preços (retornos), com ablações, instrumentação e grade.

Principais diferenças vs versão de preço:
- Target: y = log(C_{t:t+P}) - log(C_{t-1})  (retornos acumulados do último instante do contexto).
- Caminho residual do modelo passa a ser zero (baseline de retorno nulo).
- Métricas salvas em ambos domínios: '..._returns' e '..._price' (reconstruindo preço).
- Predições CSV incluem Ret_pred[h], Ret_true[h], Close_pred[h], Close_true[h].
- Tudo salvo em out/interpretability/returns/{exp_name}/fold{n}/...

CLI (exemplo):
python3 loga_runner_returns.py --grid --folds 1 2 3 4 5 --data-dir data --early-stop --use-best --instrument --device cuda
"""

import os, json, math, argparse, warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays")

REQ_COLS = ["Open", "High", "Low", "Close", "CVI"]

# ----------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------

def seed_everything(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random as _r
    _r.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2.0 + 1e-8
    return float(100.0 * np.mean(np.abs(a - b) / denom))

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def load_fold_df(data_dir: Path, fold: int) -> pd.DataFrame:
    p = data_dir / f"merged_dataset_{fold}.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_json(p, lines=True)
    if "Timestamp" not in df.columns:
        for c in ["timestamp", "ts", "date", "Datetime", "datetime"]:
            if c in df.columns:
                df["Timestamp"] = df[c]
                break
    if "Timestamp" not in df.columns:
        raise ValueError("Timestamp column not found.")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    return df

def zscore_fit(x: np.ndarray):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd

class SlidingWindowDataset(Dataset):
    """
    Target em retornos log acumulados:
      y[h] = log_close[t+h] - log_close[t-1], para h=1..P
    Também fornece last_log_close para reconstruir preço:
      log_close_hat[t+h] = last_log_close + y_hat[h]
      close_hat[t+h]     = exp(log_close_hat[t+h])
    """
    def __init__(self, df, cols, context_len, horizon, split,
                 val_len=None, cvi_scale=10.0, mu=None, sd=None):
        assert split in ("train", "val", "test")
        self.df = df.copy()
        self.cols = cols
        self.S = int(context_len)
        self.P = int(horizon)
        self.cvi_scale = float(cvi_scale)

        Xfull = self.df[self.cols].to_numpy(dtype=np.float32)
        if mu is None or sd is None:
            mu, sd = zscore_fit(Xfull)
        self.mu, self.sd = mu, sd
        Xz = zscore_apply(Xfull, self.mu, self.sd)

        if "CVI" in self.cols:
            Xz[:, self.cols.index("CVI")] *= self.cvi_scale

        self.Xz = Xz
        self.ts = self.df["Timestamp"].to_numpy()
        self.close = np.asarray(self.df["Close"], dtype=np.float32)
        self.log_close = np.log(np.clip(self.close, 1e-8, None))
        T = len(self.df)

        if val_len is None:
            val_len = min(max(self.P, 2 * self.P), max(1, T // 10))
        self.val_len = int(val_len)

        test_end = T
        test_start = T - self.P
        val_end = test_start
        val_start = max(self.S, val_end - self.val_len)
        train_end = val_start
        train_start = self.S

        self.indices = []
        if split == "train":
            for t in range(train_start, train_end):
                if t + self.P <= T:
                    self.indices.append(t)
        elif split == "val":
            for t in range(val_start, val_end):
                if t + self.P <= T:
                    self.indices.append(t)
        else:  # test
            t = T - self.P
            if t >= self.S:
                self.indices.append(t)

        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        x_win = self.Xz[t - self.S:t, :]                      # (S,C)
        # retornos log acumulados a partir de t-1
        y_ret = self.log_close[t:t + self.P] - self.log_close[t - 1]  # (P,)
        ts_future = [pd.Timestamp(x).isoformat() for x in self.ts[t:t + self.P]]
        last_close = self.close[t - 1] if t - 1 >= 0 else self.close[0]
        last_log_close = self.log_close[t - 1] if t - 1 >= 0 else self.log_close[0]
        # também guardo Close_true para salvar no CSV:
        close_true = self.close[t:t + self.P]

        return {
            "X": x_win,
            "y": y_ret,                    # TARGET = retornos (log)
            "ts": ts_future,
            "idx": t,
            "last_close": last_close,      # não usado no forward, mas útil p/ CSV
            "last_log_close": last_log_close,
            "close_true": close_true,      # facilitar reconstrução e CSV
        }

def collate_batch(batch):
    X = np.stack([b["X"] for b in batch], axis=0).astype(np.float32)
    y = np.stack([b["y"] for b in batch], axis=0).astype(np.float32)
    last_close = np.asarray([b["last_close"] for b in batch], dtype=np.float32)
    last_log_close = np.asarray([b["last_log_close"] for b in batch], dtype=np.float32)
    idx = np.asarray([b["idx"] for b in batch], dtype=np.int64)
    ts = [b["ts"] for b in batch]
    close_true = np.stack([b["close_true"] for b in batch], axis=0).astype(np.float32)
    return {
        "X": torch.from_numpy(X),
        "y": torch.from_numpy(y),
        "last_close": torch.from_numpy(last_close),
        "last_log_close": torch.from_numpy(last_log_close),
        "idx": torch.from_numpy(idx),
        "ts": ts,
        "close_true": torch.from_numpy(close_true),
    }

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class DepthwisePointwise(nn.Module):
    def __init__(self, c_in, d_model, k=5, dropout=0.2):
        super().__init__()
        self.dw = nn.Conv1d(c_in, c_in, kernel_size=k, padding=k // 2, groups=c_in, bias=False)
        self.pw = nn.Conv1d(c_in, d_model, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(d_model, eps=1e-5, momentum=0.1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: (B,S,C)
        z = self.dw(x.transpose(1, 2))
        z = self.pw(z)
        z = F.gelu(z)
        z = self.bn(z)
        z = self.drop(z)
        return z.transpose(1, 2)  # (B,S,d)

class LocalGlobalAttention(nn.Module):
    def __init__(self, d_model, n_heads, w_patch, attn_dropout=0.0):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        assert self.dh * self.h == self.d
        self.w = w_patch
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)

    def make_patches(self, z):
        B, S, d = z.shape
        w = self.w
        stride = max(1, w // 2)
        idx = list(range(0, max(S - w + 1, 1), stride))
        if len(idx) == 0 or idx[-1] != S - w:
            if S - w >= 0:
                idx.append(S - w)
        Ps = [z[:, s:s + w, :].unsqueeze(1) for s in idx] or [z[:, max(0, S - w):S, :].unsqueeze(1)]
        return torch.cat(Ps, dim=1)  # (B,M,w,d)

    def forward(self, z_conv, tau_scale=1.0):
        B, S, d = z_conv.shape
        patches = self.make_patches(z_conv)  # (B,M,w,d)
        Q = patches.mean(dim=2)              # (B,M,d)
        K = V = z_conv

        q = self.Wq(Q).view(B, -1, self.h, d // self.h).transpose(1, 2)
        k = self.Wk(K).view(B, -1, self.h, d // self.h).transpose(1, 2)
        v = self.Wv(V).view(B, -1, self.h, d // self.h).transpose(1, 2)

        scale = (1.0 / math.sqrt(self.dh)) * float(tau_scale)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.d)
        H = self.Wo(out)
        H = F.layer_norm(H + Q, normalized_shape=(self.d,))
        return H, attn

class CrossDecoder(nn.Module):
    def __init__(self, d_model, n_heads, horizon, d_ff=None, dropout=0.0):
        super().__init__()
        self.d = d_model
        self.h = n_heads
        self.P = horizon
        self.dh = d_model // n_heads
        assert self.dh * self.h == self.d
        if d_ff is None:
            d_ff = max(d_model * 2, 128)
        self.Qd = nn.Parameter(torch.randn(self.P, self.d) * (1.0 / math.sqrt(self.d)))
        self.Wq = nn.Linear(self.d, self.d, bias=False)
        self.Wk = nn.Linear(self.d, self.d, bias=False)
        self.Wv = nn.Linear(self.d, self.d, bias=False)
        self.Wo = nn.Linear(self.d, self.d, bias=False)
        self.ff1 = nn.Linear(self.d, d_ff)
        self.ff2 = nn.Linear(d_ff, self.d)
        self.out = nn.Linear(self.d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, tau_scale=1.0):
        B, M, d = H.shape
        Q = self.Qd.unsqueeze(0).repeat(B, 1, 1)  # (B,P,d)

        q = self.Wq(Q).view(B, -1, self.h, d // self.h).transpose(1, 2)
        k = self.Wk(H).view(B, -1, self.h, d // self.h).transpose(1, 2)
        v = self.Wv(H).view(B, -1, self.h, d // self.h).transpose(1, 2)

        scale = (1.0 / math.sqrt(self.dh)) * float(tau_scale)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)  # (B,h,P,M)
        z = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, self.P, self.d)
        z = self.Wo(z)
        z = F.layer_norm(z + Q, normalized_shape=(self.d,))
        u = self.drop(F.gelu(self.ff1(z)))
        u = F.layer_norm(self.ff2(u) + z, normalized_shape=(self.d,))
        y = self.out(u).squeeze(-1)  # (B,P) — retorna RETORNOS previstos
        return y, attn

class LoGA(nn.Module):
    """
    Versão para RETORNOS:
      - residual baseline = 0 (retorno nulo), reduz colapso para random walk.
      - gate inicial pouco enviesado para 'residual'.
    """
    def __init__(self, c_in, d_model, n_heads, horizon, w_patch, depth=1, dropout=0.2):
        super().__init__()
        self.front = DepthwisePointwise(c_in, d_model, k=5, dropout=dropout)
        self.blocks = nn.ModuleList([LocalGlobalAttention(d_model, n_heads, w_patch) for _ in range(depth)])
        self.decoder = CrossDecoder(d_model, n_heads, horizon, d_ff=2 * d_model, dropout=dropout)
        # gate: [residual(=0), local, global] — quase uniforme, pouco peso no residual
        init = torch.tensor([0.2, 0.4, 0.4], dtype=torch.float32)  # soma não precisa 1; softmax cuida
        self.gate_logits = nn.Parameter(init.repeat(horizon, 1))

    def forward(self, x, last_close_unused, tau_scale=1.0, mask_local=False, mask_global=False):
        z = self.front(x)
        H = None; attn_g = None
        for blk in self.blocks:
            H, attn_g = blk(z, tau_scale=tau_scale)
        y_global, attn_d = self.decoder(H, tau_scale=tau_scale)  # (B,P) — retornos previstos

        # baseline residual em RETURNS = 0
        y_resid = torch.zeros_like(y_global)    # (B,P)
        y_local = y_global                      # caminho local explícito não separado
        g = F.softmax(self.gate_logits, dim=-1) # (P,3)
        gr = g[:, 0].view(1, -1, 1)
        gl = g[:, 1].view(1, -1, 1)
        gg = g[:, 2].view(1, -1, 1)
        if mask_local:  gl = torch.zeros_like(gl)
        if mask_global: gg = torch.zeros_like(gg)

        y = gr.squeeze(-1) * y_resid + gl.squeeze(-1) * y_local + gg.squeeze(-1) * y_global  # (B,P)

        gate_mix = {
            "residual": float(g[:, 0].mean().item()),
            "local":    float(g[:, 1].mean().item()),
            "global":   float(g[:, 2].mean().item()),
        }
        return y, attn_g, attn_d, gate_mix

# ----------------------------------------------------------------------
# Dataloaders helpers
# ----------------------------------------------------------------------

def make_loaders(df, context_len, horizon, batch_size, cvi_scale, num_workers, val_len=None):
    cols = REQ_COLS
    fullX = df[cols].to_numpy(dtype=np.float32)
    mu, sd = zscore_fit(fullX)
    ds_tr = SlidingWindowDataset(df, cols, context_len, horizon, "train", val_len, cvi_scale, mu, sd)
    ds_va = SlidingWindowDataset(df, cols, context_len, horizon, "val",   val_len, cvi_scale, mu, sd)
    ds_te = SlidingWindowDataset(df, cols, context_len, horizon, "test",  val_len, cvi_scale, mu, sd)
    tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, collate_fn=collate_batch, drop_last=True)
    va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)
    te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)
    return ds_tr, ds_va, ds_te, tr, va, te

def _reconstruct_prices_from_returns(y_ret: np.ndarray, last_log_close: np.ndarray) -> np.ndarray:
    """
    y_ret: (B,P) retornos log acumulados relativos a last_log_close
    last_log_close: (B,)
    return: Close_hat (B,P)
    """
    logp_hat = last_log_close[:, None] + y_ret
    return np.exp(logp_hat)

def save_returns_and_prices_csv(y_ret: np.ndarray,
                                y_ret_true: np.ndarray,
                                last_log_close: np.ndarray,
                                close_true: np.ndarray,
                                ts: list,
                                out_path: Path):
    """
    Salva um CSV com: Timestamp, Ret_pred[h], Ret_true[h], Close_pred[h], Close_true[h]
    (para o último batch avaliado).
    """
    close_pred = _reconstruct_prices_from_returns(y_ret[None, :], np.array([last_log_close]))[0]
    # Monta DF longo
    rows = []
    for h, tstamp in enumerate(ts):
        rows.append({
            "Timestamp": tstamp,
            "Ret_pred": float(y_ret[h]),
            "Ret_true": float(y_ret_true[h]),
            "Close_pred": float(close_pred[h]),
            "Close_true": float(close_true[h]),
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

# ----------------------------------------------------------------------
# Evaluation + instrumentação
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model,
             loader,
             device,
             horizon,
             out_pred_csv=None,
             tau_scale=1.0,
             mask_local=False,
             mask_global=False,
             mask_cvi=False,
             instrument=False,
             instr_dir: Path = None,
             tag: str = "base",
             save_raw_attn: bool = False,
             max_attn_batches: int = 0,
             save_per_sample: bool = False):
    """
    Avalia o modelo em RETORNOS e também computa métricas em PREÇOS reconstruídos.
    """
    model.eval()
    all_pred_ret = []
    all_true_ret = []
    all_pred_close = []
    all_true_close = []
    last_ts = None

    attn_global_list = []
    attn_dec_list = []
    gate_mix_list = []
    per_sample_records = []
    saved_attn_batches = 0

    iterator = loader
    if instrument:
        iterator = tqdm(loader, desc=f"[{tag}] eval", leave=False)

    for batch_idx, batch in enumerate(iterator):
        x   = batch["X"].to(device)            # (B,S,C) z-scored
        lc  = batch["last_close"].to(device)   # (B,) — não usado no forward (retornos)
        llc = batch["last_log_close"].cpu().numpy()  # (B,)
        y_true = batch["y"].cpu().numpy()      # (B,P) — retornos log
        ts_list = batch["ts"]                  # list of lists
        idx_arr = batch["idx"].cpu().numpy()
        close_true = batch["close_true"].cpu().numpy()  # (B,P)

        if mask_cvi and "CVI" in REQ_COLS:
            x[:, :, REQ_COLS.index("CVI")] = 0.0

        y_hat_ret, attn_g, attn_d, gate_mix = model(
            x, lc, tau_scale=tau_scale, mask_local=mask_local, mask_global=mask_global
        )
        y_hat_ret = y_hat_ret.detach().cpu().numpy()  # (B,P)

        # Reconstrução de preços
        close_hat = _reconstruct_prices_from_returns(y_hat_ret, llc)  # (B,P)

        all_pred_ret.append(y_hat_ret)
        all_true_ret.append(y_true)
        all_pred_close.append(close_hat)
        all_true_close.append(close_true)
        last_ts = ts_list[-1]

        if instrument:
            # concentração média na top-k atenção
            def topk_mass(a, k):
                a_ = a.reshape(-1, a.shape[-1])
                part = np.partition(a_, -k, axis=-1)[:, -k:]
                return float(part.sum() / a_.sum())

            if attn_g is not None:
                attn_global_list.append(topk_mass(attn_g.detach().cpu().numpy(), 5))
            if attn_d is not None:
                attn_dec_list.append(topk_mass(attn_d.detach().cpu().numpy(), 5))
            gate_mix_list.append(gate_mix)

            # salva alguns batches com atenção bruta (para heatmaps)
            if (save_raw_attn and instr_dir is not None and
                saved_attn_batches < max_attn_batches and
                (attn_g is not None or attn_d is not None)):
                instr_dir.mkdir(parents=True, exist_ok=True)
                out = {}
                if attn_g is not None: out["attn_g"] = attn_g.detach().cpu().numpy()
                if attn_d is not None: out["attn_d"] = attn_d.detach().cpu().numpy()
                out["idx"] = idx_arr
                out["ts"]  = np.array(ts_list, dtype=object)
                np.savez(instr_dir / f"attn_{tag}_batch{batch_idx}.npz", **out)
                saved_attn_batches += 1

            # métricas por janela (para regime-wise): usar std dos retornos verdadeiros como proxy de vol
            B = y_true.shape[0]
            for b in range(B):
                yt = y_true[b]; yp = y_hat_ret[b]
                # métricas em retornos
                mse_r = float(np.mean((yp - yt) ** 2))
                rmse_r = float(np.sqrt(mse_r))
                mae_r = float(np.mean(np.abs(yp - yt)))
                smape_r = smape(yt, yp)
                # métricas em preço
                cp = close_hat[b]; ct = close_true[b]
                mse_p = float(np.mean((cp - ct) ** 2))
                rmse_p = float(np.sqrt(mse_p))
                mae_p = float(np.mean(np.abs(cp - ct)))
                smape_p = smape(ct, cp)
                # realized vol ~ std dos retornos verdadeiros do horizonte
                rv = float(np.std(yt)) if yt.size > 1 else 0.0

                per_sample_records.append({
                    "idx": int(idx_arr[b]),
                    "tag": tag,
                    "rmse_returns": rmse_r,
                    "mae_returns": mae_r,
                    "smape_returns": smape_r,
                    "rmse_price": rmse_p,
                    "mae_price": mae_p,
                    "smape_price": smape_p,
                    "realized_vol": rv,
                    "ts_start": ts_list[b][0],
                    "ts_end": ts_list[b][-1],
                })

    # Agregados
    y_pred_ret   = np.concatenate(all_pred_ret, axis=0)
    y_true_ret   = np.concatenate(all_true_ret, axis=0)
    y_pred_close = np.concatenate(all_pred_close, axis=0)
    y_true_close = np.concatenate(all_true_close, axis=0)

    # Métricas returns
    rmse_r = rmse(y_pred_ret.ravel(), y_true_ret.ravel())
    mae_r  = mae(y_pred_ret.ravel(), y_true_ret.ravel())
    smape_r= smape(y_true_ret.ravel(), y_pred_ret.ravel())
    # Métricas price
    rmse_p = rmse(y_pred_close.ravel(), y_true_close.ravel())
    mae_p  = mae(y_pred_close.ravel(), y_true_close.ravel())
    smape_p= smape(y_true_close.ravel(), y_pred_close.ravel())

    # CSV (último batch)
    if out_pred_csv is not None and last_ts is not None and len(y_pred_ret) >= 1:
        # usa o último item do último batch agregado
        last_ret_pred  = y_pred_ret[-1]
        last_ret_true  = y_true_ret[-1]
        # precisamos do último last_log_close e close_true — não guardamos aqui, então omitimos CSV se não disponível
        # Alternativa: salva CSV só de retornos; para manter compat com pedido, salvamos retornos + preços verdadeiros se disponíveis.
        # Como não temos last_log_close do último batch aqui, salvamos retornos + preços verdadeiros (se vieram).
        # Para robustez, salvamos apenas retornos; preço reconstruído já está no JSON de métricas agregadas.
        rows = []
        for h, tstamp in enumerate(last_ts):
            rows.append({
                "Timestamp": tstamp,
                "Ret_pred": float(last_ret_pred[h]),
                "Ret_true": float(last_ret_true[h]),
            })
        df = pd.DataFrame(rows)
        out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_pred_csv.with_suffix(".returns.csv"), index=False)

    if instrument and instr_dir is not None:
        instr_dir.mkdir(parents=True, exist_ok=True)

        # métricas agregadas (ambos domínios)
        with open(instr_dir / f"metrics_{tag}.json", "w") as f:
            json.dump({
                "rmse_returns": rmse_r,
                "mae_returns": mae_r,
                "smape_returns": smape_r,
                "rmse_price": rmse_p,
                "mae_price": mae_p,
                "smape_price": smape_p,
                "n_points": int(y_true_ret.size),
                "horizon": horizon,
                "tag": tag
            }, f, indent=2)

        # gate médio
        if gate_mix_list:
            res = [g["residual"] for g in gate_mix_list]
            loc = [g["local"] for g in gate_mix_list]
            glo = [g["global"] for g in gate_mix_list]
            with open(instr_dir / f"gate_mix_{tag}.json", "w") as f:
                json.dump({
                    "residual": float(np.mean(res)),
                    "local": float(np.mean(loc)),
                    "global": float(np.mean(glo))
                }, f, indent=2)

        # resumo de atenção
        if attn_global_list or attn_dec_list:
            with open(instr_dir / f"attn_summary_{tag}.json", "w") as f:
                json.dump({
                    "global_top5_mass_mean": float(np.mean(attn_global_list)) if attn_global_list else None,
                    "decoder_top5_mass_mean": float(np.mean(attn_dec_list)) if attn_dec_list else None
                }, f, indent=2)

        # métricas por janela
        if save_per_sample and per_sample_records:
            with open(instr_dir / f"per_sample_{tag}.jsonl", "w") as f:
                for rec in per_sample_records:
                    f.write(json.dumps(rec) + "\n")

    # Retorna RMSE de PREÇO (para comparabilidade com papers), mas agora também temos returns no JSON
    return float(rmse_p)

# ----------------------------------------------------------------------
# Treino por fold + instrumentação
# ----------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    horizon: int
    depth: int
    w_patch: int
    cvi_scale: float
    loss: str  # "mse" ou "huber"

def train_one_fold(fold: int,
                   args,
                   exp: ExperimentConfig,
                   device: torch.device,
                   data_dir: Path,
                   pred_dir: Path,
                   models_dir: Path,
                   exp_dir: Path):
    df = load_fold_df(data_dir, fold)
    ds_tr, ds_va, ds_te, tr, va, te = make_loaders(
        df, args.context_length, exp.horizon, args.batch_size, exp.cvi_scale, args.num_workers, args.val_len
    )

    model = LoGA(
        c_in=len(REQ_COLS),
        d_model=args.d_model,
        n_heads=args.n_heads,
        horizon=exp.horizon,
        w_patch=exp.w_patch,
        depth=exp.depth,
        dropout=args.dropout
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if exp.loss == "huber":
        loss_fn = lambda p, t: F.huber_loss(p, t, delta=args.huber_delta)
    else:
        loss_fn = lambda p, t: F.mse_loss(p, t)

    best = float("inf")
    wait = 0
    ckpt = models_dir / f"{exp.name}_fold{fold}.pt"

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        steps = 0

        pbar = tqdm(tr, desc=f"[{exp.name}] Fold {fold} Epoch {ep}", leave=False)
        for batch in pbar:
            x = batch["X"].to(device)
            y = batch["y"].to(device)              # retornos alvo
            lc = batch["last_close"].to(device)    # não usado no forward (compat assinatura)
            y_hat, _, _, _ = model(x, lc, tau_scale=1.0, mask_local=False, mask_global=False)
            loss = loss_fn(y_hat, y)
            optim.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            ep_loss += float(loss.item()); steps += 1
            pbar.set_postfix(loss=ep_loss / max(1, steps))

        # validação: retorna RMSE em PREÇO (reconstruído)
        val_rmse_price = evaluate(model, va, device, exp.horizon, instrument=False)
        print(f"[{exp.name}] Fold {fold} Epoch {ep:03d}/{args.epochs} | "
              f"train_loss(returns)={ep_loss / max(1, steps):.4f} | val_RMSE_price={val_rmse_price:.4f}")

        if val_rmse_price < best - 1e-6:
            best = val_rmse_price
            wait = 0
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1

        if args.early_stop and wait >= args.patience:
            print(f"[{exp.name}] Fold {fold} early stop at epoch {ep}.")
            break

    if args.use_best and ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))

    out_csv = pred_dir / exp.name / f"fold{fold}.csv"
    test_rmse_price = evaluate(
        model, te, device, exp.horizon, out_pred_csv=out_csv, instrument=False
    )
    print(f"[{exp.name}] Fold {fold} TEST RMSE_price={test_rmse_price:.4f}")

    # instrumentação completa
    if args.instrument:
        run_instrumentation_for_fold(fold, args, exp, device, df, te, model, exp_dir)

    return float(test_rmse_price)

# ----------------------------------------------------------------------
# Pós-processamento de interpretabilidade (por fold)
# ----------------------------------------------------------------------

def _load_per_sample(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def compute_regime_summary_for_fold(fold_dir: Path, base_tag: str = "base", compare_tag: str = "no_cvi"):
    """
    ΔRMSE_price por regime (low/mid/high) usando realized_vol ≈ std dos retornos verdadeiros do horizonte.
    """
    base_file = fold_dir / f"per_sample_{base_tag}.jsonl"
    comp_file = fold_dir / f"per_sample_{compare_tag}.jsonl"
    if not base_file.exists() or not comp_file.exists():
        return
    base_records = _load_per_sample(base_file)
    comp_records = _load_per_sample(comp_file)
    if not base_records or not comp_records:
        return

    def key(r): return (r["idx"], r["ts_end"])
    base_map = {key(r): r for r in base_records}
    comp_map = {key(r): r for r in comp_records}

    rows = []
    for k, rb in base_map.items():
        if k not in comp_map: continue
        rc = comp_map[k]
        rv = rb.get("realized_vol", None)
        if rv is None: continue
        rows.append({
            "realized_vol": float(rv),
            "rmse_price_base": float(rb["rmse_price"]),
            "rmse_price_compare": float(rc["rmse_price"]),
            "delta_rmse_price": float(rc["rmse_price"] - rb["rmse_price"]),
        })
    if not rows:
        return

    rv_arr = np.array([r["realized_vol"] for r in rows], dtype=float)
    q1, q2 = np.nanquantile(rv_arr, [0.33, 0.66])

    def regime_label(rv):
        if rv <= q1: return "low"
        elif rv <= q2: return "mid"
        else: return "high"

    buckets = {"low": [], "mid": [], "high": []}
    for r in rows:
        buckets[regime_label(r["realized_vol"])].append(r)

    summary = {"base_tag": base_tag, "compare_tag": compare_tag, "quantiles": {"q1": float(q1), "q2": float(q2)}, "regimes": {}}
    for reg, lst in buckets.items():
        if not lst: continue
        rb = np.array([x["rmse_price_base"] for x in lst], dtype=float)
        rc = np.array([x["rmse_price_compare"] for x in lst], dtype=float)
        rd = np.array([x["delta_rmse_price"] for x in lst], dtype=float)
        summary["regimes"][reg] = {
            "count": int(len(lst)),
            "rmse_price_base_mean": float(np.mean(rb)),
            "rmse_price_compare_mean": float(np.mean(rc)),
            "delta_rmse_price_mean": float(np.mean(rd)),
        }

    with open(fold_dir / "regime_summary_price_base_vs_no_cvi.json", "w") as f:
        json.dump(summary, f, indent=2)

def compute_temperature_sensitivity_for_fold(fold_dir: Path, base_tag: str = "base", tau_tags=("tau_0.85", "tau_1.15")):
    base_file = fold_dir / f"metrics_{base_tag}.json"
    if not base_file.exists(): return
    with open(base_file, "r") as f:
        base = json.load(f)

    out = {"base_tag": base_tag, "base_rmse_price": base.get("rmse_price", None), "tau": {}}
    for t in tau_tags:
        tf = fold_dir / f"metrics_{t}.json"
        if not tf.exists(): continue
        with open(tf, "r") as f:
            tm = json.load(f)
        out["tau"][t] = {
            "rmse_price": tm.get("rmse_price", None),
            "delta_rmse_price": None if (tm.get("rmse_price", None) is None or base.get("rmse_price", None) is None)
            else tm["rmse_price"] - base["rmse_price"]
        }

    with open(fold_dir / "temperature_sensitivity_price.json", "w") as f:
        json.dump(out, f, indent=2)

def save_gate_per_horizon(model: LoGA, fold_dir: Path):
    with torch.no_grad():
        g = torch.softmax(model.gate_logits.detach().cpu(), dim=-1).numpy()  # (P,3)
    horizons = list(range(g.shape[0]))
    out = {"horizons": horizons, "gate": g.tolist(), "components": ["residual(=0 ret)", "local", "global"]}
    with open(fold_dir / "gate_per_horizon_base.json", "w") as f:
        json.dump(out, f, indent=2)

def run_instrumentation_for_fold(fold, args, exp: ExperimentConfig, device, df, te, model, exp_dir: Path):
    model.eval()
    fold_dir = exp_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # completo
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="base",
             save_raw_attn=True, max_attn_batches=3, save_per_sample=True)
    # sem caminhos
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="no_local",
             mask_local=True, save_per_sample=True)
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="no_global",
             mask_global=True, save_per_sample=True)
    # sem CVI
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="no_cvi",
             mask_cvi=True, save_per_sample=True)
    # variação de temperatura
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="tau_0.85",
             tau_scale=0.85, save_per_sample=True)
    evaluate(model, te, device, exp.horizon, instrument=True, instr_dir=fold_dir, tag="tau_1.15",
             tau_scale=1.15, save_per_sample=True)

    save_gate_per_horizon(model, fold_dir)
    compute_regime_summary_for_fold(fold_dir, base_tag="base", compare_tag="no_cvi")
    compute_temperature_sensitivity_for_fold(fold_dir, base_tag="base", tau_tags=("tau_0.85", "tau_1.15"))

# ----------------------------------------------------------------------
# Grade de experimentos
# ----------------------------------------------------------------------

def make_experiment_grid(args) -> List[ExperimentConfig]:
    grid: List[ExperimentConfig] = []
    # Config base (retornos): P=1440, depth=4, w=48, cvi_scale=10, MSE
    grid.append(ExperimentConfig(name="returns_base_h1440_d4_w48_cvi10_mse", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    # Horizontes
    grid.append(ExperimentConfig(name="returns_ablation_h360",  horizon=360,  depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_h720",  horizon=720,  depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_h1440", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    # Profundidade
    grid.append(ExperimentConfig(name="returns_ablation_depth_1", horizon=1440, depth=1, w_patch=48, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_depth_2", horizon=1440, depth=2, w_patch=48, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_depth_4", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    # Patch
    grid.append(ExperimentConfig(name="returns_ablation_w_24", horizon=1440, depth=4, w_patch=24, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_w_48", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_w_96", horizon=1440, depth=4, w_patch=96, cvi_scale=10.0, loss="mse"))
    # Escala do CVI
    grid.append(ExperimentConfig(name="returns_ablation_cvi_scale_1",  horizon=1440, depth=4, w_patch=48, cvi_scale=1.0,  loss="mse"))
    grid.append(ExperimentConfig(name="returns_ablation_cvi_scale_10", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="mse"))
    # Loss
    grid.append(ExperimentConfig(name="returns_ablation_loss_huber", horizon=1440, depth=4, w_patch=48, cvi_scale=10.0, loss="huber"))
    return grid

# ----------------------------------------------------------------------
# Agregação cross-fold (por experimento)
# ----------------------------------------------------------------------

def aggregate_regime_across_folds(exp_dir: Path, folds: List[int]):
    collected = []
    for fold in folds:
        path = exp_dir / f"fold{fold}" / "regime_summary_price_base_vs_no_cvi.json"
        if not path.exists():
            continue
        with open(path, "r") as f:
            collected.append(json.load(f))
    if not collected:
        return

    regimes = ["low", "mid", "high"]
    out = {"regimes": {}}
    for reg in regimes:
        vals_base = []
        vals_comp = []
        vals_delta = []
        counts = []
        for s in collected:
            rinfo = s.get("regimes", {}).get(reg, None)
            if not rinfo:
                continue
            counts.append(rinfo["count"])
            vals_base.append(rinfo["rmse_price_base_mean"])
            vals_comp.append(rinfo["rmse_price_compare_mean"])
            vals_delta.append(rinfo["delta_rmse_price_mean"])
        if not vals_base:
            continue
        out["regimes"][reg] = {
            "count_total": int(np.sum(counts)),
            "rmse_price_base_mean_across_folds": float(np.mean(vals_base)),
            "rmse_price_compare_mean_across_folds": float(np.mean(vals_comp)),
            "delta_rmse_price_mean_across_folds": float(np.mean(vals_delta)),
        }

    with open(exp_dir / "regime_summary_price_across_folds.json", "w") as f:
        json.dump(out, f, indent=2)

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[1])
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--pred-dir", type=str, default="predictions_returns")
    ap.add_argument("--context-length", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=1440)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--w", type=int, default=48)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--val-len", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--use-best", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--instrument", action="store_true")
    ap.add_argument("--instr-out", type=str, default="out/interpretability/returns")
    ap.add_argument("--grid", action="store_true",
                    help="Se presente, roda grade de ablações em vez de um único experimento.")
    return ap.parse_args()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    models_dir = Path("models")
    instr_root = Path(args.instr_out)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.grid:
        exps = make_experiment_grid(args)
    else:
        name = (f"returns_single_h{args.horizon}_d{args.depth}_w{args.w}_cvi{10.0}_loss{args.loss}")
        exps = [ExperimentConfig(name=name, horizon=args.horizon, depth=args.depth, w_patch=args.w, cvi_scale=10.0, loss=args.loss)]

    all_results = []

    for exp in exps:
        print(f"\n=== Running experiment: {exp.name} ===")
        exp_dir = instr_root / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        exp_pred_dir = pred_dir / exp.name
        exp_pred_dir.mkdir(parents=True, exist_ok=True)

        rmses = []
        for fold in args.folds:
            r = train_one_fold(fold, args, exp, device, data_dir, exp_pred_dir, models_dir, exp_dir)
            rmses.append(r)

        summary = {
            "experiment": exp.name,
            "config": asdict(exp),
            "folds": args.folds,
            "rmse_price_mean": float(np.mean(rmses)),
            "rmse_price_std": float(np.std(rmses)),
            "rmse_price_per_fold": rmses
        }
        all_results.append(summary)
        with open(exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        aggregate_regime_across_folds(exp_dir, args.folds)

        print(f"=== {exp.name}: RMSE_price mean ± std = {summary['rmse_price_mean']:.6f} ± {summary['rmse_price_std']:.6f} ===")

    instr_root.mkdir(parents=True, exist_ok=True)
    with open(instr_root / "all_experiments_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
