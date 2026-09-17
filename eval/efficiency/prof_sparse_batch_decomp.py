"""Decompose the A1 sparse-batching E2E contribution: sparse-kernel batching vs
last-block-flash batching, independently, on REAL selections.

Since batching only touches the sparse section, the sparse-section wall-time delta
equals the E2E delta (rest of the model is untouched). We replay 4 configs per
length on REAL captured selections, with ALL stacking/gather/scatter overhead INSIDE
the timed region so the numbers are faithful to what run_herald_sweep.sh would show:
  base   : per-head body sparse  + per-head flash          (production)
  kbatch : BATCHED body sparse    + per-head flash
  fbatch : per-head body sparse   + BATCHED flash
  both   : BATCHED body sparse    + BATCHED flash           (the reverted A1)
Autotune ON (speed benchmark). E2E fractions use the measured baseline prefill.
"""
import os, sys, math, torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["REUSE_V1_FUSED_ANCHOR"] = "1"
os.environ.setdefault("http_proxy", "http://agent.baidu.com:8891")
os.environ.setdefault("https_proxy", "http://agent.baidu.com:8891")
REPO = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/sparse_refuse/ReAttn"
sys.path.insert(0, REPO)

import pbs_attn.baselines.Reuse_v1 as R
from transformers import AutoTokenizer, AutoModelForCausalLM
from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_reuse_v1_prefill
from datasets import load_dataset
from flash_attn import flash_attn_func

MODEL = "/root/paddlejob/share-storage/gpfs/system-public/yudaohai/data/Llama-3.1-8B-Instruct"
LABEL = f"{REPO}/attn_patterns/reuse_v1/Llama-3.1-8B-Instruct/hc-orig-rw=0.003-init=0.0-sp=0.8-tp=0.9-lr=0.01-ctx=8000_128000-multi_passkey10-sp8-mb8-xb64/label.pt"
DEV, d, G = "cuda:0", 128, 4
scale = 1.0 / math.sqrt(d)
LENGTHS = [8192, 16384, 32768]
E2E = {8192: 0.347085, 16384: 0.645897, 32768: 1.376613}  # measured baseline prefill (s)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="eager").to(DEV).eval()
prefill_fn = get_reuse_v1_prefill(
    label_path=LABEL, budget=32, block_size=128, segment_size=2048,
    sink_blocks=1, local_blocks=2, causal=True,
    select_mode="topp", top_p=0.95, min_blocks=8, max_blocks=48,
    last_q_full=True, per_head_topp=False)
model = apply_patch_with_prefill(model, prefill_fn)
ds = load_dataset("THUDM/LongBench-v2", split="train")
base_ids = tok(ds[0]["context"], return_tensors="pt").input_ids.to(DEV)


def capture(ids):
    CAP = []
    os_s, os_l = R._sparse_block_attn, R.reuse_v1_layer_per_hkv
    def cs(*a, **kw):
        CAP.append(("S", kw.get("k_sel"), kw.get("k_cnt"))); return os_s(*a, **kw)
    def cl(*a, **kw):
        CAP.append(("L",)); return os_l(*a, **kw)
    R._sparse_block_attn, R.reuse_v1_layer_per_hkv = cs, cl
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    R._sparse_block_attn, R.reuse_v1_layer_per_hkv = os_s, os_l
    layers, cur = [], None
    for rec in CAP:
        if rec[0] == "L":
            if cur: layers.append(cur)
            cur = []
        elif cur is not None:
            cur.append((rec[1], rec[2]))
    if cur: layers.append(cur)
    return [L for L in layers if L]


def timed(fn, iters=15, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(True); c = torch.cuda.Event(True)
        a.record(); fn(); c.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(c))
    ts.sort(); return ts[len(ts) // 2]


print(f"\n{'len':>5} {'base_ms':>8} {'kbatch':>8} {'fbatch':>8} {'both':>8} "
      f"| {'k_dEms':>7} {'f_dEms':>7} {'both_dE':>7} | E2E% k/f/both")

for LEN in LENGTHS:
    ids = base_ids
    if ids.shape[1] < LEN:
        ids = ids.repeat(1, LEN // ids.shape[1] + 1)
    ids = ids[:, :LEN].contiguous()
    with torch.no_grad():
        model.generate(input_ids=ids, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    layers = capture(ids)

    s = LEN
    block_size = 128
    nqb = s // block_size
    # captured k_sel is body (nqb-1 rows)
    nqb_body = layers[0][0][0].shape[-2]
    last_q_start = nqb_body * 128
    last_len = s - last_q_start
    b = 1

    # per-layer materialized buffers (values irrelevant to timing)
    built = []
    for L in layers:
        N = len(L)
        ms = L[0][0].shape[-1]
        qg = torch.randn(b, N, G, s, d, device=DEV, dtype=torch.bfloat16)
        kA = torch.randn(b, N, s, d, device=DEV, dtype=torch.bfloat16)
        vA = torch.randn(b, N, s, d, device=DEV, dtype=torch.bfloat16)
        og = torch.empty(b, N, G, s, d, device=DEV, dtype=torch.bfloat16)
        idx = torch.arange(N, device=DEV)
        sel_list = [c[0] for c in L]; cnt_list = [c[1] for c in L]
        built.append((N, ms, qg, kA, vA, og, idx, sel_list, cnt_list))

    def per_head_body(N, qg, kA, vA, og, sel_list, cnt_list):
        for i in range(N):
            q_h = qg[:, i].contiguous()
            R._sparse_block_attn(q_h[:, :, :last_q_start], kA[:, i:i+1], vA[:, i:i+1],
                                 None, 48, 128, True, scale,
                                 k_sel=sel_list[i], k_cnt=cnt_list[i],
                                 out=og[:, i, :, :last_q_start])

    def per_head_flash(N, qg, kA, vA, og):
        for i in range(N):
            lq = qg[:, i, :, last_q_start:].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
            kf = kA[:, i:i+1].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
            vf = vA[:, i:i+1].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
            og[:, i, :, last_q_start:] = flash_attn_func(lq, kf, vf, softmax_scale=scale,
                                                         causal=True).permute(0, 2, 1, 3)

    def batched_body(N, ms, qg, kA, vA, og, idx, sel_list, cnt_list):
        q_sp = qg[:, idx].reshape(b, N * G, s, d)          # advanced-index copy
        k_sp = kA[:, idx].contiguous(); v_sp = vA[:, idx].contiguous()
        sel4 = torch.stack(sel_list, 1).unsqueeze(2).expand(b, N, G, nqb_body, ms).reshape(b, N * G, nqb_body, ms).contiguous()
        cnt4 = torch.stack(cnt_list, 1).unsqueeze(2).expand(b, N, G, nqb_body).reshape(b, N * G, nqb_body).contiguous()
        body = torch.empty(b, N * G, last_q_start, d, device=DEV, dtype=torch.bfloat16)
        R._sparse_block_attn(q_sp[:, :, :last_q_start], k_sp, v_sp, None, 48, 128, True, scale,
                             k_sel=sel4, k_cnt=cnt4, out=body)
        og[:, idx, :, :last_q_start] = body.view(b, N, G, last_q_start, d)

    def batched_flash(N, qg, kA, vA, og, idx):
        q_sp = qg[:, idx].reshape(b, N * G, s, d)
        k_sp = kA[:, idx].contiguous(); v_sp = vA[:, idx].contiguous()
        lq = q_sp[:, :, last_q_start:].permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        kf = k_sp.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        vf = v_sp.permute(0, 2, 1, 3).contiguous().to(torch.bfloat16)
        lo = flash_attn_func(lq, kf, vf, softmax_scale=scale, causal=True).permute(0, 2, 1, 3)
        og[:, idx, :, last_q_start:] = lo.reshape(b, N, G, last_len, d)

    def cfg_base():
        for N, ms, qg, kA, vA, og, idx, sl, cl in built:
            per_head_body(N, qg, kA, vA, og, sl, cl); per_head_flash(N, qg, kA, vA, og)
    def cfg_kbatch():
        for N, ms, qg, kA, vA, og, idx, sl, cl in built:
            batched_body(N, ms, qg, kA, vA, og, idx, sl, cl); per_head_flash(N, qg, kA, vA, og)
    def cfg_fbatch():
        for N, ms, qg, kA, vA, og, idx, sl, cl in built:
            per_head_body(N, qg, kA, vA, og, sl, cl); batched_flash(N, qg, kA, vA, og, idx)
    def cfg_both():
        for N, ms, qg, kA, vA, og, idx, sl, cl in built:
            batched_body(N, ms, qg, kA, vA, og, idx, sl, cl); batched_flash(N, qg, kA, vA, og, idx)

    tb = timed(cfg_base); tk = timed(cfg_kbatch); tf = timed(cfg_fbatch); to = timed(cfg_both)
    e2e_ms = E2E[LEN] * 1000.0
    dk, df, do = tb - tk, tb - tf, tb - to   # positive = saving
    print(f"{LEN//1024:>4}k {tb:>8.3f} {tk:>8.3f} {tf:>8.3f} {to:>8.3f} "
          f"| {dk:>7.3f} {df:>7.3f} {do:>7.3f} | "
          f"{100*dk/e2e_ms:+.2f}% {100*df/e2e_ms:+.2f}% {100*do/e2e_ms:+.2f}%")
    del built; torch.cuda.empty_cache()

