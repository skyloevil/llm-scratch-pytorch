import torch

torch.manual_seed(0)

def max_abs_diff(a, b):
    return (a - b).abs().max().item()

# ----------------------------
# 超参数
# ----------------------------
B = 4          # batch
H_in = 16      # 输入维度
H_out = 12     # 输出维度
tp = 2         # 切成2份
assert H_out % tp == 0 and H_in % tp == 0

dtype = torch.float32
device = "cpu"

# 随机输入与权重（不带bias，便于对照）
X  = torch.randn(B, H_in, dtype=dtype, device=device)
Wc = torch.randn(H_out, H_in, dtype=dtype, device=device)  # for ColumnParallel
Wr = Wc.clone()                                            # 复用同一权重便于对照

# -------- Baseline（未切分）--------
Y_full = X @ Wc.t()  # [B, H_out]

# ============================================================
# 1) ColumnParallelLinear 等价性：按 out_features 切 + concat(gather)
# ============================================================
# 列切（按第0维=out_features）
chunk = H_out // tp
Wc0, Wc1 = Wc[:chunk, :], Wc[chunk:, :]

# 各 rank 本地计算（对应各自列块）
Yc0 = X @ Wc0.t()      # [B, H_out/tp]
Yc1 = X @ Wc1.t()      # [B, H_out/tp]

# all_gather → concat
Yc_gather = torch.cat([Yc0, Yc1], dim=-1)  # [B, H_out]

print("[ColumnParallel]  max|Y_full - concat(partials)| =",
      max_abs_diff(Y_full, Yc_gather))

# ============================================================
# 2) RowParallelLinear 等价性：按 in_features 切 + all_reduce(sum)
# ============================================================
# 行切（按第1维=in_features）
chunk_in = H_in // tp
X0, X1   = X[:, :chunk_in],      X[:, chunk_in:]            # [B, H_in/tp]
Wr0, Wr1 = Wr[:, :chunk_in],     Wr[:, chunk_in:]           # [H_out, H_in/tp]

# 各 rank 本地计算“部分贡献”
Yr0 = X0 @ Wr0.t()  # [B, H_out]
Yr1 = X1 @ Wr1.t()  # [B, H_out]

# all_reduce(sum) → 按元素求和
Yr_sum = Yr0 + Yr1  # [B, H_out]

print("[RowParallel]     max|Y_full - sum(partials)|    =",
      max_abs_diff(Y_full, Yr_sum))

# ============================================================
# 3) “先在分片域相加、再一次性 gather”的等价性
#    模拟 embed_fc + hidden_fc：两个线性输出先在每个 rank 相加
#    （列并行同分块），然后只做一次 gather
# ============================================================
# 两个整权重（和上面不同，模拟两条支路）
We = torch.randn(H_out, H_in, dtype=dtype, device=device)  # embed_fc
Wh = torch.randn(H_out, H_in, dtype=dtype, device=device)  # hidden_fc

# 未切分 baseline：先各自全量线性再相加
Z_full = (X @ We.t()) + (X @ Wh.t())  # [B, H_out]

# 列并行：对两套权重用**相同的 out 切分**
We0, We1 = We[:chunk, :], We[chunk:, :]
Wh0, Wh1 = Wh[:chunk, :], Wh[chunk:, :]

# 各 rank 本地：先把同 rank 的两支输出相加（推迟通信）
Ze0 = X @ We0.t()   # [B, H_out/tp]
Zh0 = X @ Wh0.t()
Zloc0 = Ze0 + Zh0

Ze1 = X @ We1.t()
Zh1 = X @ Wh1.t()
Zloc1 = Ze1 + Zh1

# 一次性 gather
Z_gather = torch.cat([Zloc0, Zloc1], dim=-1)

print("[ColPar+delay]    max|Z_full - concat(local_sum)| =",
      max_abs_diff(Z_full, Z_gather))

# ============================================================
# 可选：带 bias 的 Column / Row 验证（说明与常见做法）
# - ColumnParallel: bias 也按 out_features 切分到各 rank，先加局部 bias 再 concat
# - RowParallel:    bias 一般不切分，all-reduce 之后统一加同一份 bias
# ============================================================
use_bias = True
if use_bias:
    b = torch.randn(H_out, dtype=dtype, device=device)

    # Baseline
    Yb_full = Y_full + b

    # ColumnParallel 带 bias：各 rank 拿到各自 bias 分片
    b0, b1 = b[:chunk], b[chunk:]
    Yb_col = torch.cat([Yc0 + b0, Yc1 + b1], dim=-1)
    print("[Column+bias]    max|Y_full+b - concat(partials+b_r)| =",
          max_abs_diff(Yb_full, Yb_col))

    # RowParallel 带 bias：先 sum，再一次性加完整 bias
    Yb_row = (Yr0 + Yr1) + b
    print("[Row+bias]       max|Y_full+b - sum(partials)-b  | =",
          max_abs_diff(Yb_full, Yb_row))
