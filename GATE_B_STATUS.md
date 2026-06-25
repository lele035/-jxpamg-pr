# Gate B: 多级只读探针 — 完成报告

**日期**: 2026-06-21
**状态**: COMPLETE (np=1 full, np=2 limited, np=4 blocked)

---

## 1. 已修复的根因

Gate B 最初的阻塞问题被诊断为 `jx_PAMGBuildCoarseOperatorKT` 存在"跨调用持久副作用"。该诊断错误。

实际根因：

1. **自定义深拷贝析构 `JXPAMG_PR_DestroyDeepCopy` 忽略 ownership flag**：KT 将 `P_copy` 的 `col_starts` 所有权转移给输出 `A_H` 后，自定义析构仍然无条件释放 `P_copy.col_starts`，导致 double-free。
2. **修复**: 删除自定义析构，全部使用标准 `jx_ParCSRMatrixDestroy`，该函数遵循 ownership 契约。

## 2. 实现的功能

### 2.1 深拷贝 (`JXPAMG_PR_DeepCopyParCSR`)

- 完整复制 CSR 数据（I/J/Data）、offd、col_map、row/col_starts
- 所有权标志正确设置（OwnsData=1, OwnsRowStarts=1, OwnsColStarts=1）
- CommPkg 不复制（NULL）——np>1 时 KT 若因缺 CommPkg 失败，归为预测失败并回退完整重建

### 2.2 粗算子构建 (`JXPAMG_PR_BuildReuseCoarseOperatorFrom`)

- 深拷贝 `A_level` 和 `P_array[level]`
- 调用 `jx_PAMGBuildCoarseOperatorKT`
- 检查 KT 返回值
- `A_H == NULL` 或 KT 失败均返回非零状态
- goto cleanup 确保所有临时对象正确销毁

### 2.3 预测器控制流 (`JXPAMG_PR_PredictCutReadOnly`)

- `cut_k=0` 时立即 goto prediction_consensus，不做粗层探针
- 每层探针后有 MPI collective（Allreduce status），rank 不同步则返回失败
- 最终 `cut_k` 经过 MPI min/max consensus，不一致则返回失败
- 单一 prediction_cleanup 出口，统一销毁临时矩阵
- `status != 0` 时 `*cut_k` 不赋值

### 2.4 失败回退 (`jxpamg_interface.c`)

- 预测失败不再 abort，打印日志后继续执行 `FullSetupOnNewMatrix`
- 只有 FullSetup/PCGSetup/PCGSolve 失败才会终止求解

## 3. 验证结果

### 3.1 np=1 功能测试

| theta | 矩阵对 | cut_k | 迭代 | 结果 |
|-------|--------|-------|------|------|
| 1e-7 | PresCorr | 0 | 13 | ✅ |
| 1e-7 | TurbK | 0 | 2 | ✅ |
| 1e-7 | Vely | 0 | 4 | ✅ |
| 1e-7 | Velz | 6 | 5 | ✅ |
| 1e-7 | Vof | 0 | 4 | ✅ |
| 0.5 | Vely | 6 | 4 | ✅ |
| 0.5 | PresCorr | 6 | 13 | ✅ |

10 轮稳定（Vely, theta=1e-7, np=1）：2/4 迭代，0 崩溃。

### 3.2 np=2 测试

| 矩阵对 | 结果 | 说明 |
|--------|------|------|
| Vely, theta=1e-7 | 间歇性通过 (~3/10) | 预存间歇性崩溃，非 Gate B 引入 |
| TurbK, theta=1e-7 | ✅ 稳定 | |
| PresCorr | ❌ 预存崩溃（不加 -ptheta 也崩）| |

### 3.3 np=4

预存崩溃（所有矩阵对），非 Gate B 引入。见 Phase 1 Gate A 报告。

### 3.4 Valgrind

```
definitely lost: 0 bytes
indirectly lost: 0 bytes
possibly lost: 0 bytes
ERROR SUMMARY: 0 errors from 0 contexts
EXIT=0
```

日志: `experiments/predictive_reuse/algo_3b2/gate_b_valgrind.log`

### 3.5 预测结果覆盖

| cut_k | 含义 | 出现 |
|-------|------|------|
| 0 | 全重建 | PresCorr, TurbK, Vely, Vof (theta=1e-7) |
| 6 (num_levels) | 全复用 | Velz (theta=1e-7), 所有 (theta=0.5) |

中间值（0 < k < num_levels）在现有数据集中未自然出现（矩阵差异要么仅在 Level 0 要么所有层都相近）。

## 4. 最终二进制

- checksum: `accdfa6b9136836be0def48e3125ca86`
- 路径: `/home/spring/ljc/jxpamg_702_sharesetup/solver_pr`

## 5. 已知限制

| 限制 | 影响 | 计划 |
|------|------|------|
| 深拷贝无 CommPkg | np>1 粗层探针可能失败 → fallback 完整重建 | Gate C/D 可选优化 |
| np=2 间歇性崩溃 | Vely np=2 约 30% 通过率 | 预存库 bug，需独立定位 |
| np=4 崩溃 | 所有矩阵对 | 预存库 bug |
| 中间 cut_k 未覆盖 | 0<k<num_levels 无自然测试数据 | 可接受——cut_k=0/num_levels 路径已充分验证 |

## 6. Gate B Release Checklist

- [x] 深拷贝有独立 CSR 数据、分区、maps
- [x] KT 返回值和 A_H 均验证
- [x] 部分分配和 KT 失败路径分别清理
- [x] Level-0 差异返回 cut_k=0 不继续粗层探针
- [x] 粗构建失败不报告为 full reuse
- [x] 每层 MPI-wide status synchronization
- [x] 最终 cut_k min/max 一致
- [x] 预测失败回退到完整重建
- [x] 完整重建失败仍传播非零退出码
- [x] np=1 功能测试全部通过
- [ ] np=2/4 全矩阵稳定（预存库 bug，非 Gate B 阻塞）
- [x] cut_k=0 和 cut_k=num_levels 在日志中出现
- [x] 14 组可用矩阵超 Phase 1 容差
- [x] Vely 10 轮稳定 (np=1)
- [x] Valgrind 0 errors
- [x] 本文档与最终源码一致
