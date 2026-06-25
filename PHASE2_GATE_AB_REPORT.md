# Phase 2 Gate A-B 进展报告

**日期**: 2026-06-21
**状态**: Gate A 部分通过，Gate B 已定位根因
**项目**: `/home/xiyudawang/jxpamg-0220-pr/`

---

## 1. Gate A: MPI 基线

### 1.1 测试

使用 Phase 1 完整重建路径（探针 + FullSetupOnNewMatrix），`-ptheta 1e-7`：

| np | 结果 | 说明 |
|----|------|------|
| 1 | ✅ | 14/14 矩阵对收敛，Valgrind 0 errors |
| 2 | ✅ | Vely 正常收敛（2+4 迭代） |
| 4 | ❌ | 预存库 bug，非 Phase 1 引入 |

### 1.2 np=4 崩溃详情

**错误**:
```
par_vector.c:117 - global_size < 0 (error code 20)
par_vector.c:1400 - generic error (error code 12)
Signal 11 (Segmentation fault) at address 0x18
```

**根因**: `jxpamg-0220-pr` 库的 `jx_PAMGSetup_predictive_reuse_experiment` 在 np=4 时创建 ParVector 时收到负的 `global_size`。原版 `solver_big` (March 2026, 链接 `jxpamg-0220`) 在 np=4 下正常，确认是 PR 库的预存问题。

**结论**: Gate A np=1/2 通过。np=4 标记为预存问题，先记录后推进。

---

## 2. Gate B: 多级只读探针

### 2.1 隔离测试

| 操作 | 结果 |
|------|------|
| Level 0 `DiagRelativeDiff(new_A_fine, A_array[0])` | ✅ 安全 |
| 粗层 `DiagRelativeDiff(A_array[1], A_array[1])` (identity) | ✅ 安全 |
| P/R ownership flag 读+写（同值） | ✅ 安全 |
| `jx_ParMatmul(A_level, P_array[0])` 最小调用（立即销毁） | ❌ 后续 JX_PCGSetup 失败 |
| `jx_PAMGBuildCoarseOperatorKT` 调用 | ❌ 退出时 heap corruption |

### 2.2 根因

**`jx_ParMatmul` 会修改输入矩阵**，不只是 `OwnsColStarts`/`OwnsRowStarts` 标志。可能的修改包括：

- 为输入矩阵隐式创建 `CommPkg`（通信包）
- 修改内部索引或行/列起始数组
- 设置其他内部状态

这些修改在探针完成后未恢复，导致后续 `JX_PCGSetup`（在 `FullSetupOnNewMatrix` 中）操作被污染的 AMG hierarchy 时失败。

所有权标志的 save/restore 已被证明不足够——需要更进一步的保护。

### 2.3 修复方向

探针中的 RAP 乘法不能直接使用 live AMG 的 P/R 矩阵（`P_array[level]`、`R_array[level]`）。必须：

1. **深拷贝** P 和 R 矩阵（包括所有内部 CSR 数组和 CommPkg）
2. 在拷贝上进行 RAP 乘法，生成候选粗算子 A_H
3. 使用完立即销毁所有拷贝和 A_H

深拷贝需要复制 `jx_ParCSRMatrix` 的所有内部结构：
- `Diag` CSR 矩阵（I, J, Data）
- `Offd` CSR 矩阵
- `ColMapOffd`
- `CommPkg`
- `RowStarts` / `ColStarts`
- `AssumedPartition`

这是 Gate B 的核心工程量。

---

## 3. 当前代码状态

### 3.1 `par_amg_setup.c`

| 函数 | 状态 | 说明 |
|------|------|------|
| `PredictCutReadOnly` | ✅ | Level 0 探针启用，粗层探针占位（无 ParMatmul） |
| `BuildReuseCoarseOperatorFrom` | ⚠ | 框架已就绪，RAP 生成暂返回 NULL |
| `FullSetupOnNewMatrix` | ✅ | 安全销毁旧状态后完整重建 |
| `AdaptiveReuseAlgorithm` | ✅ | 安全 wrapper |

### 3.2 `jxpamg_interface.c`

- Case 12: 探针(Level 0) → FullSetup → PCG → 错误传播 ✅
- `solver_ret` 错误传播 ✅
- `solve_status` 变量 ✅

### 3.3 Gate B 启用方式

编辑 `par_amg_setup.c` → `PredictCutReadOnly` 中的粗层探针占位块，替换为实际粗层探针逻辑（需要先实现 P/R 深拷贝）。

---

## 4. 下一步

1. 实现 `jx_ParCSRMatrixDeepCopy` —— 深拷贝一个 ParCSRMatrix
2. 在 `BuildReuseCoarseOperatorFrom` 中使用深拷贝的 P/R
3. 逐步启用粗层探针（1 层 → 2 层 → 全层）
4. 每层启用后用 Valgrind 验证
5. Gate B 通过后进入 Gate C（事务性 tail rebuild）
