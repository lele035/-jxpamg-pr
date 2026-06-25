# Predictive Reuse 稳定性改造 — 当前状态与已知问题

**日期**: 2026-06-20  
**路径**: `/home/xiyudawang/jxpamg-0220-pr/`

---

## 一、问题背景

在 `-mat2` 共享 AMG 模式下启用 `--predictive-theta` 时，探针全部返回 `sigma=0`（导致误判 full reuse），后续 AMG 层级不一致引发段错误。

## 二、已完成的修复

### 2.1 探针归约缺失（P0 根因）

**文件**: `csrc/amg/par_amg_setup.c`
**函数**: `JXPAMG_PR_DiagRelativeDiff()`
**行号**: ~917

```c
// 原来：只归约 local_base_l1，global_diff_l1 永远是 0
jx_MPI_Allreduce(&local_base_l1, &global_base_l1, 1, JX_MPI_REAL, MPI_SUM, comm);

// 补上：
jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
```

**效果**: sigma 从恒为 0 变为正确值（`5.23e-06 ~ 5.24e-04`）。

### 2.2 Level 0 自比较

**文件**: 同上，行 ~2765

```c
// 原来：两个参数指向同一旧矩阵，永远 sigma=0
JXPAMG_PR_DiagRelativeDiff(A_array[0], saved_A[0], delta);

// 修改：
JXPAMG_PR_DiagRelativeDiff(new_A_fine, saved_A[0], delta);
```

### 2.3 A_H 临时矩阵泄漏

**文件**: 同上，行 ~2780

```c
// 原来：sigma 超阈值时 break，A_H 未销毁
if (sigma > theta_pred) { cut_k = level + 1; break; }

// 修改：
if (sigma > theta_pred) { jx_ParCSRMatrixDestroy(A_H); cut_k = level + 1; break; }
```

### 2.4 Destroy 生命周期修复

**文件**: `src/jxpamg_interface.c` + `src/solver.c`

- **问题**: `-mat2` 模式共享 AMG，case 12 末尾的 `JX_PAMGDestroy(amg_solver)` 提前销毁，导致第二轮野指针。
- **修复**:
  - `jxpamg_interface.c`: 注释全部 7 个 case 的 `JX_PAMGDestroy(amg_solver)`
  - `solver.c`: 在 `MPI_Finalize()` 前加 `if (shared_amg) JX_PAMGDestroy(shared_amg)`
- **验证**: 不加 `--predictive-theta` 时，15/15 组矩阵对正常运行，第二轮 Setup 0.00s（复用成功）。

### 2.5 spmt_rap_type guard 移除

**文件**: `csrc/amg/par_amg_setup_3b2.inc`，行 ~285

删除了 `if (spmt_rap_type == 1)` 条件，使 `interp_type == 0` 时无论 `spmt_rap_type` 均构建插值矩阵。

### 2.6 Makefile 链接修正

**文件**: `Makefile`

```makefile
# 原
JXPAMGDIR = ./jxpamg-0220
# 改
JXPAMGDIR = ./jxpamg-0220-pr
```

---

## 三、本次新增的代码（已编写但未集成）

### 3.1 只读探针 `JXPAMG_PR_PredictCutReadOnly`

**文件**: `csrc/amg/par_amg_setup.c`，行 ~587

- 不修改、不销毁 `A_array`
- 临时 coarse 矩阵用完后全部销毁
- 函数结束时旧 hierarchy 与进入前完全一致
- 保存/恢复 `P_array`/`R_array` 的 `OwnsColStarts`/`OwnsRowStarts` flag，防止探针污染 live AMG
- 返回值表示探针是否成功，`cut_k` 仅用于日志

### 3.2 显式候选矩阵的粗算子构建 `JXPAMG_PR_BuildReuseCoarseOperatorFrom`

**文件**: 同上，行 ~523

与 `JXPAMG_PR_BuildReuseCoarseOperator` 功能相同，但接收显式 `A_level` 参数而非通过 `A_array[level]` 读取，避免替换 live AMG 层级。

### 3.3 完整重建包装 `JXPAMG_PR_FullSetupOnNewMatrix`

**文件**: 同上，行 ~663

销毁旧 `A_array[0]`，清空 smoother/F/U/Vtemp，再调用 `JX_PAMGSetup` 完整重建。

### 3.4 `JXPAMG_PR_AdaptiveReuseAlgorithm` 改造

**文件**: 同上，行 ~2888

改为安全 wrapper：内部调用 `PredictCutReadOnly`，仅记录 cut_k，不执行任何破坏性操作。外层负责完整重建。

---

## 四、模组化文件修改汇总

| 文件 | 本地路径 | 改动数 | 说明 |
|------|---------|--------|------|
| `par_amg_setup.c` | `csrc/amg/` | 10+ | 探针 3 修复 + 3 新函数 + AdaptiveReuseAlgorithm 重写 + 前向声明 |
| `par_amg_setup_3b2.inc` | `csrc/amg/` | 1 | 移除 spmt_rap_type guard |
| `jx_pamg.h` | `include/` | 0（已回滚） | 新函数声明（当前回滚到原版，避免链接影响） |
| `jxpamg_interface.c` | `src/` | 4 | Destroy 注释×7 + 所有权修正 + 统一代码路径 + 参数解析禁用 |
| `solver.c` | `src/` | 1 | shared_amg 销毁上移 |
| `Makefile` | `.` | 1 | JXPAMGDIR 改为 pr |

---

## 五、待解决问题

### 5.1 `--predictive-theta` 参数触发未知崩溃（P0）

**现象**: 
- 命令中含 `--predictive-theta 1e-7` 时，解算正常完成但退出时 glibc 报 `corrupted double-linked list`（SIGABRT）
- **同数量**的 `--dummy 1e-7` 参数完全正常
- 即使将 `--predictive-theta` 的解析禁用（`0 && strcmp(...)`），让二者走**完全相同**的 else 分支，`--predictive-theta` 仍崩、`--dummy` 不崩
- 不加任何额外参数时，所有 15 组矩阵对正常运行

**已排除的原因**:
- 不是代码分支差异（统一代码路径后仍崩）
- 不是参数数量差异（`--dummy 1e-7` 同数量不崩）
- 不是探针调用（探针已跳过隔离测试，仍崩）
- 不是新函数链接（它们未被调用）

**可疑方向**:
- 字符串 `"predictive-theta"` 本身触发了 MPI runtime (hydra) 的特殊行为
- 编译优化导致的二进制布局问题（需 `-g -O0` + valgrind/ASan 排查）

### 5.2 全复用路径崩溃

即使探针正确，`theta=0.5` 时所有 sigma 低于阈值 → full reuse → `A_array` 被替换但 `P/R`/smoother 未更新 → V-cycle 状态不一致 → 负分配崩溃。需两阶段提交 + smoother 重建。

### 5.3 Velz 在 k>0 tail rebuild 崩溃

`JXPAMG_PR_SetupFromLevelExperimental(amg_data, 1, 0)` 预存 bug，非本次改动引入。

---

## 六、验证记录

### 已验证

| 测试 | 命令 | 结果 |
|------|------|------|
| 不加 `--predictive-theta`，15 组 | `solver_pr -sid 12 -mat2` | 15/15 通过 |
| `--predictive-theta 0.0` | 同上 | 正常（跳过自适应路径） |
| 探针 sigma 非零 | `--predictive-theta 1e-7` | sigma = `5.23e-06 ~ 5.24e-04`，正确检出差异 |
| Destroy 生命周期 | `solver_big2 -sid 12 -mat2` | 复用正确，无野指针 |
| `--dummy 1e-7` 同数量参数 | `solver_pr -mat2 --dummy 1e-7` | 正常 |

### 未通过

| 测试 | 问题 |
|------|------|
| `--predictive-theta 1e-7`，任意矩阵对 | 解算完成但退出时 `corrupted double-linked list` SIGABRT |
| `--predictive-theta 0.5`，任意矩阵对 | full reuse 路径负分配崩溃 |

---

## 七、下一步建议

1. 用 `-g -O0` 编译 + valgrind 定位 `--predictive-theta` 字符串触发的堆损坏
2. 或改用非 `--` 前缀参数名（如 `-ptheta`）测试，确定是否为 MPI runtime 干扰
3. 修复全复用路径：为 `JXPAMG_PR_AdaptiveReuseAlgorithm` 的 full reuse 返回前添加完整的层级提交（smoother 重建 + P/R 更新）
4. 修复 `SetupFromLevelExperimental(k=1)` 的 Velz 崩溃（预存 bug）
5. 集成 3B-2 到 `jx_PAMGSetup` 流程
