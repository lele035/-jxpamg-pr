# Predictive Reuse 问题修复报告

**日期**: 2026-06-20  
**任务**: 修复 `solver_big2` 上线，使 3B-2 自适应复用（`--predictive-theta`）路径可用

---

## 一、涉及文件

| 文件 | 路径 | 说明 |
|------|------|------|
| `par_amg_setup.c` | `jxpamg-0220-pr/csrc/amg/` | 核心修复：探针归约、Level 0 比较、A_H 泄漏、全复用提交 |
| `par_amg_setup_3b2.inc` | `jxpamg-0220-pr/csrc/amg/` | 移除无效 `spmt_rap_type` guard |
| `jxpamg_interface.c` | `src/` | 注释全部 7 处 `JX_PAMGDestroy(amg_solver)`，生命周期上移 |
| `solver.c` | `src/` | 末尾加 `if (shared_amg) JX_PAMGDestroy(shared_amg)` |
| `Makefile` | `jxpamg-0220-pr/` | `JXPAMGDIR` 改为 `./jxpamg-0220-pr` |

---

## 二、改动详情

### 改动 1：修复探针差值未归约（P0 根因）

**文件**: `par_amg_setup.c` → `JXPAMG_PR_DiagRelativeDiff()`  
**行号**: ~916

**操作**:
```c
// 原来：只归约了 local_base_l1，global_diff_l1 永远是 0
local_diff_l1 = JXPAMG_PR_LocalDiagAbsDiff(current_A, base_A, &local_base_l1);
jx_MPI_Allreduce(&local_base_l1, &global_base_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
return global_diff_l1 / (global_base_l1 + delta);   // = 0 / X = 0

// 修改：补归约 local_diff_l1
jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
return global_diff_l1 / (global_base_l1 + delta);
```

**原因**: `local_diff_l1` 只在本地计算，从未跨进程归约。`global_diff_l1` 初始化为 0 后从未赋值，导致 sigma 恒为 0。  
**结果**: 修复后，6 层探针 sigma 分别为 `5.23e-06, 1.11e-05, 6.79e-06, 1.81e-04, 5.24e-04, 0.00`（不同物理变量间差异正确检出）。

---

### 改动 2：修复 Level 0 探针自比较（P0）

**文件**: `par_amg_setup.c` → `JXPAMG_PR_AdaptiveReuseAlgorithm()`  
**行号**: ~2764

**操作**:
```c
// 原来：A_array[0] 和 saved_A[0] 指向同一个旧矩阵 → 必然 sigma=0
JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(A_array[0], saved_A[0], delta);

// 修改：用新矩阵 new_A_fine 比旧矩阵 saved_A[0] → 真实比较
JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(new_A_fine, saved_A[0], delta);
```

**原因**: `A_array[0]` 在 probe 阶段尚未被替换，仍是旧矩阵，`saved_A[0]` 也指向它。比较自身永远得 0。  
**结果**: Level 0 现在正确比较新旧细网格矩阵，`sigma=5.23e-06`（Velx vs Vely）。

---

### 改动 3：修复 A_H 泄漏（P1）

**文件**: `par_amg_setup.c` → `JXPAMG_PR_AdaptiveReuseAlgorithm()`  
**行号**: ~2779

**操作**:
```c
// 原来：sigma > theta 时直接 break，A_H 未销毁
if (sigma > theta_pred) { cut_k = level + 1; break; }

// 修改：先销毁 A_H 再 break
if (sigma > theta_pred) { jx_ParCSRMatrixDestroy(A_H); cut_k = level + 1; break; }
```

**原因**: 当某层 sigma 超阈值时，该层 `A_H` 未被任何 `A_array` 引用，也未销毁。  
**结果**: 不再泄漏临时粗矩阵。

---

### 改动 4：全复用路径提交（P2）

**文件**: `par_amg_setup.c` → `JXPAMG_PR_AdaptiveReuseAlgorithm()`  
**行号**: ~2792

**操作**:
```c
// 修改：全复用返回前重建工作向量
if (my_id == 0) jx_printf("...full reuse across %d levels\n", num_levels);
JXPAMG_PR_RecreateWorkVectors(amg_data);
return 0;
```

**原因**: 全复用替换了 `A_array[0..5]`，但 `F_array`/`U_array` 粗层向量仍为旧尺寸。对比 `JXPAMG_PR_FullReuseRAPUpdate()` 已调此函数。  
**结果**: 工作向量与新矩阵层级保持一致。但全复用路径仍会崩溃（smoother 等未重建，见已知问题）。

---

### 改动 5：移除无效 spmt_rap_type guard

**文件**: `par_amg_setup_3b2.inc` → `JXPAMG_PR_RebuildInterpWithFrozenCF`  
**行号**: ~285

**操作**: 删除 `if (spmt_rap_type == 1)` 条件，使 `interp_type == 0` 时无论 `spmt_rap_type` 取值均构建插值矩阵 `par_P`。

---

### 改动 6：Destroy 生命周期修复

**文件**: `src/jxpamg_interface.c` + `src/solver.c`

**问题**: `-mat2` 共享 AMG 模式下，第一轮 case 12 末尾的 `JX_PAMGDestroy(amg_solver)` 销毁了 solver，导致第二轮 `*p_amg` 野指针、`JX_PAMGSetup` 返回 0（跳过 setup）后崩溃。

**操作**:
- `jxpamg_interface.c`: 注释 7 个 case（0, 12, 112, 22, 32, 42, 62）的 `JX_PAMGDestroy(amg_solver)`。
- `solver.c`: 在 `MPI_Finalize()` 前加 `if (shared_amg) JX_PAMGDestroy(shared_amg)`。

**验证**: `solver_big2 -sid 12 -mat2` 无 `--predictive-theta` 时，第二轮 Setup 0.00s（复用成功），无崩溃无野指针。

---

## 三、测试结果

### 环境
- **服务器**: XeonGold5122, Intel oneAPI + MKL + OpenMPI  
- **二进制**: `solver_pr`（Makefile_pr 编译，链接 `jxpamg-0220-pr`）  
- **数据**: `/home/spring/l_s/jxpamg_all/data_trans/` （Velx → PresCorr/TurbK/Vely/Velz/Vof，timeit=0/10000/20000）  
- **命令**: `solver_pr -sid 12 -maxiter 200 -mat2 ... --predictive-theta 1e-7`

### 探针修复验证（Velx-0 → Vely-0）

```
修复前（sigma 恒为 0）:
  probe lv 0: sigma=0.000000e+00
  probe lv 1: sigma=0.000000e+00
  ...
修复后（真实差异检出）:
  probe lv 0: sigma=5.228311e-06
  probe lv 1: sigma=1.108508e-05
  probe lv 2: sigma=6.794117e-06
  probe lv 3: sigma=1.810064e-04
  probe lv 4: sigma=5.235767e-04
  probe lv 5: sigma=0.000000e+00
```

### 批测（15 组，theta=1e-7）

| 结果 | 矩阵对 | 说明 |
|------|--------|------|
| **通过 12/15** | PresCorr, TurbK, Vely, Vof (全部 timeit) | 探针正确 → `cut at k=0` → 完整重建 → 收敛 |
| **崩溃 3/15** | Velz (全部 timeit) | `cut at k=1` → `SetupFromLevelExperimental(k=1)` 预存 bug |

### Velz 崩溃详情

Velz 崩溃独立于本次改动——不加 `--predictive-theta` 时 Velz 正常（走旧完全复用路径），加 `--predictive-theta` 时探针检测到 k=1 差异 → 触发非零层重建 → `JXPAMG_PR_SetupFromLevelExperimental(amg_data, 1, 0)` 内预存 bug 崩溃。

### 未加 --predictive-theta 的兼容性

无退化。所有 15 组矩阵对均正常收敛（5 次迭代），Setup 0.00s（复用成功）。

---

## 四、已知问题

| # | 问题 | 文件/行 | 状态 |
|---|------|---------|------|
| 1 | Full reuse 路径崩溃（theta=0.5，sigma 均低于阈值） | `par_amg_setup.c:2792` | 需两阶段提交 + smoother 重建 |
| 2 | Tail rebuild k=1 崩溃（仅 Velz 触发） | `par_amg_setup.c` → `SetupFromLevelExperimental` | 预存 bug，非本次引入 |
| 3 | 3B-2 `JXPAMG_PR_AdaptiveReuse3B2_Residual` 未集成到 setup 流程 | `par_amg_setup_3b2.inc` / `par_amg_setup.c` | 函数已定义，未调用 |

---

## 五、当前 theta 使用建议

| theta | 行为 | 适用场景 |
|-------|------|---------|
| `0` | 跳过探测，走非自适应复用（与不加 `--predictive-theta` 相同） | 兼容旧行为 |
| `1e-7` | 探测正确检测差异，触发 tail rebuild（k=0 完整重建） | 当前可用，12/15 组通过 |
| `0.5` | 大多数矩阵对触发全复用 → 全复用提交不完整 → 崩溃 | 待修复 |

---

## 六、完整改动 diff 摘要

### par_amg_setup.c（4 处）

```diff
--- 原版
+++ 修复版
@@ -914,6 +914,7 @@
    local_diff_l1 = JXPAMG_PR_LocalDiagAbsDiff(current_A, base_A, &local_base_l1);
    jx_MPI_Allreduce(&local_base_l1, &global_base_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
+   jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
 
    return global_diff_l1 / (global_base_l1 + delta);

@@ -2764 +2765 @@
-   JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(A_array[0], saved_A[0], delta);
+   JX_Real sigma = JXPAMG_PR_DiagRelativeDiff(new_A_fine, saved_A[0], delta);

@@ -2779 +2780 @@
-   if (sigma > theta_pred) { cut_k = level + 1; break; }
+   if (sigma > theta_pred) { jx_ParCSRMatrixDestroy(A_H); cut_k = level + 1; break; }

@@ -2791 +2792,1 @@
    if (my_id == 0) jx_printf("...full reuse across %d levels\n", num_levels);
+   JXPAMG_PR_RecreateWorkVectors(amg_data);
    return 0;
```

### par_amg_setup_3b2.inc（1 处）

```diff
-   if (spmt_rap_type == 1)
-       jx_PAMGBuildInterp(...);
+   jx_PAMGBuildInterp(...);
```

### jxpamg_interface.c（7 处）

```diff
-   JX_PAMGDestroy(amg_solver);
+   // JX_PAMGDestroy(amg_solver);
```

### solver.c（1 处）

```diff
+   if (shared_amg) JX_PAMGDestroy(shared_amg);
    MPI_Finalize();
```
