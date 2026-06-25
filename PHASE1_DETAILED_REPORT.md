# Predictive Reuse Phase 1 — 详细技术报告

**日期**: 2026-06-20  
**作者**: 自动生成  
**路径**: `/home/xiyudawang/jxpamg-0220-pr/`  
**服务器**: `spring`（172.16.40.26），`/home/spring/ljc/jxpamg_702_sharesetup/`

---

## 目录

1. [背景与目标](#1-背景与目标)
2. [Bug 调查与根因分析](#2-bug-调查与根因分析)
3. [代码改动清单](#3-代码改动清单)
4. [新增函数设计](#4-新增函数设计)
5. [接口集成方案](#5-接口集成方案)
6. [测试结果](#6-测试结果)
7. [已知限制与待修复](#7-已知限制与待修复)
8. [编译与运行](#8-编译与运行)
9. [文件索引](#9-文件索引)

---

## 1. 背景与目标

### 1.1 问题场景

JXPAMG PR 库（`jxpamg-0220-pr`）实现了一套 Predictive Reuse 机制：在 `-mat2` 共享 AMG 模式下，用第一个矩阵（Velx）的 AMG hierarchy 加速第二个矩阵（Vely/PresCorr/TurbK/Velz/Vof）的求解。

调用路径：
```
solver.c → JXPAMG_Solver_Interface(argc, argv, ..., &shared_amg)
         ├── 第一次调用：创建 AMG，完整 setup，求解 Velx
         └── 第二次调用：复用 shared_amg，求解 Vely/...
```

### 1.2 两个原始 bug

| Bug | 触发条件 | 现象 |
|-----|---------|------|
| A. 野指针崩溃 | case 12 末尾 `JX_PAMGDestroy(amg_solver)` 销毁共享 solver | 第二轮 `*p_amg` 野指针，`JX_PAMGSetup` 返回 0（跳过 setup）→ 崩溃 |
| B. 探针 sigma 恒为 0 | `--predictive-theta 0.5` 触发全复用路径 | `local_diff_l1` 未 `Allreduce` → `global_diff_l1` 永远为 0 → 误判 full reuse → 替换 A_array 但未更新 P/R/smoother → V-cycle 负分配崩溃 |

### 1.3 Phase 1 目标

1. 修复 Bug A 和 Bug B
2. 保证任何 `theta` 值都不破坏 AMG hierarchy
3. 探针只读（不修改 live AMG），用于诊断记录
4. 不启用 k>0 tail rebuild 和 full reuse 性能路径
5. 所有矩阵对无崩溃

---

## 2. Bug 调查与根因分析

### 2.1 Bug A: 野指针崩溃

**位置**: `src/jxpamg_interface.c` case 12

旧代码在求解完成后无条件调用 `JX_PAMGDestroy(amg_solver)`。在 `-mat2` 共享模式下，第一次调用创建的 AMG 被销毁，`*p_amg` 成为野指针。第二次调用时 `JX_PAMGSetup` 检测到 AMG 已初始化但内部数据已被释放，跳过 setup，后续 CG 求解崩溃。

**修复**: 注释 7 个 case（0, 12, 112, 22, 32, 42, 62）的 `JX_PAMGDestroy(amg_solver)`，生命周期管理上移到 `solver.c`，在 `MPI_Finalize()` 前统一调用 `JX_PAMGDestroy(shared_amg)`。

### 2.2 Bug B: 探针 sigma 恒为 0

**位置**: `csrc/amg/par_amg_setup.c` → `JXPAMG_PR_DiagRelativeDiff()`

```c
// 行 ~917: bug 所在
local_diff_l1 = JXPAMG_PR_LocalDiagAbsDiff(current_A, base_A, &local_base_l1);
jx_MPI_Allreduce(&local_base_l1, &global_base_l1, 1, JX_MPI_REAL, MPI_SUM, comm);
// 缺失: jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, ...)
return global_diff_l1 / (global_base_l1 + delta);  // = 0 / X = 0 永远
```

`local_diff_l1` 只在局部计算，从未跨进程归约，而 `global_diff_l1` 初始化后未赋值。修复后 sigma 正确为 `5.23e-06 ~ 5.24e-04`。

**关联发现**: Level 0 探针还存在自比较问题：
```c
// 行 ~2764: 两个参数指向同一旧矩阵 → 必然 sigma=0
JXPAMG_PR_DiagRelativeDiff(A_array[0], saved_A[0], delta);
// 修复: 用 new_A_fine 比旧矩阵
JXPAMG_PR_DiagRelativeDiff(new_A_fine, saved_A[0], delta);
```

### 2.3 A_H 泄漏

**位置**: 同上，行 ~2779

探针循环中 `sigma > theta_pred` 时 `break`，已分配的 `A_H` 未销毁。修复：
```c
if (sigma > theta_pred) { jx_ParCSRMatrixDestroy(A_H); cut_k = level + 1; break; }
```

### 2.4 `--predictive-theta` 被 MPI 拦截

**定位过程**:

| 测试 | 结果 |
|------|------|
| `--predictive-theta 1e-7` via mpirun | SIGABRT（退出时 corrupted linked list） |
| 同数量 `--dummy 1e-7` | 正常 |
| 无任何额外参数 | 正常 |
| `--predictive-theta 1e-7` via mpirun，参数解析 `0 &&` 禁用 | 仍崩 |
| `--predictive-theta 1e-7` 直接执行（无 mpirun） | 正常 |
| `-ptheta 1e-7` via mpirun | 正常 |
| `--ptheta 1e-7` via mpirun | 正常 |

**结论**: Intel MPI hydra（`mpirun`）对 `--predictive-*` 前缀有特殊处理，导致传递给应用的运行时状态异常。将参数名改为 `-ptheta` 解决。

### 2.5 粗层探针 RAP 堆损坏

**定位过程**:

| 测试 | 结果 |
|------|------|
| 探针全部跳过 | 正常 |
| 仅 Level 0 探针（`DiagRelativeDiff`） | 正常 |
| 仅 P/R ownership flag 读写 | 正常 |
| Level 0 + 1 层 RAP（`BuildReuseCoarseOperatorFrom`） | SIGABRT |

**根因**: `spmt_rap_type` 默认为 1 → `jx_PAMGBuildCoarseOperatorKT` 被调用 → 内部堆分配/释放顺序与后续 AMG 清理冲突 → 退出时 `corrupted double-linked list`。

`BuildReuseCoarseOperatorFrom` 已实现 ownership flag save/restore，但 RAP 乘法内部的堆操作无法从外部保护。

---

## 3. 代码改动清单

### 3.1 `csrc/amg/par_amg_setup.c`（核心文件）

| 行号 | 改动 | 类型 |
|------|------|------|
| ~593 | 添加 `DiagRelativeDiff` / `LocalDiagAbsDiff` 前向声明 | 编译修复 |
| ~917 | 补 `Allreduce(local_diff_l1, &global_diff_l1)` | P0 bug fix |
| ~2765 | Level 0 探针 `A_array[0]` → `new_A_fine` | P0 bug fix |
| ~2780 | sigma 超阈值时 `Destroy(A_H)` | P1 leak fix |
| ~523-585 | 新增 `BuildReuseCoarseOperatorFrom`（所有权 flag save/restore） | 新功能 |
| ~587-660 | 新增 `PredictCutReadOnly`（Level 0 探针 + 粗层禁用的占位） | 新功能 |
| ~662-685 | 新增 `FullSetupOnNewMatrix`（非 static，含 smoother/F/U/Vtemp 清理） | 新功能 |
| ~2888-2910 | 重写 `AdaptiveReuseAlgorithm` 为安全 wrapper | 重构 |

### 3.2 `csrc/amg/par_amg_setup_3b2.inc`

| 行号 | 改动 |
|------|------|
| ~285 | 删除 `if (spmt_rap_type == 1)` guard，使 interp_type=0 时始终构建 par_P |

### 3.3 `include/jx_pamg.h`

```c
// 新增声明
JX_Int JXPAMG_PR_PredictCutReadOnly(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Real, JX_Int *);
JX_Int JXPAMG_PR_FullSetupOnNewMatrix(jx_ParAMGData *, jx_ParCSRMatrix *);
```

### 3.4 `src/jxpamg_interface.c`

| 行号 | 改动 |
|------|------|
| ~807 | `--predictive-theta` → `-ptheta`，`--predictive-lreuse` → `-plreuse` |
| ~1178, 1296, 1390, 1730, 2623, 2832, 2930 | 注释 `JX_PAMGDestroy(amg_solver)` |
| ~236 | 删除 `old_amg` 变量 |
| ~262-270 | 矩阵所有权改为 `owned_A[0] != A_Par` 检查 |
| ~1262-1282 | Case 12 改为：Level 0 只读探针 → 复用旧 AMG（Phase 1 基线） |

### 3.5 `src/solver.c`

| 行号 | 改动 |
|------|------|
| ~301 | `MPI_Finalize()` 前加 `if (shared_amg) JX_PAMGDestroy(shared_amg)` |

### 3.6 `Makefile`

```makefile
JXPAMGDIR = ./jxpamg-0220-pr    # 原为 ./jxpamg-0220
```

---

## 4. 新增函数设计

### 4.1 `JXPAMG_PR_BuildReuseCoarseOperatorFrom`

```c
static jx_ParCSRMatrix *
JXPAMG_PR_BuildReuseCoarseOperatorFrom(
    jx_ParAMGData *amg_data,
    JX_Int level,
    jx_ParCSRMatrix *A_level,    // ← 显式传入，不读 A_array[level]
    JX_Int *opt_icor);
```

与 `JXPAMG_PR_BuildReuseCoarseOperator` 功能相同，但接收显式 `A_level`。

**保护措施**: RAP 操作前保存 `OwnsColStarts(P_array[level])` 和 `OwnsRowStarts(R_array[level])`，操作后恢复。防止探针污染 live AMG hierarchy。

**当前状态**: 函数正确实现，但调用方（粗层探针）因 RAP 堆损坏已禁用。

### 4.2 `JXPAMG_PR_PredictCutReadOnly`

```c
JX_Int
JXPAMG_PR_PredictCutReadOnly(
    jx_ParAMGData *amg_data,
    jx_ParCSRMatrix *new_A_fine,
    JX_Real theta_pred,
    JX_Int Lreuse,
    JX_Real delta,
    JX_Int *cut_k);
```

**保证**:
- 不修改、不销毁 `A_array`
- 不调用 `SetupFromLevelExperimental()`
- 不调用 `RecreateWorkVectors()`
- 函数结束时旧 hierarchy 与进入前完全一致
- `*cut_k` 仅用于诊断日志

**当前实现**: Level 0 探针（`DiagRelativeDiff`）已启用，粗层探针已禁用（占位 + TODO 注释）。

### 4.3 `JXPAMG_PR_FullSetupOnNewMatrix`

```c
JX_Int
JXPAMG_PR_FullSetupOnNewMatrix(
    jx_ParAMGData *amg_data,
    jx_ParCSRMatrix *new_A);
```

**流程**:
1. 若 `A_array[0] != new_A`，销毁旧 `A_array[0]`
2. 清空 `Smoother`/`F_array`/`U_array`/`Vtemp`（让 JX_PAMGSetup 完整重建）
3. 调用 `JX_PAMGSetup(amg_data, new_A)`

**当前状态**: 完整实现但未启用——共享 AMG 上调 JX_PAMGSetup 与粗层探针同源堆损坏。

### 4.4 `JXPAMG_PR_AdaptiveReuseAlgorithm`（重写版）

改为安全 wrapper：内部调用 `PredictCutReadOnly`，仅记录 cut_k，不执行任何破坏性操作。由外层决定后续重建策略。

---

## 5. 接口集成方案

### 5.1 Case 12 当前流程（Phase 1）

```
第一次调用 (amg_created=1):
  JX_PCGSetPrecond(solver, JX_PAMGPrecond, JX_PAMGSetup, amg_solver)
  JX_PAMGSetup(amg_solver, par_matrix)          ← 完整 setup
  JX_PCGSetup(solver, ...)
  JX_PCGSolve(solver, ...)

第二次调用 (amg_created=0, predictive_theta > 0):
  JXPAMG_PR_PredictCutReadOnly(...)             ← Level 0 探针（诊断）
  JX_PCGSetPrecond(solver, ...)                 ← 复用旧 AMG
  JX_PCGSetup(solver, ...)
  JX_PCGSolve(solver, ...)
```

### 5.2 计划中的 Phase 2 流程（待粗层探针修复后）

```
第二次调用 (amg_created=0, predictive_theta > 0):
  JXPAMG_PR_PredictCutReadOnly(...)             ← 完整探针（Level 0 + 粗层）
  JXPAMG_PR_FullSetupOnNewMatrix(...)           ← 总是完整重建
  JX_PCGSetPrecond(solver, ...)
  JX_PCGSetup(solver, ...)
  JX_PCGSolve(solver, ...)
```

### 5.3 矩阵所有权修正

旧代码依赖 `old_amg == NULL` 判断：
```c
// 旧逻辑
if (old_amg == NULL && *p_amg != NULL) { /* 不销毁 */ }
else { jx_ParCSRMatrixDestroy(A_Par); }
```

新代码检查 AMG 是否持有矩阵：
```c
// 新逻辑
jx_ParCSRMatrix **owned_A = NULL;
if (*p_amg != NULL)
    owned_A = jx_ParAMGDataAArray((jx_ParAMGData *)*p_amg);
if (owned_A == NULL || owned_A[0] != A_Par)
    jx_ParCSRMatrixDestroy(A_Par);
```

---

## 6. 测试结果

### 6.1 环境

- **CPU**: Xeon Gold 5122
- **MPI**: Intel oneAPI MPI 2021.6.0, np=1
- **编译**: `mpiicc -O2`, `JX_Real = long double`, `JX_USING_OPENMP=0`
- **数据**: Velx → PresCorr/TurbK/Vely/Velz/Vof, timeit=0
- **参数**: `-sid 12 -maxiter 200 -mat2 A2.csr -rhs2 b2.csr -ptheta <value>`

### 6.2 探针修复验证

**修复前**（sigma 恒为 0）:
```
probe lv 0: sigma=0.000000e+00
probe lv 1: sigma=0.000000e+00
...
```

**修复后**（Level 0）:
```
probe lv 0: sigma=5.228311e-06
```

### 6.3 批测结果（theta=1e-7）

| 矩阵对 | 结果 | 第二轮迭代数 | 说明 |
|--------|------|-------------|------|
| Velx → TurbK | ✅ | 5 | 收敛 |
| Velx → Vely | ✅ | 5 | 收敛 |
| Velx → Velz | ✅ | 5 | 收敛 |
| Velx → PresCorr | ※ | 200 | 未收敛（复用旧 AMG 精度不足，耗时 68s） |
| Velx → Vof | ※ | 200 | 未收敛（复用旧 AMG 精度不足，耗时 78s） |

※ = 非崩溃，解算正常完成

### 6.4 theta=0.5 验证

| 矩阵对 | 结果 |
|--------|------|
| Velx → Vely | ✅ 无崩溃，收敛正常 |

### 6.5 未加 -ptheta 时的兼容性

所有 15 组（timeit=0/10000/20000 × 5 变量）正常通过，无退化。

---

## 7. 已知限制与待修复

| # | 问题 | Phase | 说明 |
|---|------|-------|------|
| 1 | 粗层探针 RAP 堆损坏 | 2 | `jx_PAMGBuildCoarseOperatorKT` 需改为不破坏堆 |
| 2 | FullSetupOnNewMatrix 崩溃 | 2 | 同源 bug |
| 3 | PresCorr/Vof 未收敛 | 2 | 完整重建后可收敛 |
| 4 | Velz tail rebuild k=1 崩溃 | 2 | `SetupFromLevelExperimental(k=1)` 预存 bug |
| 5 | np≥2 MPI 验证 | 2 | 尚未测试 |
| 6 | 3B-2 集成 | 2 | `par_amg_setup_3b2.inc` 已定义但未接入 setup 流程 |

---

## 8. 编译与运行

### 8.1 编译

```bash
# 1. 编译 PR 库
source /opt/intel/oneapi/setvars.sh --force
cd /home/spring/ljc/jxpamg_702_sharesetup
make -C jxpamg-0220-pr -j8

# 2. 编译 solver_pr
make -f Makefile_pr clean
make -f Makefile_pr -j8

# 3. 编译 solver_big2（OpenMP 版本）
make -f Makefile clean
make -f Makefile -j8
```

### 8.2 运行

```bash
# 无探针（复用旧 AMG）
mpirun -np 1 ./solver_pr \
  A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr

# Level 0 探针诊断
mpirun -np 1 ./solver_pr \
  A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr \
  -ptheta 1e-7

# 直接执行（无 mpirun）
./solver_pr A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr \
  -ptheta 0.5
```

### 8.3 新增参数

| 参数 | 说明 |
|------|------|
| `-ptheta <value>` | 探针阈值（0 跳过探针，1e-7 推荐值，0.5 大阈值） |
| `-plreuse <value>` | 最大探针层数（-1 全部） |

---

## 9. 文件索引

### 本地（`/home/xiyudawang/jxpamg-0220-pr/`）

| 文件 | 行数 | 说明 |
|------|------|------|
| `csrc/amg/par_amg_setup.c` | ~2940 | **核心**：探针修复 + 3 新函数 + AdaptiveReuseAlgorithm 重写 |
| `csrc/amg/par_amg_setup_3b2.inc` | 518 | 3B-2 算法实现（spmt_rap_type 修复） |
| `include/jx_pamg.h` | 1615 | 新函数声明（PredictCutReadOnly, FullSetupOnNewMatrix） |
| `src/jxpamg_interface.c` | 3242 | Case 12 流程 + Destroy 修复 + 所有权修正 + 参数重命名 |
| `src/solver.c` | ~305 | shared_amg 生命周期管理 |
| `Makefile` | ~80 | JXPAMGDIR → pr |
| `Makefile_pr` | ~45 | solver_pr 专用 Makefile |
| `FIX_REPORT.md` | 200+ | 前期修复报告 |
| `STATUS.md` | 120+ | 完整状态文档 |
| `PHASE1_STABILITY_PLAN.md` | 160+ | Phase 1 方案 |
| `PHASE1_DETAILED_REPORT.md` | — | 本文档 |
| `experiments/predictive_reuse/algo_3b2/BUG_FULL_REUSE_CRASH.md` | 79 | 全复用崩溃 bug 文档 |

### 服务器

| 路径 | 说明 |
|------|------|
| `/home/spring/ljc/jxpamg_702_sharesetup/jxpamg-0220-pr/` | PR 库（服务器副本） |
| `/home/spring/ljc/jxpamg_702_sharesetup/src/jxpamg_interface.c` | 接口文件 |
| `/home/spring/ljc/jxpamg_702_sharesetup/src/solver.c` | 求解器主程序 |
| `/home/spring/ljc/jxpamg_702_sharesetup/Makefile` | 主 Makefile（solver_big2） |
| `/home/spring/ljc/jxpamg_702_sharesetup/Makefile_pr` | PR Makefile（solver_pr） |
| `/home/spring/ljc/jxpamg_702_sharesetup/solver_pr` | 编译产物（无 OpenMP） |
| `/home/spring/ljc/jxpamg_702_sharesetup/solver_big2` | 编译产物（OpenMP） |
| `/home/spring/l_s/jxpamg_all/data_trans/` | 测试数据 |
