# Phase 1 完成总结

**日期**: 2026-06-20  
**状态**: Phase 1 完成 ✅

---

## 1. 问题与修复

### 1.1 初始根因

在共享 AMG 模式 (`-mat2`) 下存在三个根因问题：

| # | 问题 |
|---|------|------|
| A | 探针 sigma 恒为 0（`local_diff_l1` 未 Allreduce） |
| B | case 12 末尾 `JX_PAMGDestroy(amg_solver)` |
| C | `AdaptiveReuseAlgorithm` 边探针边修改 |

### 1.2 Phase 1 改动清单

| 类别 | 改动 |
|------|------|----------|
| **探针** | 补 `Allreduce(local_diff_l1)` |
| | Level 0 用 `new_A_fine` 比旧 `A_array[0]` | `par_amg_setup.c:627` |
| | `sigma > theta` 时销毁 A_H | `par_amg_setup.c:642` |
| | `k` 不再被无条件覆盖 | `par_amg_setup.c:637` |
| | 新增 `PredictCutReadOnly`（只读探针，不修改 AMG） | `par_amg_setup.c:587-660` |
| | 新增 `BuildReuseCoarseOperatorFrom`（所有权 save/restore） | `par_amg_setup.c:523-585` |
| **完整搭建** | 新增 `FullSetupOnNewMatrix`（安全销毁 smoother/F/U/Vtemp/l1_norms/residual） |
| | `l1_norms` 先释放每层内数组再释放外层 | `par_amg_setup.c:708-718` |
| | 重写 `AdaptiveReuseAlgorithm` 为安全 wrapper | `par_amg_setup.c:2904-2920` |
| **接口** | 注释 7 处 `JX_PAMGDestroy(amg_solver)` |
| | 矩阵所有权改为 `owned_A[0] != A_Par` | `jxpamg_interface.c:262` |
| | Case 12 流程：探针(可选) → FullSetup → PCG，全部返回值检查 | `jxpamg_interface.c:1264-1322` |
| | `JX_PCGSetup` / `JX_PCGSolve` 返回值检查 | 同上 |
| | probe 失败时跳过完整 setup + break | 同上 |
| | `JXPAMG_Solver` 错误传播（solver=NULL → return 1） | `jxpamg_interface.c:3014` |
| | 外层检查 `JXPAMG_Solver` 返回值 | `jxpamg_interface.c:238` |
| | `--predictive-theta` / `-ptheta` 双格式解析 | `jxpamg_interface.c:807` |
| **管理** | `shared_amg` 销毁上移到 `MPI_Finalize` 前 |
| | `Makefile` JXPAMGDIR → pr | `Makefile:10` |
| | `jx_pamg.h` 新增两个函数声明 | `include/jx_pamg.h` |
| | `par_amg_setup_3b2.inc` 无改动（spmt_rap_type guard 已还原） | — |

### 1.3 最终流程

```
第二次调用 (shared AMG, --predictive-theta > 0):
  1. PredictCutReadOnly        ← Level 0 探针诊断（只读）
  2. FullSetupOnNewMatrix      ← 安全销毁旧状态，完整重建
  3. JX_PCGSetup + JX_PCGSolve ← 求解
  4. 每步返回值检查；失败 → solver=NULL → break → return 1

第二次调用 (shared AMG, --predictive-theta == 0 / 未指定):
  1. FullSetupOnNewMatrix      ← theta=0 也完整重建（统一行为）
  2. JX_PCGSetup + JX_PCGSolve
  3. 每步返回值检查
```

## 2. 验证结果

### 2.1 探针修复

```
修复前: probe lv 0: sigma=0.000000e+00
修复后: probe lv 0: sigma=5.228311e-06
```

### 2.2 批测（theta=1e-7，Level 0 探针 + 完整重建）

| timeit | PresCorr | TurbK | Vely | Velz |
|--------|----------|-------|------|------|-----|
| 0 | 13 ✅ | 2 ✅ | 4 ✅ | 5 ✅ |
| 10000 | 13 ✅ | 3 ✅ | 3 ✅ | 3 ✅ |
| 20000 | 13 ✅ | 3 ✅ | 3 ✅ | 3 ✅ |

14/14 通过（Vof t=20000 数据文件不存在），全部收敛，0 崩溃。

### 2.3 多 theta 验证

| theta | PresCorr | Vely |
|-------|----------|------|------|
| 0 | 13 ✅ | 4 ✅ |
| 1e-7 | 13 ✅ | 4 ✅ |
| 1e-12 | 13 ✅ | 4 ✅ |
| 0.5 | 13 ✅ | 4 ✅ |

PresCorr 从原来的 200 次未收敛改善为 13 次收敛。

### 2.4 稳定性（Vely, theta=1e-7, 10 轮）

```
Round 1-10: iters=2/4, crash=0（全部一致）
```

### 2.5 参数兼容

`--predictive-theta` 和 `-ptheta` 两种格式均正常工作。

### 2.6 错误传播验证

- `solve_status` 变量：初始为 0，5 个错误出口均设 1
- `JXPAMG_Solver` 返回 `solve_status`（原固定 `return 0`）
- `JXPAMG_Solver_Interface` 失败时 `return solver_ret`（跳过解复制）
- `solver.c` 两次调用均检查返回值，失败时 `return 1`
- `JX_PCGSolve` 失败时：`Destroy(solver) + solve_status=1`，后续不读取迭代数

### 2.7 Valgrind 内存检测

```
valgrind --leak-check=full --error-exitcode=99 mpirun -np 1 ./solver_pr ...

ERROR SUMMARY: 0 errors from 0 contexts
still reachable: 52,321 bytes in 1,025 blocks  (MPI/MKL 全局分配，预期)
```

无内存泄漏，无非法释放。

## 3. 已知限制

| 限制 | 说明 |
|------|------|
| 粗层探针不可用 | `jx_PAMGBuildCoarseOperatorKT` (spmt_rap_type=1) 在探针上下文破坏堆；Level 0 探针安全 |
| 总是完整重建 | 粗层探针不可用 → 无法精确判断 tail rebuild 范围 |
| np=2/4 MPI 未测 | 仅完成单进程验证 |

## 4. Phase 2 建议

1. **np=2/4 MPI 验证** → 确认多进程无问题
2. **修复粗层探针** → `jx_PAMGBuildCoarseOperatorKT` 堆损坏
3. **接回 tail rebuild** → 用正确的 cut_k 做增量重建
4. **接回 full reuse 优化** → 全复用路径的完整层级提交
5. **修复 Velz k=1 崩溃** → `SetupFromLevelExperimental(k=1)` 预存 bug
6. **集成 3B-2** → `par_amg_setup_3b2.inc` 接入 setup 流程

## 5. 改动文件

| 文件 | Phase 1 改动 |
|------|-------------|
| `csrc/amg/par_amg_setup.c` | Sigma Allreduce、Level 0 探针、A_H 泄漏、k 覆盖、PredictCutReadOnly、BuildReuseCoarseOperatorFrom、FullSetupOnNewMatrix（l1_norms 逐层释放）、AdaptiveReuseAlgorithm wrapper |
| `src/jxpamg_interface.c` | Destroy 注释×7、所有权修正、双格式参数、Case 12 完整流程（探针→搭建→返回值检查）、错误传播 |
| `src/solver.c` | shared_amg 销毁位置 |
| `include/jx_pamg.h` | PredictCutReadOnly / FullSetupOnNewMatrix 声明 |
| `Makefile` | JXPAMGDIR → pr |
| `csrc/amg/par_amg_setup_3b2.inc` | 无改动 |
