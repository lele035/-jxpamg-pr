# JXPAMG Predictive Reuse — Phase 1 稳定性基线

**日期**: 2026-06-20  
**状态**: Phase 1 完成  
**路径**: `/home/xiyudawang/jxpamg-0220-pr/`

---

## 概述

在共享 AMG 模式（`-mat2`）下，Phase 1 修复了 `--predictive-theta` 的三个根因问题：

1. **探针 sigma 恒为 0**：`local_diff_l1` 未 `Allreduce`，导致误判 full reuse
2. **case 12 野指针**：`JX_PAMGDestroy(amg_solver)` 提前销毁共享 solver
3. **`AdaptiveReuseAlgorithm` 边探边改**：探针循环内销毁并替换 A_array

并建立稳定的只读探针 + 完整重建流程。

---

## 流程

```
第一次调用 (amg_created=1):
  JX_PAMGSetup(amg_solver, par_matrix)     ← 完整 setup
  JX_PCGSetup + JX_PCGSolve                ← 求解

第二次调用 (amg_created=0):
  PredictCutReadOnly (theta > 0 时)        ← Level 0 探针诊断（只读）
  FullSetupOnNewMatrix                     ← 安全销毁旧状态，完整重建
  JX_PCGSetup + JX_PCGSolve                ← 求解
  每步返回值检查 → 失败立即传播到 solver.c
```

---

## 验证结果

| 测试 | 结果 |
|------|------|
| 14/14 矩阵对（timeit=0/10000/20000） | 全部收敛，0 崩溃 |
| theta=0/1e-7/1e-12/0.5 | 全部完整重建，全部收敛 |
| 10 轮连续（Vely, theta=1e-7） | 一致通过 |
| Valgrind | 0 errors（无泄漏，无非法释放） |
| 错误传播 | `solve_status` 从 case 12 → `JXPAMG_Solver` → `JXPAMG_Solver_Interface` → `solver.c` |

---

## 使用

```bash
# 编译
source /opt/intel/oneapi/setvars.sh --force
make -C jxpamg-0220-pr -j8
make -f Makefile_pr clean && make -f Makefile_pr

# 运行
mpirun -np 1 ./solver_pr \
  A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr \
  --predictive-theta 1e-7
```

参数 `--predictive-theta` 和 `-ptheta` 均可用。

---

## 改动文件

| 文件 | 改动 |
|------|------|
| `csrc/amg/par_amg_setup.c` | Sigma Allreduce、Level 0 探针、A_H 泄漏、`k` 覆盖、`PredictCutReadOnly`、`BuildReuseCoarseOperatorFrom`、`FullSetupOnNewMatrix`（l1_norms 逐层释放）、`AdaptiveReuseAlgorithm` wrapper |
| `src/jxpamg_interface.c` | Destroy 注释×7、所有权修正、双格式参数、Case 12 完整流程、`solve_status` 错误传播 |
| `src/solver.c` | shared_amg 销毁位置、两次调用返回值检查 |
| `include/jx_pamg.h` | PredictCutReadOnly / FullSetupOnNewMatrix 声明 |
| `Makefile` | JXPAMGDIR → pr |
| `csrc/amg/par_amg_setup_3b2.inc` | 无改动 |

---

## 已知限制

- 粗层探针（`jx_PAMGBuildCoarseOperatorKT`）暂不可用 → 总是完整重建
- 仅单进程验证 → np=2/4 待测

---

## 相关文档

- `PHASE1_COMPLETE.md` — 详细技术报告
- `STATUS.md` — 中间状态记录
- `FIX_REPORT.md` — 前期修复详情
- `PHASE1_STABILITY_PLAN.md` — 原始方案
