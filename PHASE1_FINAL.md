# Predictive Reuse Phase 1 — 完成总结

**日期**: 2026-06-20
**状态**: 完成
**项目**: JXPAMG PR 库 (`jxpamg-0220-pr`)
**路径**: `/home/xiyudawang/jxpamg-0220-pr/`

---

## 1. 做了什么

共享 AMG 模式 (`-mat2`) 下 `--predictive-theta` 存在三个根因问题，Phase 1 全部修复：

| # | 问题 | 修复 |
|---|------|------|
| 1 | 探针 sigma 恒为 0（`local_diff_l1` 未 Allreduce） | 补 `jx_MPI_Allreduce(&local_diff_l1, &global_diff_l1, ...)` |
| 2 | 探针 Level 0 自比较（新旧矩阵同一个指针） | 用 `new_A_fine` 替代 `A_array[0]` |
| 3 | `AdaptiveReuseAlgorithm` 边探边改 | 重写为只读探针 + 独立完整重建流程 |

## 2. 最终流程

```
第一次调用 (amg_created=1):
  JX_PAMGSetup → JX_PCGSetup → JX_PCGSolve

第二次调用 (amg_created=0):
  PredictCutReadOnly (theta > 0)      ← Level 0 只读探针
  FullSetupOnNewMatrix                ← 安全销毁旧状态 + 完整重建
  JX_PCGSetup → JX_PCGSolve           ← 求解
  solve_status 错误传播链             ← 每步返回值检查
```

成功路径按以上执行。任意步骤失败：
- case 12: `solve_status = 1; solver = NULL; break`
- `JXPAMG_Solver`: `return solve_status`
- `JXPAMG_Solver_Interface`: 跳过解复制，走统一清理
- `solver.c`: `goto cleanup` → `JX_PAMGDestroy + MPI_Finalize`

## 3. 验证结果

### 3.1 功能测试

| theta | 矩阵对 (14/14) | 结果 |
|-------|---------------|------|
| 0 | PresCorr(13), TurbK(2), Vely(4), Velz(5), Vof(4) | 全部收敛 |
| 1e-7 | 同上 | 全部收敛 |
| 1e-12 | PresCorr(13), Vely(4) | 收敛 |
| 0.5 | PresCorr(13), Vely(4) | 收敛 |

10 轮连续（Vely, theta=1e-7）：全部收敛 2/4 迭代，0 崩溃。

### 3.2 内存检测

- **二进制 checksum**: `fc6cc5781d5380e96bee9534aa58d28a`
- **Valgrind**: `0 errors from 0 contexts`，exit code 0
- **日志**: `valgrind_test.log`（本地）

**解算输出**（正常收敛）:
```
iteration = 2, residual = 5.158317e+00, RES = 4.047409e-07
[predictive-reuse] probe lv 0: sigma=5.228311e-06
[predictive-reuse] probe ok, predicted cut_k=0
[predictive-reuse] full setup completed
iteration = 4, residual = 4.225773e-03, RES = 9.420829e-07
```

## 4. 改动文件

| 文件 | 改动 |
|------|------|
| `csrc/amg/par_amg_setup.c` | Sigma Allreduce 修复、Level 0 探针修复、A_H 泄漏修复、`k` 覆盖修复、`PredictCutReadOnly`、`BuildReuseCoarseOperatorFrom`、`FullSetupOnNewMatrix`（l1_norms 逐层释放 + smoother/F/U/Vtemp 安全销毁）、`AdaptiveReuseAlgorithm` 安全 wrapper |
| `src/jxpamg_interface.c` | Destroy 注释x7、所有权检查修正、`--predictive-theta`/`-ptheta` 双格式参数、Case 12 完整流程、`solve_status` 错误传播、失败跳过解复制 |
| `src/solver.c` | shared_amg 销毁上移、两次调用返回值检查、`goto cleanup` 统一清理 |
| `include/jx_pamg.h` | PredictCutReadOnly / FullSetupOnNewMatrix 声明 |
| `Makefile` | JXPAMGDIR = ./jxpamg-0220-pr |
| `csrc/amg/par_amg_setup_3b2.inc` | 无改动 |

## 5. 使用

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

`--predictive-theta` / `-ptheta` 均可。

## 6. 已知限制

- 粗层探针不可用（`jx_PAMGBuildCoarseOperatorKT` 堆损坏），仅 Level 0 探针工作
- 总是完整重建（粗层探针不可用导致无法判断 tail rebuild 范围）
- 仅单进程验证

## 7. 相关文档

| 文件 | 说明 |
|------|------|
| `PHASE1_COMPLETE.md` | 详细技术报告 |
| `PHASE1_DETAILED_REPORT.md` | 逐 bug 调查记录 |
| `PHASE1_STABILITY_PLAN.md` | 原始方案 |
| `STATUS.md` | 中间状态 |
| `FIX_REPORT.md` | 前期修复详情 |
| `valgrind_test.log` | Valgrind 完整日志 |
