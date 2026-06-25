# JXPAMG Predictive Reuse — 进展总结

**日期**: 2026-06-21
**项目**: `/home/xiyudawang/jxpamg-0220-pr/`
**服务器**: `spring` (172.16.40.26)

---

## 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1 | ✅ 完成 | 稳定性基线：只读探针 + 完整重建 + 错误传播 |
| Phase 2 Gate A | ✅ np=1/2 通过 | MPI 基线；np=4 为预存库 bug |
| Phase 2 Gate B | ✅ 完成 | 多级只读探针：6 层全通，Valgrind 0 errors |
| Phase 2 Gate C | 待开始 | 事务性 tail rebuild |

---

## Phase 1: 稳定性基线

修复了三个根因问题：

1. **探针 sigma 恒为 0** — `local_diff_l1` 未 `Allreduce`
2. **case 12 野指针** — `JX_PAMGDestroy` 提前销毁共享 solver
3. **`AdaptiveReuseAlgorithm` 边探边改** — 重写为只读探针

最终流程：
```
第二次调用 → PredictCutReadOnly(Level 0) → FullSetupOnNewMatrix → PCG → solve_status 传播
```

验证：14/14 矩阵对收敛，Valgrind 0 errors，10 轮稳定。

---

## Phase 2 Gate B: 多级只读探针

修复了粗层探针的双重释放 bug：

- **根因**: 自定义析构 `JXPAMG_PR_DestroyDeepCopy` 忽略 ownership flag，KT 转移 `col_starts` 所有权后仍然释放 → double-free
- **修复**: 删除自定义析构，全部使用标准 `jx_ParCSRMatrixDestroy`（遵循 ownership 契约）

新实现：
- `DeepCopyParCSR` — 完整深拷贝 CSR 数据
- `BuildReuseCoarseOperatorFrom` — KT + 深拷贝 + 状态返回 + goto cleanup
- `PredictCutReadOnly` — MPI consensus、cut_k=0 立即退出、单出口清理
- 接口 — 预测失败不 abort，继续完整重建

验证：
```
Level 0: sigma=5.228311e-06
Level 1: sigma=1.108508e-05
Level 2: sigma=6.794117e-06
Level 3: sigma=1.810064e-04
Level 4: sigma=5.235767e-04
Level 5: sigma=0.000000e+00
```

np=1：全部通过，Valgrind 0 errors。np=2 部分通过（预存间歇 bug）。

---

## 已知限制

| 限制 | 来源 | 状态 |
|------|------|------|
| np=4 崩溃 | 预存库 bug | 需独立定位 |
| np=2 间歇崩溃 | 预存库 bug | Vely 约 30% 通过 |
| 深拷贝无 CommPkg | 探针对 np>1 需 fallback | 可优化 |
| 中间 cut_k 未自然覆盖 | 数据集特性 | 可接受 |

---

## 改动文件

| 文件 | 改动 |
|------|------|
| `csrc/amg/par_amg_setup.c` | Sigma 修复、DeepCopy、BuildReuse、PredictCutReadOnly 重写、FullSetupOnNewMatrix |
| `src/jxpamg_interface.c` | Destroy 注释、所有权修正、Case 12 完整流程、solve_status 传播 |
| `src/solver.c` | shared_amg 销毁、返回值检查、goto cleanup |
| `include/jx_pamg.h` | 新函数声明 |
| `Makefile` | JXPAMGDIR → pr |

---

## 编译与运行

```bash
source /opt/intel/oneapi/setvars.sh --force
make -C jxpamg-0220-pr -j8
make -f Makefile_pr clean && make -f Makefile_pr

mpirun -np 1 ./solver_pr \
  A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 \
  -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr \
  --predictive-theta 1e-7
```

---

## 相关文档

| 文件 | 说明 |
|------|------|
| `PHASE1_FINAL.md` | Phase 1 完成总结 |
| `PHASE1_COMPLETE.md` | Phase 1 详细报告 |
| `PHASE2_STABLE_INCREMENTAL_REUSE_DESIGN.md` | Phase 2 设计 |
| `GATE_B_STATUS.md` | Gate B 完成报告 |
| `GATE_B_COMPLETION_PLAN.md` | Gate B 实施计划 |
| `PHASE2_GATE_AB_REPORT.md` | Gate A-B 进展 |
| `valgrind_test.log` | Phase 1 Valgrind 日志 |
