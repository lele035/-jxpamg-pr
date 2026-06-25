# JXPAMG Predictive Reuse — 总体状态

**日期**: 2026-06-22
**checksum**: `7d7a41821b1023e68f476de0a27bd538`
**项目**: `/home/xiyudawang/jxpamg-0220-pr/`
**服务器**: `spring`, `/home/spring/ljc/jxpamg_702_sharesetup/`

---

## 1. 总体进度

| 阶段 | 状态 | 核心交付 |
|------|------|---------|
| Phase 1 | ✅ | 稳定性基线：完整重建 + 错误传播 |
| Gate A | ⚠ | MPI 基线 (np=1 ✅, np=2 间歇, np=4 预存崩溃) |
| Gate B | ⚠ | 多级只读探针 (np=1 ✅, np>1 未验收) |
| Gate C | ✅ np=1 | 候选 A[cut_k] tail rebuild (3 cut_k, Valgrind 0) |
| **Gate D** | **✅ np=1** | **cut_k=N full reuse via no-op AMG setup; Valgrind 0** |
| 3B-2 | 待 | 代码在 `.inc`，未接入 setup |

---

## 2. 完整路由

```
第二次调用 (shared AMG, np=1):

  PredictCutReadOnlyWithCandidate(new_A) → cut_k + candidate_A_cut
  │
  ├─ probe 失败              → FullSetupOnNewMatrix
  ├─ cut_k == 0              → FullSetupOnNewMatrix
  ├─ 0 < cut_k < N, candidate → SetupFromLevelWithACut(cut_k, candidate)
  │                             失败 → FullSetupOnNewMatrix
  └─ cut_k == N, CanFullReuse → Gate D: no-op setup (跳过重建, 复用旧 AMG)
                                 PCG 求解 → 收敛
```

| np | cut_k 行为 |
|----|-----------|
| 1 | 四路分流：full setup / tail rebuild / full reuse / fallback |
| >1 | 强制 FullSetupOnNewMatrix（Gate C/D 未释放） |

---

## 3. Gate D: Full Reuse

**原理**: `cut_k == num_levels` 表示所有层均在阈值内。不再重建 AMG，直接用现有 hierarchy 作 PCG 预条件器。

**实现**:
- `JXPAMG_PR_NoOpSetup`: PCG setup 回调，直接返回 0（不调 JX_PAMGSetup）
- `JXPAMG_PR_CanFullReuse`: 检查 hierarchy 完整（A/P/F/U 全部非 NULL）、维度匹配、Lreuse 不截断
- case 12 路由: `CanFullReuse` + `np==1` → no-op setup → PCG solve

**证据**: `gate_d_full_reuse_Vely_0.5.log` (Valgrind 0, 5 iters), `gate_d_results.tsv`

---

## 4. 已修复的问题汇总

| 阶段 | # | 修复 |
|------|---|------|
| Phase 1 | 1-10 | Sigma Allreduce, Level 0 自比较, A_H 泄漏, k 覆盖, 所有权, 野指针×7, 生命周期, FullSetup 安全化, 返回值, 错误传播 |
| Gate B | 11-15 | 自定义析构→标准 Destroy, 深拷贝 ownership, KT status-returning, MPI consensus+goto, 预测失败 fallback |
| Gate C | 16-17 | 候选 A[cut_k] 捕获+安装, case 12 路由 |
| Gate D | 18-19 | NoOpSetup, CanFullReuse 检查 |

---

## 5. 改动文件

| 文件 | 改动 |
|------|------|
| `csrc/amg/par_amg_setup.c` | DeepCopy, BuildReuseCoarseOperatorFrom (status+A_H_ptr), PredictCutReadOnly (+WithCandidate, 候选捕获), FullSetupOnNewMatrix, SetupFromLevelExperimental (去 static), SetupFromLevelWithACut (新增), NoOpSetup (新增), CanFullReuse (新增) |
| `src/jxpamg_interface.c` | Destroy×7, 所有权, Case 12 (cut_k 四路路由), solve_status 传播 |
| `src/solver.c` | shared_amg 销毁, 返回值检查, goto cleanup |
| `include/jx_pamg.h` | 8 个新函数声明 |
| `Makefile` | JXPAMGDIR → pr |

---

## 6. 验证结果

| 路径 | 矩阵 | theta | cut_k | 迭代 | Valgrind |
|------|------|-------|-------|------|---------|
| full_setup | Vely | 1e-7 | 0 | 4 | 0 |
| tail_rebuild | Velz | 1e-7 | 1 | 5 | 0 |
| full_reuse | Vely | 0.5 | 6 | 5 | 0 |

批测 5/5: PresCorr(13), TurbK(2), Vely(4), Velz(5), Vof(4)

---

## 7. 已知限制

| 限制 |
|------|
| np>1 full reuse 和 tail rebuild 均未释放 |
| np=4 崩溃, np=2 间歇 (预存 bug) |
| 非严格事务提交 (旧 tail 在重建前销毁, 失败由完整重建兜底) |
| 3B-2 未集成 |
| partial Lreuse 不触发 full reuse |

---

## 8. 平滑器字段

`include/jx_pamg.h`, `jx_ParAMGData`:
`smooth_num_levels`(192), `smooth_type`(193), `smoother`(194, `JX_Solver *` 数组)

---

## 9. 编译与运行

```bash
source /opt/intel/oneapi/setvars.sh --force
make -C jxpamg-0220-pr -j8
make -f Makefile_pr clean && make -f Makefile_pr

# full reuse: Vely theta=0.5 (k=6)
mpirun -np 1 ./solver_pr A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr -ptheta 0.5

# tail rebuild: Velz theta=1e-7 (k=1)
mpirun -np 1 ./solver_pr A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 -mat2 A_Velz-0.csr -rhs2 b_Velz-0.csr -ptheta 1e-7

# full setup: Vely theta=1e-7 (k=0)
mpirun -np 1 ./solver_pr A_Velx-0.csr b_Velx-0.csr 1e-6 \
  -sid 12 -maxiter 200 -mat2 A_Vely-0.csr -rhs2 b_Vely-0.csr -ptheta 1e-7
```

---

## 10. 本地证据

| 文件 | 说明 |
|------|------|
| `valgrind_test.log` | Phase 1 Valgrind |
| `gate_b_valgrind.log` | Gate B theta=1e-7 |
| `gate_b_valgrind_theta0.5.log` | Gate B theta=0.5 (6 层) |
| `gate_b_results.tsv` | Gate B 批测 |
| `gate_c_candidate_*.log` | Gate C 3 路径 Valgrind |
| `gate_c_candidate_results.tsv` | Gate C 批测 |
| `gate_d_*.log` | Gate D 3 路径 Valgrind |
| `gate_d_results.tsv` | Gate D 结果 TSV |

## 11. 文档索引

| 文件 | 说明 |
|------|------|
| `OVERALL_STATUS.md` | 本文档 |
| `PHASE2_STABLE_INCREMENTAL_REUSE_DESIGN.md` | Phase 2 总体设计 |
| `GATE_B_STATUS.md` | Gate B 完成报告 |
| `GATE_B_COMPLETION_PLAN.md` | Gate B 实施计划 |
| `GATE_C_NEXT_STEPS.md` | Gate C 正确性方案 |
| `GATE_D_FULL_REUSE_PLAN.md` | Gate D 实施计划 |
| `PHASE1_FINAL.md` | Phase 1 完成总结 |
