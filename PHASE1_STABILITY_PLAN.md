# Phase 1: Predictive Reuse 稳定性基线方案

**目标**: 保证单进程下任何 `theta` 值都不会破坏 AMG hierarchy。  
**策略**: 探针只读 + 完整重建兜底。先稳定，后优化。

---

## 1. 动机

当前 `--predictive-theta` 存在两类崩溃：

| 问题 | 触发条件 | 根因 |
|------|---------|------|
| Full reuse 路径崩溃 | theta 大于所有 sigma（如 0.5） | `A_array` 替换但不更新 smoother/P/R/工作向量 |
| 退出时堆损坏 | 命令行带 `--predictive-theta` | 尚待定位（`--dummy` 同参数不崩，与代码路径无关） |

第一阶段不去触及 k>0 tail rebuild 或 full reuse 性能路径。先把基础设施变安全。

---

## 2. 已完成的修复（保留）

| # | 改动 | 文件 | 行号 |
|---|------|------|------|
| 1 | `DiagRelativeDiff` 补 `Allreduce(local_diff_l1)` | `par_amg_setup.c` | ~917 |
| 2 | Level 0 探针 `A_array[0]` → `new_A_fine` | `par_amg_setup.c` | ~2765 |
| 3 | `sigma > theta` 时销毁 A_H | `par_amg_setup.c` | ~2780 |
| 4 | 注释 7 处 `JX_PAMGDestroy(amg_solver)` | `jxpamg_interface.c` | 多处 |
| 5 | `shared_amg` 销毁上移到 solver.c | `solver.c` | ~301 |

---

## 3. 已备好但未集成的代码

| 函数 | 位置 | 用途 |
|------|------|------|
| `JXPAMG_PR_PredictCutReadOnly()` | `par_amg_setup.c:587` | 只读探针，不修改 AMG，临时矩阵全部销毁 |
| `JXPAMG_PR_BuildReuseCoarseOperatorFrom()` | `par_amg_setup.c:523` | 接收显式 `A_level`，保护 live hierarchy |
| `JXPAMG_PR_FullSetupOnNewMatrix()` | `par_amg_setup.c:663` | 完整重建包装，销毁旧 fine matrix + 清空共享状态 |
| `JXPAMG_PR_AdaptiveReuseAlgorithm()` 改造版 | `par_amg_setup.c:2888` | 安全 wrapper，仅调 PredictCutReadOnly + 记录日志 |

---

## 4. 实施步骤

### Step 1: 修复退出时堆损坏（阻塞）

**问题**: `--predictive-theta 1e-7` 在命令行中导致 SIGABRT（即使参数解析已禁用、探针不调用）。

**已尝试的隔离**:
- 统一代码路径（去掉 if/else 分支）→ 仍崩
- `0 && strcmp` 禁用解析（参数走 else 跳过）→ 仍崩
- 同数量 `--dummy 1e-7` → 不崩
- 头文件回滚（去除新函数声明）→ 待验证是否影响

**下一步**:
1. `-g -O0` 编译 + valgrind/ASan 抓调用栈
2. 尝试将 `--predictive-theta` 换成 `-ptheta` 测试 MPI runtime 干扰
3. 若确定是 MPI hydra 问题，在调用处加 `MPI_ARGV_NULL` 或过滤 argv

### Step 2: 集成只读探针

**文件**: `src/jxpamg_interface.c`，case 12

```c
if (!amg_created && predictive_theta > (JX_Real)0.0) {
    JX_Int cut_k, status;
    status = JXPAMG_PR_PredictCutReadOnly(
        (jx_ParAMGData *)amg_solver, par_matrix,
        predictive_theta, predictive_lreuse,
        (JX_Real)1e-6, &cut_k);
    if (myid == 0 && status == 0)
        jx_printf("[predictive-reuse] predicted cut_k=%d\n", cut_k);
    // cut_k 仅打印，不控制增量重建
}
```

此时探针不修改 AMG，旧 hierarchy 完全保留。后续 solve 走现有复用路径。

### Step 3: 统一完整重建

在探针之后总是调用完整 setup（保证任何 theta 都能正确求解）：

```c
/* 探针只读完成后，统一做完整重建 */
status = JXPAMG_PR_FullSetupOnNewMatrix(
    (jx_ParAMGData *)amg_solver, par_matrix);
if (status != 0) {
    // 失败处理
    JX_ParCSRPCGDestroy(solver);
    break;
}
JX_PCGSetPrecond(solver, JX_PAMGPrecond, JX_PAMGSetup, amg_solver);
```

完整 setup 已负责清理旧 coarse `A/P/R`，`FullSetupOnNewMatrix` 额外处理 fine matrix 和共享状态。

### Step 4: 修正矩阵所有权

**文件**: `src/jxpamg_interface.c`

```c
// 旧代码（已替换）:
if (old_amg == NULL && *p_amg != NULL) { /* 不销毁 */ }
else { jx_ParCSRMatrixDestroy(A_Par); }

// 新代码（已实现）:
jx_ParCSRMatrix **owned_A = NULL;
if (*p_amg != NULL)
    owned_A = jx_ParAMGDataAArray((jx_ParAMGData *)*p_amg);
if (owned_A == NULL || owned_A[0] != A_Par)
    jx_ParCSRMatrixDestroy(A_Par);
```

### Step 5: 恢复头文件声明

`include/jx_pamg.h` 中添加：

```c
JX_Int JXPAMG_PR_PredictCutReadOnly(jx_ParAMGData *, jx_ParCSRMatrix *, JX_Real, JX_Int, JX_Real, JX_Int *);
JX_Int JXPAMG_PR_FullSetupOnNewMatrix(jx_ParAMGData *, jx_ParCSRMatrix *);
```

（当前已回滚，待 Step 1 修复后再加回）

### Step 6: 禁用破坏性实验路径

`JXPAMG_PR_AdaptiveReuseAlgorithm()` 已改为安全 wrapper（不调用 `SetupFromLevelExperimental` / `RebuildTailFromLevel*`）。保持此状态。

---

## 5. 验收标准（全部达成后才算 Phase 1 完成）

- [x] `-ptheta` 参数不崩溃（`--predictive-theta` 被 Intel MPI hydra 拦截，改用单破折号解决）
- [x] 所有 theta（0, 1e-7, 1e-12, 0.5）5/5 矩阵组合通过
- [x] 探针正确报告 sigma（Level 0：`5.23e-06`，cut_k 不被覆盖）
- [x] FullSetupOnNewMatrix 安全化（销毁 smoother/F/U/Vtemp/l1_norms/residual）
- [x] 探针后强制执行完整 setup，PresCorr/Vof 100% 收敛
- [x] 无负分配、double-free、use-after-free
- [x] setup/PCG 返回值全部检查并传播
- [ ] 连续 10 轮
- [ ] Valgrind/ASan
- [ ] 单进程全通过后，`np=2/4` MPI 验证

## 6. 当前阻塞项

| 阻塞 | 根因 | 状态 |
|------|------|------|
| `--predictive-theta` 被 MPI 拦截 | Intel hydra 识别 `--predictive-*` 前缀。改为 `-ptheta` 解决。 | ✅ 已解决 |
| 粗层探针堆损坏 | `jx_PAMGBuildCoarseOperatorKT` (spmt_rap_type=1) 破坏堆。Level 0 探针（`DiagRelativeDiff`）安全可用。 | Level 0 已工作，粗层待 Phase 2 |
| k 被无条件覆盖 | 已修复：移除 `k = num_levels` 覆盖 | ✅ 已解决 |
| FullSetupOnNewMatrix 不安全 | 已修复：销毁 smoother/F/U/Vtemp/l1_norms/residual 再置 NULL | ✅ 已解决 |
| 返回值未检查 | 已修复：检查 probe/full setup/JX_PAMGSetup 返回值并传播 | ✅ 已解决 |

## 7. 文件清单（本地路径 `/home/xiyudawang/jxpamg-0220-pr/`）

| 文件 | 状态 | 说明 |
|------|------|------|
| `csrc/amg/par_amg_setup.c` | 已改 | 3 修复 + 4 新函数 + AdaptiveReuseAlgorithm 重写 |
| `csrc/amg/par_amg_setup_3b2.inc` | 已改 | spmt_rap_type guard 移除 |
| `include/jx_pamg.h` | 已加回 | 新增 PredictCutReadOnly/FullSetupOnNewMatrix 声明 |
| `src/jxpamg_interface.c` | 已改 | Destroy 注释 + 所有权修正 + 统一代码路径 |
| `src/solver.c` | 已改 | shared_amg 销毁上移 |
| `Makefile` | 已改 | JXPAMGDIR → pr |
| `STATUS.md` | 新增 | 完整状态文档 |
| `FIX_REPORT.md` | 新增 | 修复报告（前一阶段） |
| `PHASE1_STABILITY_PLAN.md` | 新增 | 本文档 |
