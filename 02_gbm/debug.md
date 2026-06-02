# 审计报告：gbm_best_models.ipynb

审计日期：2026-06-02
审计对象：D:\quant\ML_teamwork\02_树模型\02_gbm\gbm_best_models.ipynb
审计范围：代码逻辑正确性 + 数据泄露

---

## 一、结论先行

| 维度 | 结论 |
|------|------|
| 数据泄露（notebook 内） | 干净。5 项防护全部到位，无 notebook 级泄露 |
| 数据泄露（上游 factortb） | 无法验证，是真正风险敞口（见 四） |
| 逻辑正确性 | 3 个确认 bug + 1 个需验证。最严重：early stopping 形同虚设 |

---

## 二、数据泄露分析（逐项核对）

| 检查项 | 判定 | 依据 |
|--------|------|------|
| 特征是否混入未来收益 | 安全 | feature_cols = [c for c in df.columns if c.startswith('ALPHA')] 白名单，正确规避历史 [P0] 排除法泄漏坑 |
| 标签前视（gap vs horizon） | 安全 | tei = tsi-15，train_end 标签终点 = train_end+10 < test_start = train_end+15，留 5 日 buffer |
| 预处理统计量泄露 | 安全 | 仅 inf→nan 逐元素，无跨样本标准化/归一化（树模型 P1_raw） |
| 验证集隔离 | 安全 | 验证集取训练期尾部 20%，从训练集切出，不碰测试集 |
| 自定义 obj/eval 分组 | 安全 | obj_sub 用 sub_s/sub_e、evl_val 用 val_s/val_e，分组与各自 DMatrix 行序一致，无测试集分组流入 |
| 全量重训 | 安全 | 不重训，验证集数据从不进梯度 |
| 跨窗口数据混用 | 安全 | 各窗口独立索引切片全局数组，无串扰 |

泄露结论：notebook 本体设计是同类代码里较干净的一份。gap=15>10 的保守设计、特征白名单、验证集尾部切分都对。

旁证：同目录 baseline notebook 用 GAP_DAYS=10（恰等于 horizon，边界泄露风险），本 notebook 改用 15 —— 作者明显意识到了这个坑。

---

## 三、逻辑 Bug（按严重程度）

### [P1] Bug 1：predict 未截断到 best_iteration，early stopping 白做

核心问题。cell-12：

```python
preds = m_search.predict(d_test)   # ← 没有 iteration_range
```

xgboost 原生 Booster.predict() 默认用全部树（iteration_range=(0,0)），不自动用 best_iteration —— 这与 sklearn 接口相反（官方文档确认）。

后果链：
1. 第二次 train 带 early_stopping_rounds=30，触发停止时模型含 best+30 棵树（ES 多训练的 30 棵正是想丢弃的过拟合尾部）。
2. best_n_trees 算出来只用于打印 "Trees" 列，不进预测。
3. 实际预测用了全部树 → early stopping 选 best 的努力完全失效，报告的 Trees 数与真实预测树数不符。

修复：

```python
preds = m_search.predict(d_test, iteration_range=(0, best_n_trees))
```

参考：https://xgboost.readthedocs.io/en/stable/prediction.html

---

### [P1] Bug 2：验证集 QuantileDMatrix 未传 ref，分箱不一致

cell-12：

```python
d_sub = xgb.QuantileDMatrix(wd['X_sub'], label=wd['y_sub'])
d_val = xgb.QuantileDMatrix(wd['X_val'], label=wd['y_val'])   # ← 缺 ref
```

验证集用 QuantileDMatrix 时必须传 ref=训练集 DMatrix，否则验证集按自己的分布独立分箱，与训练集分箱边界错位 → early stopping 评估的 IC 有偏 → 可能选错 best_iteration。

修复：

```python
d_val = xgb.QuantileDMatrix(wd['X_val'], label=wd['y_val'], ref=d_sub)
```

注：Bug 1 与 Bug 2 当前部分相互抵消（best_iteration 选错了，但 predict 又没用它）。一旦修 Bug 1，Bug 2 的影响才会显现 —— 两个必须一起修。

---

### [P1] Bug 3：best_n_trees 计算依赖未验证的 continue-training 语义

```python
best_n_trees = m_search.best_iteration + MIN_TREES + 1
```

continue training（xgb_model=）时，best_iteration 是本次调用相对轮次还是全局轮次，xgboost 跨版本行为变过。

- 若相对本次（0-based）：best_iteration + 50 + 1 正确。
- 若已含热身 50 棵：多加 50，预测范围错。

验证方法（跑一窗打印即可定位）：

```python
print(m_search.num_boosted_rounds(), m_search.best_iteration)
# num_boosted_rounds 应 = 50 + (本次训练轮数)；best_iteration 若 < 本次轮数 → 相对本次
```

更稳的做法：第二次 train 用 callbacks=[xgb.callback.EarlyStopping(rounds=30, save_best=True)]，直接拿裁剪好的模型，绕开手算。

---

### [P2] Bug 4：split 混淆"有效计数"与"位置索引"

cell-9：

```python
n_train = int(vm_tr.sum())      # 有效标签【计数】
split = int(n_train * 0.8)      # 基于计数算的数
vm_sub[split:] = False          # ← 当成【位置索引】用在含 NaN 的全长 mask 上
vm_val[:split] = False
```

vm_tr 全长含 NaN 行，split 却按有效计数算 → NaN 多时切分点偏移，验证集实际比例 >20%。

当前数据 NaN 极少（W00：574/140000），实测切分 79.99%/20.01%，影响微小。但逻辑是隐患，最后一个窗口若末尾标签未实现会放大。

修复：

```python
valid_pos = np.where(vm_tr)[0]
split_pos = valid_pos[int(len(valid_pos) * (1 - VAL_RATIO))]
vm_sub = vm_tr.copy(); vm_sub[split_pos:] = False
vm_val = vm_tr.copy(); vm_val[:split_pos] = False
```

---

### 次要观察（非 bug，提示）

- ic_aware_obj 的 Hessian 是简化对角近似（h = inv，未含 pc 二次项）。IC-aware 目标常见做法，可能略影响收敛速度，不影响正确性。
- 梯度推导正确：grad = -(yc - r·pc)/(‖yc‖‖pc‖) 是 -Pearson 对 p 的标准梯度。
- device='cuda' + numpy 输入：自动 CPU→GPU 传输，仅 warning。GPU hist 固定 seed 基本可复现，极端追求 bit 级复现可加 deterministic_histogram（P2）。
- W7 IC=-0.10、W15 IC=-0.03：模型在 2021H2、2025H2 失效，非代码问题，是市场 regime。

---

## 四、审计边界 —— 真正的泄露风险在这里

本 notebook 只消费 factortb.parquet，构造脚本不在 ML_teamwork 下（已 grep 确认：FWD_RET 无构造代码，仅引用）。两个泄露关键点无法从本文件验证，必须追溯上游：

1. ALPHA_* 因子是否 PIT 对齐 —— 因子值是否用了计算时点尚不可得的数据（财报未公告期、复权用未来除权因子等）。历史 checklist 里 [P0] qfq latest_mult 含前视、[P0] PIT 成分股过滤漏步 都是这类，高发。
2. FWD_RET_10D 精确定义 —— 必须是 t+1 起算到 t+11（次日建仓）。若用 t→t+10（含当日收益），则当日泄露。

建议：定位 factortb 构造脚本（大概率在 mods2/01_factor_things），单独审一轮。notebook 再干净，喂进来的因子若带前视，结论照样失真。

---

## 五、命中历史 checklist

| 历史条目 | 本次 |
|----------|------|
| [P0] FEATURE_COLS 排除法导致未来收益当因子 | 正确规避（白名单 startswith('ALPHA')） |
| xgb 案例：early stopping val 必须从 train 切出 | 正确（尾部 20% 从 train 切） |
| [P1] QuantileDMatrix 验证集需传 ref | 新发现，建议固化 |
| [P1] 原生 predict 默认全树非 best_iteration | 新发现，建议固化 |

---

## 六、一句话总结

泄露防护合格，但 early stopping 实际没生效（Bug 1）—— 当前 IC 数字是"全树预测"而非"best 树预测"的结果，修复后指标会变化，需重跑确认。

修复优先级：Bug 1 + Bug 2 必须一起改（否则 ES 仍不生效）→ Bug 3 跑一窗验证语义 → Bug 4 改稳健写法 → 追溯 factortb 上游审因子 PIT。

