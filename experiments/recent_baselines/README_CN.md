# LegalFactAware 近期基线实验包

本目录包含三条近期生成式基线的 CAIL 适配、冻结输入、统一评测、自动测试和 GPU 诊断结果：

- DELTA：EMNLP 2026 的训练免费字符长度控制方法；
- Black-box LCG：ACL 2025 的 MH+IS 黑盒长度控制方法。
- Zero-Shot LCS：Findings of NAACL 2025 的样本筛选与自动修订方法。

当前三者都已成功运行。DELTA 的中文法律生成明显退化；LCG 与 Zero-Shot LCS 在严格上限
目标下很少或完全不触发搜索/修订，且输出偏短。Zero-Shot LCS 已完成冻结 100 篇，可作为
单独标注的近期生成式适配对照。完整判断见 `RUN_STATUS.md`。

## 建议阅读顺序

1. `RUN_STATUS.md`：先看当前结论和是否继续全量运行。
2. `PREREGISTRATION.md`：DELTA 和通用生成式基线的冻结边界。
3. `LCG_PREREGISTRATION.md`：LCG 在查看输出前冻结的上限适配。
4. `prepare_cail_dev.py`：生成 942 篇统一输入和冻结系统输出。
5. `protocol.py`：共同字符、ROUGE、Expression 和截取规则。
6. `run_delta_cail.py`：DELTA 的 CAIL 入口。
7. `prepare_lcg_demo.py`：冻结不重叠的 one-shot 训练示例。
8. `lcg_protocol.py`：LCG 长度分布、接受概率和裁判分数解析。
9. `run_lcg_blackbox.py`：Qwen 直接生成、修改提议、成对裁判和 MH+IS 选择。
10. `evaluate_cail_baselines.py`：统一质量、长度、来源支持和 bootstrap 评价。
11. `CODE_WALKTHROUGH_CN.md`：按上述顺序准备好的口头讲解稿。
12. `ZERO_SHOT_LC_PREREGISTRATION.md`：Zero-Shot LCS 的 5 篇冻结边界。
13. `zero_shot_lc_protocol.py` 与 `run_zero_shot_lc.py`：样本筛选、修订和运行入口。
14. `ZERO_SHOT_LC_SUBSET100_REPORT.md`：冻结 100 篇的最终比较和报告边界。

## 1. 本地准备与冻结系统验收

```bash
python3 experiments/recent_baselines_20261031/prepare_cail_dev.py \
  --verification-jsonl LegalFactAware_Verification_Package_20261020/jsonl/full.jsonl \
  --output-dir experiments/recent_baselines_20261031/data

python3 experiments/recent_baselines_20261031/evaluate_cail_baselines.py \
  --dataset experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --prediction LegalFactAware-v2.5-B=experiments/recent_baselines_20261031/data/legalfactaware_predictions.jsonl \
  --output-dir experiments/recent_baselines_20261031/results/baseline_validation
```

验收值：942 篇、Expression F1 `0.478526`、平均 `396.485` 字符、超预算 `0`。

## 2. 自动测试

```bash
python3 -m unittest discover \
  -s experiments/recent_baselines_20261031/tests -v
```

当前应有 25 项测试通过。

## 3. GPU 依赖

```bash
python -m pip install -r experiments/recent_baselines_20261031/requirements-gpu.txt
```

7B 模型建议至少 24GB 显存并使用 4bit。正式报告必须记录模型名称、revision、量化方式、
GPU、采样参数和随机种子。

## 4. 运行 DELTA 五篇诊断

```bash
python experiments/recent_baselines_20261031/run_delta_cail.py \
  --input experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --output experiments/recent_baselines_20261031/results/delta_dream_smoke5.jsonl \
  --official-code-dir experiments/recent_baselines_20261031/vendor/delta-diffusion \
  --model-id Dream-org/Dream-v0-Instruct-7B \
  --model-name dream \
  --target-chars 450 \
  --avg-chars-per-token 1.5263636665452414 \
  --quantization 4bit \
  --num-samples 5
```

脚本支持逐篇追加和断点续跑，不会把超长输出事后截到 450 字符。冻结字符/词元比只能从
开发集估计一次，测试集必须复用，不能读取测试参考摘要重新估计。

## 5. 准备 LCG one-shot 示例

```bash
python experiments/recent_baselines_20261031/prepare_lcg_demo.py \
  --train-jsonl PATH_TO_CAIL_TRAIN_JSONL \
  --evaluation-jsonl experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --output experiments/recent_baselines_20261031/data/lcg_demo_train_row0.jsonl \
  --index 0
```

脚本会拒绝与评测集 ID 重叠的示例。私有交付包中包含本轮实际使用的示例；公开代码包不
包含原始文书和预测文本。

## 6. 运行 LCG 预注册上限适配

```bash
python experiments/recent_baselines_20261031/run_lcg_blackbox.py \
  --input experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --demo experiments/recent_baselines_20261031/data/lcg_demo_train_row0.jsonl \
  --output experiments/recent_baselines_20261031/results/lcg_mhis_qwen25_smoke5.jsonl \
  --initial-output experiments/recent_baselines_20261031/results/qwen25_direct_smoke5.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target-chars 450 \
  --length-objective upper_bound \
  --max-iterations 5 \
  --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 \
  --num-samples 5
```

该模式严格对应本任务的“不超过 450 字符”。本轮五篇初始输出都已经合法，因此没有进入
修改或裁判步骤。

## 7. 运行 LCG 精确长度探索

将上条命令改为 `--length-objective exact`，并使用新的输出文件，避免复用上限提示的初始
结果。该模式是查看预注册结果后增加的诊断，不是主结果：

```bash
python experiments/recent_baselines_20261031/run_lcg_blackbox.py \
  --input experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --demo experiments/recent_baselines_20261031/data/lcg_demo_train_row0.jsonl \
  --output experiments/recent_baselines_20261031/results/lcg_mhis_qwen25_exact_smoke5.jsonl \
  --initial-output experiments/recent_baselines_20261031/results/qwen25_direct_exact_smoke5.jsonl \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --target-chars 450 --length-objective exact \
  --max-iterations 5 --quantization 4bit \
  --top-k 20 --top-p 0.8 --temperature 0.7 \
  --repetition-penalty 1.05 --seed 0 \
  --num-samples 5
```

## 8. 合并评价

```bash
python experiments/recent_baselines_20261031/evaluate_cail_baselines.py \
  --dataset experiments/recent_baselines_20261031/data/cail_dev_942.jsonl \
  --prediction LegalFactAware-v2.5-B=experiments/recent_baselines_20261031/data/legalfactaware_predictions.jsonl \
  --prediction LCG=experiments/recent_baselines_20261031/results/lcg_mhis_qwen25_smoke5.jsonl \
  --output-dir experiments/recent_baselines_20261031/results/lcg_eval \
  --limit 5
```

`--limit 5` 只用于烟雾测试。正式 942 篇评测必须去掉它，并确保所有方法 ID 集合完全一致。
第一项预测被视为 bootstrap 锚点。

## 9. 运行 Zero-Shot LCS

先阅读 `ZERO_SHOT_LC_PREREGISTRATION.md`。5 篇验收使用
`run_zero_shot_lc_smoke5.sh`，冻结 100 篇严格上限实验使用：

```bash
bash experiments/recent_baselines_20261031/run_zero_shot_lc_subset100.sh
```

该脚本保存三个初始候选、每轮修订和最终输出，不执行事后截断。英文长度近似和目标校准
系数没有直接迁移到中文，也没有在评测文档上重新拟合。

## 方法身份与报告边界

这些结果都是 CAIL 任务适配，不是论文作者发布的 CAIL checkpoint 或原论文数据集结果。
论文中应写作 `DELTA-Dream (CAIL adaptation)`、
`LCG-MHIS-Qwen2.5-7B (CAIL adaptation)` 与
`ZeroShotLC-SF3-AR5-Qwen2.5-7B (CAIL-adapted, independent)`，并披露中文提示、
head-tail 截取、字符口径、量化和硬预算违反率。五篇诊断不能进入正式主表；冻结 100 篇
结果可进入单独的资源受限比较表。

