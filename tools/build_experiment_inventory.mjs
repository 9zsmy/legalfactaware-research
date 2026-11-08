import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/admin/Documents/ChatGPT/1/outputs/paper_comparison_experiments_20261101";
const outputPath = `${outputDir}/论文对比实验总表_20261101.xlsx`;
await fs.mkdir(outputDir, { recursive: true });

const columns = [
  "编号", "实验/方法", "分组", "原方法年份", "会议/期刊", "级别/性质",
  "本项目做了什么", "数据与规模", "当前状态", "关键结果/发现",
  "能否继续做", "能否进论文主表", "本地代码完整度", "官方代码/资产",
  "结果证据完整度", "主要问题/风险", "建议下一步", "本地证据", "官方来源"
];

const rows = [
  [1, "LegalFactAware v2.5-B（中文核心）", "核心方法", "内部方法", "—", "本文方法；无外部级别", "冻结中文事实感知句子评分器与完整句预算解码；作为全部中文对照的主方法。", "CAIL2020-clean：7495/942/948；测试集948；450字符、最多8句", "已完成（正式测试）", "测试 R1/R2/RL=60.94/42.10/50.73；Fact Recall=70.99；0次硬截断。", "可以；仅需统一论文口径与归档", "可（中文主表）", "完整；训练、推理、评测和冻结包齐全", "不适用（自研）", "完整；汇总、逐文档输出与验证包齐全", "Fact/Expression两套Recall定义必须分开；事实正确性仍需人工评估。", "锁定正式版本；自动生成论文表；补盲法人工裁判结果评估。", "LEGALFACTAWARE_CORE_METHOD_HANDOFF_20260919.md", "—"],
  [2, "E1 英文扩展", "核心方法扩展", "内部方法", "—", "本文扩展；无外部级别", "将同一轻量评分与完整句预算选择迁移到IN-Abs英文开发集。", "IN-Abs：6325 train / 703 dev；官方100 test保持封存；参考长度预算", "已完成（开发集）", "Dev R1/R2/RL=66.42/38.15/62.95；0预算违规。", "可以；官方test需独立解封流程", "仅可进开发集/附录；不能冒充官方test", "完整", "不适用（自研）", "完整开发集证据；官方test未运行", "不是官方测试结果；与中文450字符协议不同。", "论文明确标Dev；如投稿需要，再一次性运行官方test。", "LEGALFACTAWARE_CORE_METHOD_HANDOFF_20260919.md", "—"],
  [3, "Phase 6A 状态依赖诊断", "机制诊断", "内部实验", "—", "诊断实验", "比较静态排序与逐步选择状态，量化边际收益、排序和选择路径变化。", "IN-Abs dev 703；643词预算轨道", "已完成", "状态翻转77.81%；成对次序翻转34.77%；正边际变非正31.98%；与Oracle选择一致率5.79%。", "可以扩展，但当前证据已够支持动机", "仅附录/机制分析", "完整", "不适用", "完整", "诊断不等于动态模型优于静态模型。", "作为动态重排动机，不写成性能提升结论。", "inabs_extension_work/PHASE6A_STATE_DEPENDENCE_REPORT.md", "—"],
  [4, "Phase 6B 动态重排", "计划实验", "内部方法", "—", "未发表计划", "拟在每次选句后重算覆盖、冗余与事实状态。", "计划在IN-Abs dev和CAIL dev运行", "未运行", "无结果。", "能做；开发和成本中等", "运行并通过冻结协议后再判断", "设计/预注册为主；无正式结果", "不适用", "无", "容易与现有MMR/动态选择贡献重叠；需预注册。", "先做小规模正确性审计，再做完整dev。", "LEGALFACTAWARE_CORE_METHOD_HANDOFF_20260919.md", "—"],
  [5, "多预算适配", "计划实验", "内部方法", "—", "敏感性实验", "拟比较不同字符/词预算下的质量、违规率和方法排序。", "计划CAIL多字符预算；IN-Abs多词预算", "未运行", "无结果。", "能做；成本低到中", "适合作为附录；主表保留主预算", "尚未实现统一正式脚本", "不适用", "无", "不同语言预算单位不能混写；需锁定预算网格。", "先在CAIL dev预注册300/400/450等预算；避免测试集调参。", "LEGALFACTAWARE_CORE_METHOD_HANDOFF_20260919.md", "—"],
  [6, "完整句选择器：贪心 vs 精确0/1背包", "预注册控制", "内部实验", "2026预注册", "选择器因果对照", "固定同一冻结句分数、450字符和8句上限，仅改变可行选择算法。", "CAIL dev 942", "已预注册，未见结果文件", "无正式结果。", "能做；CPU成本低", "结果可进机制附录", "预注册完整；结果未生成", "不适用", "仅有预注册", "必须先复现无冗余贪心归档预测；不能用结果反调协议。", "直接运行预注册脚本并输出配对检验。", "experiments/chinese_feasible_selector_control_20261020/PREREGISTRATION.md", "—"],
  [7, "标签构造：450字符约束 vs 无字符约束", "预注册控制", "内部实验", "2026预注册", "Oracle标签因果对照", "保持8句上限和全部模型组件，只切换标签构造时是否启用450字符约束。", "CAIL train/dev 7495/942", "已预注册，未见结果文件", "无正式结果。", "能做；需重造标签并重训两臂", "结果可进机制附录", "预注册完整；结果未生成", "不适用", "仅有预注册", "不要与“正标签数量匹配”实验混为一谈。", "按冻结协议运行；报告标签一致率和配对检验。", "experiments/chinese_matched_oracle_control_20261020/PREREGISTRATION.md", "—"],
  [8, "逐文档匹配正标签数量", "审稿人控制", "内部实验", "2026", "开发集机制证据", "将独立标签的每篇正例数精确匹配Oracle，排除标签密度混杂。", "CAIL dev 942", "已完成", "Full vs matched：60.27/41.17/49.84 vs 58.29/38.92/47.47；ΔRL=+2.37；Expression F1仅+0.06且不显著。", "已完成；可补更严格长度匹配", "可进消融/附录，不替代主表", "完整", "不适用", "完整；含配对CI与校正检验", "输出仍相差约58字符；未排除长度偏好和分数分布。", "正文写成“排除数量混杂但未排除长度差异”。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [9, "词法、位置与辅助特征拆分", "审稿人控制", "内部实验", "2026", "特征消融", "把原始词法、位置装饰和Legal14/Section6辅助特征拆开。", "CAIL dev 942", "已完成", "完整60.27/41.17/49.84；原始词法单独59.86/40.55/49.20；仅位置38.92/15.85/24.64。", "已完成；无需继续堆弱消融", "可进消融表", "完整", "不适用", "完整", "历史表有约0.01重跑差；位置交互未做显著性检验。", "统一自动生成表；结论聚焦“词法为主、辅助特征小幅增益”。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [10, "字符预算 × 候选上限网格", "审稿人控制", "内部实验", "2026", "解码敏感性", "运行B=300/400/450与K=4/6/8/10/12/无限的18格网格。", "CAIL dev 942；18个设置", "已完成", "B450/K8 RL=49.84；K10/12/无限分别49.04/48.60/48.39；K8触发率85.24%。", "已完成；除非审稿人要求无需扩大", "仅附录/敏感性", "完整", "不适用", "完整", "训练标签也受K=8影响，不能证明所有重训条件下K8全局最优。", "分别报告“达K停止”和“剩余全放不下”。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [11, "fit/skip、drop-only、旧尾截断", "审稿人控制", "内部实验", "2026", "解码机制对照", "区分避免候选内截断与继续选择替代候选两个作用。", "CAIL dev 942", "已完成", "RL：49.84 / 48.33 / 48.66；Expression F1：47.85 / 44.53 / 45.50；旧版有379次硬截断。", "已完成", "可进关键消融表", "完整", "不适用", "完整", "流程有路径交互，不能做可加性贡献分解。", "用“支持替代候选的重要性”，避免写primarily/全部由。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [12, "空表达与分类型Expression审计", "评测审计", "内部实验", "2026", "指标口径审计", "审计空参考文档并分别报告金额、日期、法律表达、裁判结果。", "CAIL dev 942；支持数303/93/942/908", "已完成", "无全类型皆空文档；Full micro F1=39.10/23.10/50.58/79.34。", "可以继续到人工评估", "仅附录/评测说明", "完整", "不适用", "完整", "fixed-union Expression Recall与旧conditional Fact Recall不可直接合并；类型极不均衡。", "正文固定一个口径；附录解释旧口径；补人工正确性评价。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [13, "训练正例长度/位置偏好诊断", "后续诊断", "内部实验", "2026", "标签偏好分析", "比较Oracle与独立标签正例的长度、位置和角色倾向。", "CAIL dev/train派生统计", "已完成", "Oracle正例均长54.18字符，独立标签44.27；配对约+11.10字符；位置差较小。", "可以补角色分布，但优先级低", "仅附录", "完整", "不适用", "完整", "长度差是机制线索，不是因果证明。", "与匹配数量实验联读，避免归因到“校准”。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [14, "实际输出长度接近子集", "后续诊断", "内部实验", "2026", "长度混杂审计", "只在两臂输出长度相近的文档上比较匹配标签与完整系统。", "CAIL dev；|长度差|≤25字符，n=308", "已完成", "ΔR1/R2/RL=+0.76/+0.54/+0.78；Expression=-0.46。", "可以做更正式的加权/回归控制", "仅附录；不可替代全样本主分析", "完整", "不适用", "完整", "事后筛选子集可能引入选择偏差。", "若要加强，做预注册的长度分层或加权分析。", "experiments/chinese_reviewer_controls_20261028/RESULTS.md", "—"],
  [15, "完整流水线成本分析", "效率实验", "内部实验", "2026", "同机完整流程计时", "从可审计本地流水线比较训练/准备/推理总耗时，三次重复。", "5种核心方法；同一服务器；3次重复", "已完成", "分钟：SSE14.42、LFA31.18、SENDE34.19、MemSum95.86、TermDiffuSum155.52。", "已完成；可补显存/能耗", "可进效率表（限这5种）", "完整", "依赖各方法本地适配", "完整", "不能外推到未同口径计时的全部基线。", "论文明确覆盖范围；不要声称“对全部方法最快”。", "experiments/computational_cost_analysis_20261023/RESULTS_FINAL.md", "—"],
  [16, "统一起点推理基准", "效率实验", "内部实验", "2026", "统一输入后的5次计时", "尽量统一从同一准备状态开始，测推理/选择阶段。", "5种核心方法；5次重复", "已完成", "秒：SSE4.00、LFA16.03、SENDE19.52、MemSum35.78、TermDiffuSum1341.11。", "已完成", "可进效率附录", "完整", "依赖本地适配", "完整", "不同架构预计算边界仍需解释。", "与完整流水线表并列，不互相替代。", "experiments/unified_runtime_benchmark_20261021/RESULTS.md", "—"],
  [17, "冻结模型旧版推理基准", "效率补充", "内部实验", "2026", "支持性计时", "在既有冻结模型/缓存状态下重复计时。", "5种方法；3次重复", "已完成（支持性）", "LFA17.081s；SSE5.047s；MemSum35.784s；SENDE5.076s（缓存嵌入）；TermDiffuSum1325.366s。", "无需继续；保留审计", "不作为唯一正式效率表", "完整", "依赖缓存/冻结资产", "完整", "不是严格raw-text同起点；SENDE缓存使其不可直接与完整流程等同。", "只在附录标注边界。", "experiments/full_runtime_benchmark_20261021/RESULTS.md", "—"],
  [18, "SENDE", "正式效果基线", "2026", "The Journal of Supercomputing", "SCI期刊；非CCF会议", "按论文思路独立重建中文与英文版本，并在统一预算协议评测。", "CAIL test 948；IN-Abs dev 703", "已完成（独立重建）", "CAIL 59.72/41.13/50.65；IN-Abs 66.68/39.19/63.16。", "能做；优先补版本匹配的逐文档显著性", "可进主表，但必须标“独立重建”", "较完整；非作者checkpoint", "论文信息可得；本地不是官方完整复现", "聚合结果完整；缺版本匹配配对输出", "当前中文审稿控制没有对齐的SENDE逐文档版本。", "固定最终SENDE版本，重导逐文档输出并做配对检验。", "inabs_extension_work/PREREGISTRATION_PHASE5D_SENDE.md", "https://doi.org/10.1007/s11227-026-08248-4"],
  [19, "SSE（Structure-Enhanced）", "正式效果基线", "2023", "ECAI 2023", "CCF B（主会）", "按结构增强框架做中文重建；英文为结构proxy适配。", "CAIL test 948；IN-Abs dev 703", "已完成（重建/proxy）", "CAIL 57.14/37.25/46.25；Fact 61.47；IN-Abs 54.36/26.22/46.85。", "能做；可补更忠实结构映射", "中文可进主表；英文需标proxy", "完整的本地实现；非官方逐位复现", "论文可得；官方资产完整性有限", "结果完整", "英文结构标签不是原任务官方结构；不能称官方复现。", "正文身份写清；与主方法做版本一致配对检验。", "inabs_extension_work/PREREGISTRATION_PHASE5E_SSE.md", "https://doi.org/10.3233/FAIA230403"],
  [20, "Agarwal-ST", "正式效果基线", "2022", "Findings of EMNLP 2022", "Findings；不等同EMNLP主会长文", "对原监督式法律摘要方法进行受限重实现/迁移。", "CAIL test 948；IN-Abs dev 703", "已完成（受限重实现）", "CAIL 54.98/35.90/45.66；Fact 70.69；IN-Abs 66.41/38.94/62.77。", "能做；但提高忠实度需原预处理/权重", "可进主表并标身份", "可运行且完整；不是作者checkpoint", "论文正式页可得；资产需按仓库实际说明", "结果完整", "迁移协议和原任务不同。", "保留为已完成基线；不夸大复现身份。", "inabs_extension_work/PREREGISTRATION_PHASE5_SYMMETRIC_BASELINES.md", "https://aclanthology.org/2022.findings-emnlp.134/"],
  [21, "Gist（对称适配）", "正式效果基线", "历史方法；项目按AACL 2022基准实现", "AACL 2022基准论文", "AACL区域会；非ACL主会", "使用AACL法律摘要基准中的Gist代码/逻辑做中文和IN-Abs适配。", "CAIL test 948；IN-Abs dev 703", "已完成（适配）", "CAIL 53.62/34.35/44.27；Fact 69.42；IN-Abs 63.38/33.74/59.36。", "能做；但无必要继续调K", "可进主表并标“适配”", "完整适配；非bit-identical checkpoint", "官方基准仓库/代码存在", "结果完整", "原始任务与统一K/预算规则不同；官方原任务数值也未复现。", "保留主表身份说明；不声称完整官方复现。", "inabs_extension_work/PREREGISTRATION_PHASE5_SYMMETRIC_BASELINES.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [22, "TermDiffuSum（中文独立实现）", "正式效果基线", "2025", "COLING 2025主会", "CCF B", "根据论文和部分代码在CAIL上独立实现扩散式法律抽取摘要。", "CAIL test 948", "已完成（独立实现）", "54.55/34.98/43.80；Fact 60.21。", "能做；完整忠实复现受资产缺失限制", "可进主表，但必须标“独立实现”", "中文实现完整；官方代码仅部分", "LegalAFSum、checkpoint、标注及部分模块缺失", "本地结果完整", "不能称作者官方复现。", "保留对照；把缺失资产写入复现限制。", "inabs_extension_work/PHASE5H_TERMDIFFUSUM_FEASIBILITY.md", "https://aclanthology.org/2025.coling-main.216/"],
  [23, "TermDiffuSum（IN-Abs）", "可行性审计", "2025", "COLING 2025主会", "CCF B", "审计官方发布代码能否无损适配到统一400句候选协议。", "IN-Abs dev 703；101篇超过200句", "停止：协议不兼容", "未产出正式英文结果；现有代码固定200句位置。", "技术上可改，但不能称忠实复现；优先级低", "不可进当前英文主表", "部分代码；缺英语400句模型改造", "官方源仓库仅部分资产", "可行性报告完整；无效果结果", "直接运行会改变101篇候选集合。", "除非取得完整模型/作者支持，否则保持缺失项说明。", "inabs_extension_work/PHASE5H_TERMDIFFUSUM_FEASIBILITY.md", "https://aclanthology.org/2025.coling-main.216/"],
  [24, "CivilSum LED-tail", "生成式独立轨道", "2024", "SIGIR 2024主会", "CCF A", "用CivilSum式头/尾1024 token输入和LED生成，做CAIL与IN-Abs独立适配。", "CAIL test 948；IN-Abs dev 703", "已完成（非官方复现）", "CAIL 53.57/28.59/41.35；Fact43.60；IN-Abs 48.19/26.64/42.01。", "能做；可改成长上下文但会改变对照定义", "仅生成式分组，不进抽取式主排名", "完整本地适配", "官方数据和实验代码公开", "结果完整", "绝大多数英文文档只看尾部1024 token；不能把差距单归因于LED。", "保留独立轨道；报告输入截断和剩余预算。", "inabs_extension_work/PREREGISTRATION_PHASE5F_CIVILSUM_LED.md", "https://doi.org/10.1145/3626772.3657859"],
  [25, "MemSum", "正式效果基线", "2022", "ACL 2022主会长文", "CCF A", "使用官方架构重新训练，并适配中文450字符和英文参考词预算。", "CAIL test 948；IN-Abs dev 703", "已完成（架构重训）", "CAIL 49.56/28.16/38.52；Fact55.66；IN-Abs 59.18/30.23/55.09。", "能做；无需作为最优基线继续调参", "可进抽取式主表，标重训适配", "完整", "官方代码存在", "结果与计时完整", "英文P50/预算Oracle属于适配，不是原论文原任务设置。", "保留正式基线身份；说明预算适配。", "inabs_extension_work/PREREGISTRATION_PHASE5_SYMMETRIC_BASELINES.md", "https://aclanthology.org/2022.acl-long.450/"],
  [26, "EUR-Lex-Sum LexRank", "正式效果基线", "2022", "EMNLP 2022主会", "CCF A", "把论文公开的图排序/LexRank流程迁移到中英文法律摘要预算协议。", "CAIL test 948；IN-Abs dev 703", "已完成（语言/预算适配）", "CAIL 42.18/21.23/29.84；Fact36.91；IN-Abs 63.50/34.48/59.93。", "能做；已足够作为无监督图方法", "可进主表并标适配", "完整", "官方数据、基线代码公开", "结果完整", "论文主要贡献是数据集；这里比较的是其公开基线而非新模型复现。", "保留并准确命名为“EUR-Lex-Sum LexRank baseline”。", "inabs_extension_work/PREREGISTRATION_PHASE4_UNSUPERVISED.md", "https://aclanthology.org/2022.emnlp-main.519/"],
  [27, "Budget Recipe公开checkpoint", "生成式/零样本轨道", "2025", "JUST-NLP 2025 Workshop", "Workshop；非CCF主会", "加载公开Qwen3-4B checkpoint做零样本迁移；未执行论文完整两阶段SFT+RLVR。", "CAIL test 948；IN-Abs dev 703", "已完成（零样本迁移）", "CAIL 40.19/19.05/26.26；Fact33.46；IN-Abs 54.44/29.23/48.03；英文5空、27触及token上限。", "能做完整训练，但成本较高且任务不完全一致", "仅生成式/零样本分组", "迁移脚本完整；未实现完整训练配方", "论文代码和checkpoint公开", "迁移结果完整", "不能把checkpoint迁移写成论文方法完整复现。", "保留独立轨道；若资源足再做完整SFT+RLVR。", "inabs_extension_work/PREREGISTRATION_PHASE5G_BUDGET_RECIPE.md", "https://aclanthology.org/2025.justnlp-main.11/"],
  [28, "SummaRuNNer（IN-Abs split-safe）", "英文正式基线", "2017", "AAAI 2017", "CCF A", "在IN-Abs训练集重新训练，避免划分泄漏，并按统一词预算解码。", "IN-Abs train/dev 6325/703", "已完成（重训适配）", "67.73/40.73/64.46；当前英文dev三项第一；显著高于E1。", "能做官方test；开发集已足够比较", "可进英文开发集主表", "完整", "原论文/历史代码可得；本地为split-safe重训", "完整开发集结果与配对CI", "不是原checkpoint直接复现；官方test仍封存。", "保留为英文强基线；明确Dev。", "inabs_extension_work/PREREGISTRATION_PHASE4B_SUPERVISED.md", "https://doi.org/10.1609/AAAI.V31I1.10958"],
  [29, "DSDR（IN-Abs统一预算）", "英文正式基线", "2012", "AAAI 2012", "CCF A", "实现论文Algorithm 1线性重构并适配统一预算。", "IN-Abs dev 703", "已完成（算法适配）", "64.75/34.17/61.05；0预算违规。", "能做；无需继续调参", "可进英文开发集主表", "完整", "论文算法公开；无现代checkpoint依赖", "结果完整", "与原论文DUC任务不同。", "保留为经典无监督强基线。", "inabs_extension_work/PREREGISTRATION_PHASE4_UNSUPERVISED.md", "https://ojs.aaai.org/index.php/AAAI/article/view/8202"],
  [30, "CaseSummarizer（IN-Abs）", "英文正式基线", "历史方法；项目按AACL 2022基准实现", "AACL 2022基准论文", "AACL区域会；非ACL主会", "使用AACL仓库中的法律词典/规则系统做统一预算适配。", "IN-Abs dev 703", "已完成（受限适配）", "61.79/33.33/58.73；0预算违规。", "能做；但不值得大规模重写旧系统", "可进英文开发集基线表", "完整可运行的Python3移植/适配", "官方AACL仓库提供旧代码和词典", "结果完整", "原代码Python2/Windows；移植会产生非逐位一致性。", "保留身份说明。", "validation_unverified/CASESUMMARIZER_UKABS_VALIDATION_REPORT_20260923.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [31, "FedSum三随机种子审计", "原任务复现审计", "2025", "AAAI 2025", "CCF A", "按官方代码运行FedAvg/FedSum三种子成对实验，检查论文核心增益。", "论文原任务；3 seeds：666/123/2025", "已完成，但核心优势未复现", "FedAvg 33.82/12.13/30.65；FedSum 33.89/12.20/30.66；Δ仅+.08/.06/.01，远小于论文。", "能继续，但与法律预算任务关联弱", "不可进CAIL正式主表", "完整审计；紧凑证据包存在", "官方代码公开", "完整三种子证据", "绝对量级接近不等于核心结论复现。", "停止继续调参；在复现附录如实报告。", "validation_unverified/FEDSUM_3SEED_VALIDATION_REPORT_20260922.md", "https://ojs.aaai.org/index.php/AAAI/article/view/34129"],
  [32, "SummaRuNNer官方checkpoint UK-Abs审计", "原任务复现审计", "2017", "AAAI 2017", "CCF A", "恢复匹配词表/嵌入，运行官方UK-Abs 100篇测试。", "UK-Abs official test 100", "已完成（宽松数值验证）", "46.23/22.46/22.47；论文基准50.20/20.50/23.70。", "能继续，但预处理/评测细节缺失", "不可作为CAIL主表结果；可作复现附录", "checkpoint与运行链完整", "官方checkpoint可得；部分预处理未发布", "完整逐文档/汇总证据", "不能称精确复现。", "保留“宽松验证”表述。", "validation_unverified/SUMMARUNNER_UKABS_VALIDATION_REPORT_20260923.md", "https://doi.org/10.1609/AAAI.V31I1.10958"],
  [33, "CaseSummarizer UK-Abs审计", "原任务复现审计", "历史方法；AACL 2022基准", "AACL 2022", "AACL区域会；非ACL主会", "将官方Python2/Windows代码移植到Python3并运行UK-Abs 100篇。", "UK-Abs official test 100", "已完成（工程通过，数值未对齐）", "41.88/20.09/33.29；修复版42.10/20.25/33.31；论文44.50/16.60/22.70。", "能继续，但旧环境还原收益有限", "仅复现附录", "移植版完整", "官方旧代码/词典存在", "完整", "环境/分词/ROUGE差异明显。", "不再声称论文数值复现。", "validation_unverified/CASESUMMARIZER_UKABS_VALIDATION_REPORT_20260923.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [34, "DSDR UK-Abs审计", "原任务复现审计", "2012", "AAAI 2012", "CCF A", "在UK-Abs 100篇运行Algorithm 1、lambda=0.1，对比AACL 2022基准数值。", "UK-Abs official test 100", "已完成（工程通过，数值未验证）", "43.33/21.00/38.83；AACL基准48.40/17.40/22.10。", "能继续，但不是DSDR原论文任务", "仅复现附录", "完整", "算法描述充分；比较数来自后续AACL基准", "完整", "不能把差异归因于原论文；评测实现不同。", "保留为工程审计。", "validation_unverified/DSDR_UKABS_VALIDATION_REPORT_20260924.md", "https://ojs.aaai.org/index.php/AAAI/article/view/8202"],
  [35, "MMR UK-Abs审计", "原任务复现审计", "历史方法；AACL 2022实现", "AACL 2022基准仓库", "经典方法；无单一当前级别", "运行官方AACL MMR；发现72篇超预算后增加完整句、去重修复。", "UK-Abs official test 100", "已完成（工程通过，数值未复现）", "修复后0违规，41.43/19.53/37.37；论文44.0/15.1/20.5。", "能做，但继续追分意义低", "仅复现附录", "完整修复版", "官方AACL实现存在", "完整", "修复改变原实现；原实现又违反统一预算。", "把原始与修复版同时存档，不进正式效果主表。", "validation_unverified/MMR_UKABS_VALIDATION_REPORT_20260924.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [36, "KMM UK-Abs审计", "原任务复现审计", "历史方法；AACL 2022实现", "AACL 2022基准仓库", "基准实现；非独立新会议方法", "运行官方K-mixture排序模块和旧完整流水线，检查预算与数值。", "UK-Abs official test 100", "已完成（部分宽松通过）", "排序模块0违规但分数偏离；旧完整流水线42.99/17.05/18.34 vs 43.0/13.8/20.1。", "能做；但协议冲突仍在", "仅复现附录", "代码可运行", "官方AACL实现存在", "完整", "纯排序与完整旧管线不是同一协议。", "不混合两套结果；仅保留审计说明。", "validation_unverified/KMM_UKABS_VALIDATION_REPORT_20260924.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [37, "LetSum UK-Abs审计", "原任务复现审计", "历史方法；AACL 2022实现", "AACL 2022基准仓库", "经典法律摘要规则方法", "运行官方34%摘要比例、主题配额和TF-IDF逻辑。", "UK-Abs official test 100", "已完成（停止继续调参）", "39.67/13.55/14.83 vs 论文43.7/15.8/23.3。", "技术可继续；收益低", "仅复现附录", "完整可运行", "官方AACL代码存在", "完整", "数值未充分对齐；旧规则系统环境敏感。", "维持停止决定。", "validation_unverified/LETSUM_UKABS_VALIDATION_REPORT_20260924.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [38, "Gist UK-Abs官方模型审计", "原任务复现审计", "历史方法；AACL 2022实现", "AACL 2022基准仓库", "AACL区域会基准", "用官方模型/代码跑UK-Abs 100篇并做Top-K敏感性。", "UK-Abs official test 100", "已完成（论文数值失败）", "61.22/24.66/23.15 vs 论文42.7/13.2/21.5；无统一K可解释。", "可继续但不建议", "仅复现附录", "官方模型/运行链可用", "官方模型与代码存在", "完整", "高R1并不代表复现；K和评测协议不统一。", "停止追分，保留失败复现记录。", "validation_unverified/UNVERIFIED_METHOD_VALIDATION_QUEUE_20260922.md", "https://aclanthology.org/2022.aacl-main.77/"],
  [39, "IGCS公开任务验证", "原任务复现审计", "2025", "TACL 2025", "CCF A期刊", "运行公开checkpoint覆盖可取得的任务，审计第六任务资产许可。", "5个公开任务完成；OpenAsp/DUC资产受限", "部分完成（5/6）", "5个公开任务通过；第6个OpenAsp因DUC/NIST许可阻塞。", "只有取得合法数据许可后可补第6项", "不可写“完整6任务复现”", "5任务代码/资产完整", "官方checkpoint可得；DUC数据受许可限制", "5任务完整；第6任务无结果", "许可问题不能用非官方数据绕过。", "论文如提及，写“5个公开任务验证”。", "validation_unverified/UNVERIFIED_METHOD_VALIDATION_QUEUE_20260922.md", "https://aclanthology.org/2025.tacl-1.74/"],
  [40, "DELTA-Dream中文法律适配", "最新顶会小样本", "2026", "EMNLP 2026主会", "CCF A", "接入DELTA官方核心解码与Dream 7B，适配中文字符长度控制，先跑5篇烟雾测试。", "CAIL 5 cases；目标≤450字符", "已完成5篇；全量停止", "DELTA R1/R2/RL=28.43/11.81/15.70；Expr20.02；均长446.4；0/5超预算；生成退化明显。", "技术能跑，但当前配置不值得全量942", "不可进主表；仅诊断", "核心快照+适配器完整", "官方代码存在；本地已vendor核心", "5篇证据完整；无全量结果", "5篇太小；Dream中文法律生成质量差；不能据此否定原方法。", "保留烟测；若换强中文扩散底座，需作为新配置重新预注册。", "experiments/recent_baselines_20261031/SMOKE5_REPORT.md", "https://aclanthology.org/2026.emnlp-main.121/"],
  [41, "Black-box LCG上限版", "最新顶会小样本", "2025", "ACL 2025主会长文", "CCF A", "用Qwen2.5-7B做黑盒长度控制适配；先用“不超过450字符”上限目标测试。", "CAIL 5 cases", "已完成5篇；MHIS未触发", "R1/R2/RL=47.72/28.10/32.29；Expr21.56；均长212.8；0超预算；与直接生成相同。", "能做，但上限目标未激活原方法核心机制", "不可进主表", "自实现适配完整", "论文正式页可得；CAIL适配非官方", "5篇证据完整", "原方法偏精确长度控制；上限任务导致链不启动。", "不扩全量；只用于说明任务目标不匹配。", "experiments/recent_baselines_20261031/LCG_SMOKE5_REPORT.md", "https://aclanthology.org/2025.acl-long.825/"],
  [42, "Black-box LCG精确长度探索版", "最新顶会小样本", "2025", "ACL 2025主会长文", "CCF A", "把目标改为接近指定长度，探索MHIS/迭代控制在中文法律摘要上的行为。", "CAIL 5 cases", "已完成5篇探索", "R1/R2/RL=52.53/34.95/38.87；Expr27.27；均长422.6；2/5超450。", "能做全量，但预算违规与调用成本需先解决", "当前不可进主表", "自实现适配完整", "论文正式页可得；非作者CAIL实现", "5篇证据完整", "探索协议改变了“不超过450”目标；2/5违规。", "若继续，先锁定硬预算后处理和调用上限，再跑30篇门槛。", "experiments/recent_baselines_20261031/LCG_SMOKE5_REPORT.md", "https://aclanthology.org/2025.acl-long.825/"],
  [43, "HiGoE", "最新方法候选", "2026", "ACL 2026主会长文", "CCF A", "尚未运行；拟比较层次证据图检索/生成与本文候选选择。", "计划CAIL或IN-Abs", "未运行", "无本地结果。", "能做，但适配重、需要生成模型和人工事实评估", "运行成功后放生成/混合方法分组", "无本地实现", "官方论文页提供代码链接；尚未验证能否复现", "无", "任务和语言迁移大；自动Expression F1不足以评价证据正确性。", "优先阅读并做10篇端到端可运行性审计，再决定全量。", "—", "https://aclanthology.org/2026.acl-long.902/"],
  [44, "PACO", "最新方法候选", "2026", "ACL 2026主会长文", "CCF A", "尚未运行；拟比较MCTS多属性规划与轻量预算选择。", "计划CAIL小样本", "未运行", "无本地结果。", "能做，但多步LLM调用成本高；属性需合法适配", "仅在形成冻结适配后进入生成式分组", "无本地实现", "论文正式页可得；代码状态待核", "无", "不能把新增法律属性写成原论文原生能力。", "资源足时排在HiGoE之后；先10篇成本/稳定性门槛。", "—", "https://aclanthology.org/2026.acl-long.845/"],
  [45, "Principled Content Selection（DPP）", "最新方法候选", "2025", "ACL 2025主会长文", "CCF A", "尚未运行；考虑只迁移DPP多样性选择模块，与MMR式冗余控制对比。", "原任务为多文档；计划CAIL选择器对照", "未运行", "无本地结果。", "模块级能做；完整方法与单文档任务不匹配", "只可作为“DPP选择器适配”，不称完整复现", "无本地实现", "论文正式页可得；代码状态待核", "无", "多文档个性化设定与本文单文档法律摘要差异大。", "若做，只实现冻结分数上的DPP选择器并清楚改名。", "—", "https://aclanthology.org/2025.acl-long.1445/"],
  [46, "Context-Aware Hierarchical Merging", "补充候选", "2025", "Findings of ACL 2025", "Findings；非ACL主会长文", "尚未运行；作为长文分层合并/事实性相关工作候选。", "计划IN-Abs/CAIL小样本", "未运行", "无本地结果。", "能做；LLM成本较高", "不优先进入主表", "无本地实现", "论文正式页可得；代码状态待核", "无", "发表类别不能写成ACL主会长文；生成式协议差异大。", "先用于相关工作；资源富余再做。", "—", "https://aclanthology.org/2025.findings-acl.289/"],
  [47, "StrucSum", "补充候选", "2026", "Findings of EACL 2026", "Findings；非EACL主会", "尚未运行；训练免费图结构推理与抽取式设置较接近。", "计划CAIL小样本", "未运行", "无本地结果。", "相对可做；需中文提示和候选适配", "完成冻结适配后可进补充表", "无本地实现", "论文正式页可得；代码状态待核", "无", "不能写成EACL主会；LLM提示与图构造需明确是适配。", "在未跑候选中优先级高于PACO，先做10篇可行性。", "—", "https://aclanthology.org/2026.findings-eacl.192/"]
];

// Only retain experiments that directly compare the proposed method with
// published baselines or currently considered recent methods. Mechanism
// ablations, metric audits, runtime-only studies, and original-task
// reproduction diagnostics are intentionally excluded from this deliverable.
const comparisonIds = new Set([
  1, 2,
  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  40, 41, 42, 43, 44, 45, 46, 47,
]);
const comparisonRows = rows
  .filter((row) => comparisonIds.has(row[0]))
  .map((row, index) => [index + 1, ...row.slice(1)]);

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("论文对比实验");
sheet.showGridLines = false;
sheet.tabColor = "#1F4E78";

const startRow = 1;
const endRow = startRow + comparisonRows.length;
sheet.getRange(`A${startRow}:S${startRow}`).values = [columns];
sheet.getRange(`A${startRow + 1}:S${endRow}`).values = comparisonRows;

sheet.getRange(`A${startRow}:S${startRow}`).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#B4C7DC" }
};
const dataRange = sheet.getRange(`A${startRow + 1}:S${endRow}`);
dataRange.format = {
  font: { name: "Arial", size: 9, color: "#1F2937" },
  verticalAlignment: "top",
  horizontalAlignment: "left",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D9E2F3" }
};
sheet.getRange(`A${startRow + 1}:A${endRow}`).format.horizontalAlignment = "center";
sheet.getRange(`D${startRow + 1}:F${endRow}`).format.horizontalAlignment = "center";
sheet.getRange(`I${startRow + 1}:I${endRow}`).format.horizontalAlignment = "center";
sheet.getRange(`L${startRow + 1}:M${endRow}`).format.horizontalAlignment = "center";

const table = sheet.tables.add(`A${startRow}:S${endRow}`, true, "ExperimentInventoryTable");
table.style = "TableStyleMedium2";
table.showFilterButton = true;
table.showBandedRows = true;

sheet.freezePanes.freezeRows(startRow);
sheet.freezePanes.freezeColumns(2);

const widths = [7, 27, 16, 14, 25, 22, 42, 32, 22, 48, 31, 23, 28, 32, 27, 39, 42, 52, 46];
for (let i = 0; i < widths.length; i++) {
  sheet.getRangeByIndexes(0, i, endRow, 1).format.columnWidth = widths[i];
}
sheet.getRange(`${startRow}:${startRow}`).format.rowHeight = 42;
sheet.getRange(`${startRow + 1}:${endRow}`).format.rowHeight = 92;

const statusRange = sheet.getRange(`I${startRow + 1}:I${endRow}`);
statusRange.conditionalFormats.add("containsText", { text: "已完成", format: { fill: "#E2F0D9", font: { color: "#2E6B2E", bold: true } } });
statusRange.conditionalFormats.add("containsText", { text: "未运行", format: { fill: "#E7E6E6", font: { color: "#595959" } } });
statusRange.conditionalFormats.add("containsText", { text: "预注册", format: { fill: "#DDEBF7", font: { color: "#1F4E78", bold: true } } });
statusRange.conditionalFormats.add("containsText", { text: "停止", format: { fill: "#FCE4D6", font: { color: "#9C0006", bold: true } } });
statusRange.conditionalFormats.add("containsText", { text: "部分完成", format: { fill: "#FFF2CC", font: { color: "#7F6000", bold: true } } });

const mainRange = sheet.getRange(`L${startRow + 1}:L${endRow}`);
mainRange.conditionalFormats.add("containsText", { text: "可（", format: { fill: "#E2F0D9", font: { color: "#2E6B2E", bold: true } } });
mainRange.conditionalFormats.add("containsText", { text: "仅", format: { fill: "#FFF2CC", font: { color: "#7F6000" } } });
mainRange.conditionalFormats.add("containsText", { text: "不可", format: { fill: "#F4CCCC", font: { color: "#9C0006", bold: true } } });

const codeRange = sheet.getRange(`M${startRow + 1}:M${endRow}`);
codeRange.conditionalFormats.add("containsText", { text: "完整", format: { fill: "#E2F0D9", font: { color: "#2E6B2E" } } });
codeRange.conditionalFormats.add("containsText", { text: "部分", format: { fill: "#FFF2CC", font: { color: "#7F6000" } } });
codeRange.conditionalFormats.add("containsText", { text: "无本地", format: { fill: "#F4CCCC", font: { color: "#9C0006" } } });

workbook.recalculate();
const preview = await workbook.render({ sheetName: "论文对比实验", range: "A1:J14", scale: 1, format: "png" });
await fs.writeFile("/tmp/paper_comparison_experiments_preview.png", new Uint8Array(await preview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

console.log(JSON.stringify({ outputPath, rowCount: comparisonRows.length, preview: "/tmp/paper_comparison_experiments_preview.png" }));
