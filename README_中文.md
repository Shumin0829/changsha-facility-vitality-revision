# 体育设施活力分析：代码包说明

## 交付状态

本仓库公开此前 `1009` 代码包中的**修回重分析与图5—11绘图代码**。
仓库：https://github.com/Shumin0829/changsha-facility-vitality-revision
这不是未经改动的2025年探索性Notebook，也不包含上游GIS／热力图提取或后来新增的
GWR／MGWR比较试算。发布范围和仅涉及打包／文档的调整见 `RELEASE_NOTES.md`。
没有声明DOI，亦未擅自添加MIT等额外软件复用许可证；公开查看不在这里被表述为
“已采用开源许可证”。许可说明见 `NOTICE.md`。
包内不包含真实设施名单、坐标原表、热力图、POI 原始库、私人聊天、原投稿 CSV 或真实模型。
`examples` 全部是程序生成的合成数据，400 行并非论文的 1,650 个设施。
烟测通过只能说明接口和代码流程能运行，不能证明任何真实科学结论。

## 样本和变量

正式输入应为 1,650 条、四级数量 13／40／182／1,415，可用 `--expected-n 1650` 检查。
两个目标来自**一个代表工作日、一个代表周末日**的热力点值，并非整周平均。
单日对比不能证明一般性工作日／周末行为规律；热力值也不等于已核验的体育参与人数。

真实原附件 0727 有 25 个特征，包含 SID。0728 删除的是 SD、ResD、TD、PFD、DID
五个特征，保留 20 个特征，并未删除设施。另一份工作副本增加 POID 后有 26 个特征。
本代码的 26 特征及训练折内 VIF 是扩展敏感性分析，**不是原 25 特征模型的严格复现**。
代码按字段名读取，避免位置选列在 30／31 列文件之间产生 SID 错选。

主表必需字段为 `x`、`y`、`等级`、`SV_weekday`、`SV_weekend`，以及以下 20 个特征：

```text
BD PR MD ED ResiD ECED SID SI Mix FL SFA NDVI NDWI DSS DBS IRD RD DCBD DRA DSF
```

`x/y` 是 WGS84 经纬度；研究区距离计算固定用 UTM 49N（EPSG:32649），不适用于任意地区。
`等级` 为 1—4；`FL` 是服务半径编码，1／2／3／4 对应 5000／2500／1000／500 米，
不是普通连续等级。允许别名 ID→IRD、POIL 或 SFR→SFA，冲突别名会报错。
保持原始单位；代码不自动归一化 Y，也不自动换算 DRA 等距离单位。
指标定义、GIS 计算、熵公式及单位仍须由原数据方法说明支持，字段验证不能替代这些核查。

扩展表必须提供全部 26 个特征并与主表顺序和数值对应；保留扩展表原列顺序，以便复现
带随机列抽样的树模型。区县表需提供 x、y、等级、Y_点活、adname，Y_点活须对应
SV_weekday。区县表仅用于五城区子样本，不用于反推或强行凑出历史 1,400 条。
不得擅自删除缺失行或重复点；校验器会停止并要求作者明确处理。

## 分析协议

比较 OLS、Ridge、随机森林、GBDT、XGBoost、LightGBM。外层 5 折留出预测，
内层 3 折按平均 RMSE 调参，每个需调参模型随机搜索 12 组，种子为 7；OLS 不调参。
线性模型标准化仅在训练折拟合。没有外层测试集早停或测试集选参数。

随机验证用打乱的 KFold；空间验证用固定 UTM 网格、2 km 块和训练点中心距测试点中心
500 m 排除，在内外层均执行。敏感性包括 1／4 km 块、1000 m 排除、扩展 26 特征、
训练折内 VIF、五城区子集。这不是重算设施服务缓冲区；大型服务范围仍可能重叠。
折外汇总指标与折均值／标准差分开导出。SHAP 是描述性模型归因，不是因果或显著性证据。
残差 Moran 采用对称行标准化 kNN、999 次置换及汇总 BH-FDR。
模型定义、搜索空间和折叠协议保留已执行版本的核心实现。

## 使用和限制

英文 README 提供完整字段表和运行命令。默认 UTF-8；历史表需显式 `--encoding gbk`。
`--input` 和 `--out` 都必填，不内置个人路径，也不复制真实数据进代码目录。

```sh
python smoke_test.py
python reanalysis.py --input data/main.csv --encoding gbk --out results/main --expected-n 1650 --stage main
python verify_results.py --input data/main.csv --encoding gbk --out results/main
```

全敏感性运行还必须显式提供 `--extended-input` 和 `--facility-input`。
输入哈希或分析配置改变时应换一个输出目录，不能混用旧检查点。
`verify_results.py` 默认只核对保存预测和折叠；仅对自己生成的模型才可加
`--trust-local-models`。joblib／pickle 文件可执行代码，不能加载陌生来源模型。
`--refit` 会额外拟合两个模型，轻量烟测不需要，也未在本次真实数据上执行。

模型分析接口仅对合成数据做 schema／协议／OLS 烟测；绘图接口另用已有授权输入和
保存的折外 SHAP 做了完整渲染检查，**没有重跑真实 180 个外层模型**。
包中的 requirements-lock 是既有运行环境记录，不保证每个平台均可安装所有版本。
真实运行输出可能包含位置及目标值，必须按数据权限保管，不能随代码仓库自动公开。
真实数据和运行输出仍不公开；额外软件复用许可和上游单位核查仍需作者另行确认。
历史验证JSON保留原测试范围，不代表此次重新运行真实分析，也不代表已经取得存档DOI。
`build_release.py`仅生成并核对本地压缩包，不执行上传或证明远端公开状态。

## 图 5—11 脚本

`plot_revision_figures_1009.py` 仅读取既有折外 XGBoost SHAP，不训练模型。必须显式提供
`--analysis-root`、`--input`、`--out`；主表须为原顺序的 1,650 条，必须匹配兼容分析
记录的输入哈希（main_input，或旧主表键 0728.csv；不读取／复写原 manifest 的私人路径）。
旧输出缺少兼容哈希时默认停止；作者独立核实逐行身份后才可显式加
`--allow-unverified-input-hash`，并会留下“哈希未核验”记录，不能把这个选项当作身份核验。
保留原特征和 SHAP 值；SFA 仅用明确标记的 symlog 显示轴，
DRA 不换算单位。脚本图注固定为本研究的两个观测日期和验证协议，不是通用日期模板。
`--metadata-only` 只写拟用图注和绘图约定，不代表数据已读入或图片已经绘制。
DRA、SFA 的上游单位仍需作者确认。

原始绘图脚本已在单独 Python 3.12.8 环境生成图 5—11，并完成逐图视觉 QA。
版本记在 `render_environment_reference.json`；公共副本 12 个绘图／图注函数 AST
完全相同，只改了输入加载和 CLI。`requirements-plot.txt` 锁定该独立绘图环境的
直接依赖，不代表与核心模型环境混装验证过。先建立并激活独立环境，不要替换模型环境
的 numpy／pandas 版本：

```sh
python -m venv .plot-venv
```

激活 `.plot-venv` 后再显式安装和运行：

```sh
python -m pip install -r requirements-plot.txt
python plot_revision_figures_1009.py --analysis-root results/main --input data/main.csv --encoding gbk --out outputs/figures
```

会输出 SVG、PDF、600 dpi PNG、拼图及源数据／QA 表。这些运行结果可能包含真实数据，
不能自动随代码上传。公共副本已在上述独立绘图环境完成实际端到端运行：七张 PNG 与
完成逐图视觉 QA 的冻结版均 SHA256／逐像素完全相同，归因份额、点数、输入哈希及
12 个绘图函数均匹配。实际 QA 图件与源数据留在私有 QA 目录，未纳入代码包。
此检查没有重拟合模型，也不能替代上游单位核查或证明新的科学结论。

