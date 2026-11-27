# BreathLLM

BreathLLM 是基于时序大语言模型的睡眠呼吸事件连续检测与五分类研究项目。当前主线是 **UCDDB｜60 秒因果历史窗、1 秒步长｜五分类｜轨道 A 整夜连续监测**。模型每秒发布 Normal、阻塞性暂停 OSA、中枢性暂停 CSA、混合性暂停 MSA、低通气 Hypopnea 的概率。

**本仓库不包含任何数据集。使用者必须自行下载 UCDDB，并根据说明准备本地数据目录。** 波形、逐人标注、模型权重和整夜预测均通过本地路径提供或生成。 [UCDDB 1.0.0 官方下载](https://physionet.org/content/ucddb/1.0.0/) · [外部资产清单](docs/资产审计与外部依赖.md)

## 模型和开发协议

四路 8 Hz 输入依次为气流、胸带、腹带、SpO₂，形状为 `[B,4,480]`。冻结的 OpenTSLM/Gemma 读取时序词元，T03 的分层读出估计异常概率、暂停条件概率和暂停亚型分布。窗口内最后五秒各有一个因果监督位置，实际发布当前秒。日常开发固定为外折 1 下的 20 人三内折，种子 4121–4123，最多 512 步。 [模型设计](docs/T03模型设计.md) · [实验协议](docs/实验协议.md) · [OpenTSLM 源码](https://github.com/OpenTSLM/OpenTSLM) · [OpenTSLM 论文](https://proceedings.mlr.press/v306/langer26a.html)

## 安装和检查

使用 Python 3.12。历史运行环境使用 PyTorch 2.4.1、CUDA 12.4 和 RTX 4090。安装对应的 CUDA wheel 后，安装剩余依赖。命令里的设备可以改成其他 CUDA 编号。合成测试和若干传统基线支持 CPU。

```bash
git clone https://github.com/zhupeng12345/breathllm.git
cd breathllm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python breathllm.py smoke
```

### 数据

从 PhysioNet 1.0.0 取得开发受试者的 `*.rec`、`*_stage.txt` 和 `*_respevt.txt`，平铺到 `data/ucddb/`。可以直接下载 [锁定清单](config/外部资产.json) 中 60 个文件。数据已经放在别处时，设置 `BREATHLLM_DATA_ROOT` 指向包含这些文件的目录。预处理核对原始文件散列，重建 RESP 时钟、睡眠掩码、60 秒窗与末五标签。

```bash
python breathllm.py prepare
```

### 模型权重

在 Hugging Face 接受 Gemma 的访问条款，并使用 `hf auth login`。以下固定版本的文件默认保存于 `weights/`。使用其他目录时设置 `BREATHLLM_WEIGHTS_ROOT`，下载目标与该路径对应。

```bash
hf auth login
hf download google/gemma-3-270m --revision 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1 --include README.md added_tokens.json config.json generation_config.json model.safetensors special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json --local-dir weights/gemma-3-270m
hf download OpenTSLM/gemma-3-270m-sleep-sp README.md model_checkpoint.pt --revision d82f13e50d4b192f72ca38f15c21dc60115aace2 --local-dir weights/gemma-3-270m-sleep-sp
python breathllm.py weights
python breathllm.py smoke --real --device cuda:0
```

### 重建当前基座和运行对照

本地生成位置默认是 `outputs/`，也可设置 `BREATHLLM_OUTPUT_ROOT`。下面第一条重建一个 T03，第二条覆盖全部三折三种子并复用身份一致的已有结果。基线可单独运行一项，也可按默认参数运行完整 20 项。

```bash
python breathllm.py t03 --folds 1 --seeds 4121 --device cuda:0
python breathllm.py t03 --folds 1 2 3 --seeds 4121 4122 4123 --device cuda:0
python breathllm.py baseline --methods XGBoost --folds 1 --seeds 4121 --device cpu
python breathllm.py baseline --help
```

T03 本地重建所需的权重、标签、训练索引和整夜预测通过以上流程准备。拥有经授权的历史库时，也可用 `BREATHLLM_HISTORICAL_T03` 指向该库，逐条校验原权重、概率和收据的散列。复现值可能随硬件和浮点计算后端产生偏差。历史 BF16 和本地独立 FP32 结果应依各自协议解读。 [归档字段及来源](docs/资产审计与外部依赖.md)

### 读取和复核现有研究

R0/R1/R2 需要九个 T03 封存结果。R0 是只读覆盖与瓶颈审计，R1 是已完成的 128 步排序与校准实验。R2 检查低通气漏检和暂停分型错误，PAEL 是最新已结案的 FP32 探针。

```bash
python breathllm.py r0 --device cuda:0
python breathllm.py r1 --device cuda:0
python breathllm.py r2 --device cuda:0
python breathllm.py pael --steps 128 --device cuda:0
python breathllm.py pael --steps 512 --device cuda:0
```

R0 可加 `--statistics-only` 只复算归档。PAEL 的 512 步入口从本机已封存的 128 步检查点继续。真实环境验收已覆盖开发标签重建、短片段前向及 R0 九组只读复算。完整 T03 重训和 Baseline、R1、PAEL 拟合属于后续科研运行。 [交接验收](docs/资产审计与外部依赖.md)

## 关键结果与研究状态

下表为 UCDDB 20 人、三内折、三种子的历史整夜开发结果。

| 方法 | 逐秒 Macro-F1 | 检出 Event F1 | 分型 Event F1 |
| --- | ---: | ---: | ---: |
| T03 | 0.3271 | 0.6170 | 0.3653 |
| T03-KD | 0.3120 | 0.6113 | 0.3634 |
| XGBoost | 0.3590 | 0.6964 | 0.4671 |

WD-AMT 有过事件级增益，同时带来误报和种子依赖。MC-DPT v2 与 DT-EMM v2 未通过各自机制门槛。R2 把低通气漏检定位在异常分支。PAEL 的 512 步对照已完成，无组晋级。正式基座仍为 T03，下一轮算法方案待研究负责人确定。 [完整主表](docs/关键实验结果.md) · [历史创新](docs/历史创新与失败结论.md) · [当前任务](docs/当前研究任务.md)

## 阅读顺序与引用

1. [项目交接指南](docs/项目交接指南.md)
2. [实验协议](docs/实验协议.md)与[模型设计](docs/T03模型设计.md)
3. [关键实验结果](docs/关键实验结果.md)与[历史创新](docs/历史创新与失败结论.md)
4. [当前研究任务](docs/当前研究任务.md)与[资产审计](docs/资产审计与外部依赖.md)
5. [参考文献与 20 种基线映射](docs/参考文献.md)

UCDDB 数据库遵循 **ODC-By 1.0**。使用时引用 PhysioNet 的数据记录和标准引用，OpenTSLM 与模型按各自的代码许可证及模型卡使用。 [UCDDB 许可](https://physionet.org/content/ucddb/view-license/1.0.0/) · [第三方许可证和版本](docs/资产审计与外部依赖.md)
