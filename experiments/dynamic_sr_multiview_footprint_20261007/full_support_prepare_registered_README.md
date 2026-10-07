# 完整原生父模型的已注册 X 支持准备

`full_support_prepare_registered.py` 只为已选择的 X/MX 分支准备完整训练相机的原始 HR 几何矩和冻结支持缓存。未选择 X 的方法不要求这些额外依赖，也不生成缓存。方案仍须由完整短窗证据选型，最多两个候选；本入口不进行方法选择或确认场景 HR 质量评价。

这里的 HR 指父模型在高分辨率相机网格上渲染的几何矩 `[A,M1,M2]`，不是 HR 真值图像。矩估计、先汇聚原始矩再归一化、软权重、硬遮挡筛选和 mipmap 公式均使用已冻结 `support_cache.py`。新入口仅适配数据身份、相机数量、网格、父模型计数和输出路径。旧完整与短窗源码、缓存及校准均保留。

| 场景 | 原生 LR（宽×高） | HR 网格（宽×高） | 实际训练相机 | 校准锚点 |
| --- | --- | --- | ---: | ---: |
| Cook | 336×252 | 1344×1008 | 20 | 80 |
| Cut | 336×252 | 1344×1008 | 19 | 76 |
| MeetRoom discussion / VR | 320×180 | 1280×720 | 12 | 48 |
| Coffee | 336×252 | 1344×1008 | 17 | 68 |

每个合法训练相机在固定帧 `0/100/200/299` 各有一个锚点。每个锚点消费同刻三个相机的 RGB 和矩；上表对应 RGB/矩校准次数分别为 `240/228/144/204`。Cook 完整数据含合法 cam01，因此是 80 个锚点；历史短窗 train76 不能移用。Flame 使用其最终已验收 manifest 的实际训练相机数；缺数据时不能宣称准备就绪。

完整父模型必须是原生 14 项状态与一个含八组参数的 Adam。batch2 场景完整父模型的 RGB 计数为 34,000；MeetRoom 与 Coffee 的 batch4 为 68,000。两者均要求 coarse3000、fine14000、backward17000、Adam16999。MeetRoom 是固定 Wu generic N3DV 配置向新域的显式迁移，不是作者原 MeetRoom 场景复现。

生产缓存使用新的 `output/dynamic_sr_multiview_footprint_20261007/full_support_registered/` 独立版本。完整 manifest、独立 seed 父模型、配对日程、选择证据和源代码都须绑定 SHA。旧 U6000、短窗 1140 矩、旧 train76 校准及不同父模型/种子的支持缓存均不能替代。

CPU 合同检查：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 \
  experiments/dynamic_sr_multiview_footprint_20261007/full_support_prepare_registered.py \
  --mode contract
```

该检查只读取实际 manifest 的元数据并使用明确标注的依赖夹具；不导入 tensor/模型，不渲染，不扫全部图像 SHA。它核对原始 HR 导出及冻结支持函数 AST、网格、相机/锚点、原生父计数，以及错误身份和缺依赖的拒绝。真实 CUDA 相机、原始矩渲染、缓存生成和零更新模型/RNG验收仍由根代理单独分配 GPU 执行。

`--mode plan` 输出就绪或缺项，`--mode register` 固定独立版本的计划；二者不派发 GPU。实际 `--mode prepare` 必须满足选择、完整父模型及日程依赖，并由操作者显式提供 GPU UUID、共享锁和资源已解决标记。新校准入口为 `full_refine_registered.py --mode calibrate`，消费同一个完整父/数据/日程支持索引，不凭确认 HR 挑选权重。
