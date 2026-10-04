# ms-swift 训练与评估数据

`processed/<dataset>/train.jsonl`、`validation.jsonl` 和 `test.jsonl` 是保留的
Canonical 中间表示。本目录由 [Manifest](../../manifests/train_manifest.json) 生成：

- `train.jsonl`：所有启用数据集的 Canonical train 样本。
- `eval.jsonl`：各数据集 validation 后接 test 中**有答案**的样本。
- `datasets/<dataset>.train.jsonl` 与 `datasets/<dataset>.eval.jsonl`：按数据集拆分的副本。
- `stats.json`：各数据集、来源 split、跳过原因和图片路径检查结果。

输出使用 ms-swift 的 `messages` 与 `images` 格式。转换按 Manifest 顺序写入，
不去重、不打乱，也不设置采样权重。DocVQA test 等无答案样本会记录为跳过。

从项目根目录重新生成 Manifest 和数据：

```bash
python3 -m src.training.manifest
python3 -m src.training.ms_swift_adapter \
  --manifest manifests/train_manifest.json
```

小规模检查（每个数据集的每个来源 split 最多读取 100 条）：

```bash
python3 -m src.training.ms_swift_adapter \
  --manifest manifests/train_manifest.json \
  --limit 100 \
  --output-dir tmp/ms_swift_smoke
```

ms-swift 数据使用示例；`MODEL` 由具体实验设置：

```bash
swift sft \
  --model "$MODEL" \
  --dataset training/ms_swift/train.jsonl \
  --val_dataset training/ms_swift/eval.jsonl
```

不要同时传入合并文件和其中的数据集拆分文件，否则样本会重复。

## 修改图片路径前缀

先预览匹配数量：

```bash
python3 training/change_image_prefix.py \
  --old-prefix /mnt/data/william/mmllm_jinshan/dataset_built/processed \
  --new-prefix /new/location/processed \
  --dry-run
```

确认后，可使用 `--in-place` 修改本目录全部 JSONL（包括合并文件和按数据集拆分文件），
或使用 `--output-dir /new/output/directory` 另存一份。脚本只改 `images` 数组，
不会改 `processed/` 的 Canonical JSONL，也不会检查新位置的图片是否已经存在。
