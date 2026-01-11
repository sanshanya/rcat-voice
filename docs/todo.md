两个 examples 已实现完成并编译通过：

1. speaker_id_gate.rs
多样本注册 (--enroll x.wav --enroll y.wav)，embedding 取平均
强制 mono + 16kHz，自动 stereo 下混
可配置 --threshold, --num-threads, --provider, --json
输出 cosine similarity 值和匹配决策
2. diarize_offline.rs
使用 pyannote 分段 + 3dspeaker embedding 做离线分离
Progress callback 显示进度
输出人类可读表格 + --json 格式
配置 --num-clusters, --min-on, --min-off
文档更新
README.md 新增"说话人识别与分离"章节，含模型下载链接
请下载模型后运行测试：

bash
# Speaker ID
cargo run --example speaker_id_gate --features asr-sherpa -- \
    --model 3dspeaker.onnx --enroll owner.wav --test test.wav
# Diarization
cargo run --example diarize_offline --features asr-sherpa -- \
    pyannote/model.onnx 3dspeaker.onnx meeting.wav