# pre-process
python setup.py build_ext --inplace

# 下载现成的语音模型，中性发音，公共语料库

# 将所有视频（无论是上传的还是下载的，且必须是.mp4格式）提取音频
python scripts/video2audio.py

# 将所有音频（无论是上传的还是从视频抽取的，必须是.wav格式）去噪，人声分离等处理
# 无背景的声音，或有条件在录音棚录制最佳
python scripts/denoise_audio.py

# 分割并标注长音频, whisper识别语音，转录拆分为多段wav和对应标注
python scripts/long_audio_transcribe.py

# 现成的语音模型采样率可能与辅助数据不同，需要重采样
python scripts/resample.py

# tensorboard --logdir "./OUTPUT_lyh"
# netstat -ano | findstr :8000

# training，训练需要使用GPU
python scripts/preprocess_v2.py
python finetune_speaker_v2.py -m "./OUTPUT_lyh" --max_epochs 2000 --drop_speaker_embed True

tts --text "大家好，欢迎来到开星果乐园。听，这是来自脑博士的声音。" --speaker_idx 0 --out_path output/speech.wav --model_path /Users/yongle/Library/CloudStorage/GoogleDrive-yongle.work@gmail.com/我的云端硬盘/yjc_10k/G_latest.pth --config_path /Users/yongle/Library/CloudStorage/GoogleDrive-yongle.work@gmail.com/我的云端硬盘/yjc_10k/finetune_speaker.json

# inference，推理可以使用CPU进行 
python scripts/VC_inference.py --model_dir ./pretrained_models/lyh-colab/G_latest.pth
python scripts/VC_inference.py --model_dir ./pretrained_models/lyh_48mins/G_latest.pth --config_dir ./pretrained_models/lyh_48mins/finetune_speaker.json 
python scripts/VC_inference.py --model_dir ./lyh_0424/G_latest.pth --config_dir ./lyh_0424/finetune_speaker.json
python scripts/VC_inference.py --model_dir /Users/yongle/project/Python/VITS-fast-fine-tuning/colab/G_3500.pth --config_dir /Users/yongle/Library/CloudStorage/GoogleDrive-yongle.work@gmail.com/我的云端硬盘/yjc_10k/finetune_speaker.json
python scripts/VC_inference.py --model_dir  /Users/yongle/Library/CloudStorage/GoogleDrive-yongle.work@gmail.com/我的云端硬盘/yjc_0425_3/OUTPUT_MODEL/G_latest.pth --config_dir /Users/yongle/Library/CloudStorage/GoogleDrive-yongle.work@gmail.com/我的云端硬盘/yjc_0425_3/finetune_speaker.json

python cmd_inference.py -m 模型路径 -c 配置文件路径 -o 输出文件路径 -l 输入的语言 -t 输入文本 -s 合成目标说话人名称
python cmd_inference.py -m "./OUTPUT_lyh" -c "./OUTPUT_lyh/finetune_speaker.json" -o "./OUTPUT_lyh/speech.wav" -l "zh" -t "大家好，欢迎来到开星果乐园。听，这是来自罗永浩的声音。" -s "lyh"