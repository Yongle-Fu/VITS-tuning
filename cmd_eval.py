"""该模块用于计算语音质量指标
使用方法
python cmd_eval.py
"""

import pesq
import pystoi
import librosa
import mir_eval
import numpy as np


def eval_diff(ref_audio, synth_audio, sr):
    # 计算STOI得分
    # STOI (Short Time Objective Intelligibility)得分范围为0到1之间，分数越高表示语音的清晰度和可懂度越好。
    # 常见的STOI得分范围在0.2到0.9之间，通常在0.6以上的得分被认为是相当好的。
    # Trim audio signals to have the same length
    min_len = min(len(ref_audio), len(synth_audio))
    ref_audio = ref_audio[:min_len]
    synth_audio = synth_audio[:min_len]
    stoi_score = pystoi.stoi(ref_audio, synth_audio, sr)
    print("STOI score:", stoi_score)

    # 计算SDR、SIR、SAR指标
    # Calculate SDR, SIR, SAR scores
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
        np.array(synth_audio), np.array(ref_audio)
    )
    print("SDR: ", sdr, "SIR: ", sir, "SAR: ", sar)

    # 计算PESQ得分
    # PESQ得分的范围在-0.5到4.5之间。其中-0.5表示质量极差，4.5表示质量非常好。一般来说，得分越高，语音质量越好。具体而言，PESQ得分如下：
    # - 大于4.0：语音质量非常好，听起来就像真实录制的语音一样。
    # - 3.0到4.0之间：语音质量较好，听起来有些许差异，但不影响理解。
    # - 2.0到3.0之间：语音质量一般，听起来有较明显的差异，但仍然可以理解。
    # - 1.0到2.0之间：语音质量较差，听起来有很大的差异，可能会影响理解。
    # - 小于1.0：语音质量极差，听起来就像是噪声或乱码。
    pesq_score = pesq.pesq(sr, ref_audio, synth_audio)
    print("PESQ score:", pesq_score)


if __name__ == "__main__":
    # 读取音频文件
    ref_audio, sr = librosa.load("output_wav/yjc_150轮_单人.wav", sr=16000)
    synth_audio_100, sr2 = librosa.load("output_wav/yjc_100轮_单人.wav", sr=16000)
    synth_audio_50, _ = librosa.load("output_wav/yjc_50轮_单人.wav", sr=16000)
    synth_audio_20, _ = librosa.load("output_wav/yjc_20轮_单人.wav", sr=16000)
    # print(sr, sr2)

    eval_diff(ref_audio, synth_audio_100, sr)
    eval_diff(ref_audio, synth_audio_50, sr)
    eval_diff(ref_audio, synth_audio_20, sr)
