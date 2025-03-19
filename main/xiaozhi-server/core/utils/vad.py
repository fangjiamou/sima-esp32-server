from abc import ABC, abstractmethod
from config.logger import setup_logging
import opuslib_next
import time
import numpy as np
import torch

TAG = __name__
logger = setup_logging()

class VAD(ABC):
    @abstractmethod
    def is_vad(self, conn, data):
        """检测音频数据中的语音活动"""
        pass


class SileroVAD(VAD):
    def __init__(self, config):
        logger.bind(tag=TAG).info("SileroVAD", config)
        # torch.hub.load 是PyTorch提供的一个函数，用于从指定的源加载预训练模型。
        self.model, self.utils = torch.hub.load(repo_or_dir=config["model_dir"],
                                                source='local',
                                                model='silero_vad',
                                                force_reload=False)
        # get_speech_timestamps 用于获取语音的时间戳
        (get_speech_timestamps, _, _, _, _) = self.utils

        # 创建一个 opuslib_next 库的解码器实例，用于解码OPUS格式的音频数据，采样率为16000H，1表示单声道音频
        self.decoder = opuslib_next.Decoder(16000, 1)
        # 语音活动检测的阈值，用于判断音频中是否存在语音
        self.vad_threshold = config.get("threshold")
        # 最小静默时长阈值，用于判断语音是否结束
        self.silence_threshold_ms = config.get("min_silence_duration_ms")

    def is_vad(self, conn, opus_packet):
        """
        主要用于处理音频数据的缓冲区，检测语音活动，并根据语音活动的状态更新连接对象的状态
        """
        try:
            # 解码后的PCM格式音频帧
            pcm_frame = self.decoder.decode(opus_packet, 960)
            conn.client_audio_buffer += pcm_frame  # 将新数据加入缓冲区

            # 处理缓冲区中的完整帧（每次处理512采样点，每个采样点2个字节）
            client_have_voice = False
            while len(conn.client_audio_buffer) >= 512 * 2:
                # 提取前512个采样点（1024字节）
                chunk = conn.client_audio_buffer[:512 * 2]
                # 截断缓冲区，移除已处理的部分
                conn.client_audio_buffer = conn.client_audio_buffer[512 * 2:]

                # 转换为模型需要的张量格式
                # 字节数据 chunk 转换为 np.int16 类型的NumPy数组
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                #  np.int16 类型的数组转换为 np.float32 类型，并将其归一化到 [-1, 1] 范围内
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                # 将NumPy数组转换为PyTorch张量
                audio_tensor = torch.from_numpy(audio_float32)

                # 检测语音活动
                # 输入音频张量和采样率（16000Hz），返回语音存在的概率
                speech_prob = self.model(audio_tensor, 16000).item()
                # 大于语音活动检测的阈值则认为存在语音
                client_have_voice = speech_prob >= self.vad_threshold

                # 如果之前有声音，但本次没有声音，且与上次有声音的时间差已经超过了静默阈值，则认为已经说完一句话
                if conn.client_have_voice and not client_have_voice:
                    stop_duration = time.time() * 1000 - conn.client_have_voice_last_time
                    # 距离上次有声音的时间超过静默阈值，则认为已经说完一句话
                    # 注意，如果用户说话停顿较久，会导致语音识别不准确，所以需要调整静默阈值
                    if stop_duration >= self.silence_threshold_ms:
                        conn.client_voice_stop = True

                # 本次有声音，记录时间
                if client_have_voice:
                    conn.client_have_voice = True
                    conn.client_have_voice_last_time = time.time() * 1000

            return client_have_voice
        except opuslib_next.OpusError as e:
            logger.bind(tag=TAG).info(f"解码错误: {e}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error processing audio packet: {e}")


def create_instance(class_name, *args, **kwargs) -> VAD:
    # 获取类对象
    cls_map = {
        "SileroVAD": SileroVAD,
        # 可扩展其他SileroVAD实现
    }

    if cls := cls_map.get(class_name):
        return cls(*args, **kwargs)
    raise ValueError(f"不支持的SileroVAD类型: {class_name}")
