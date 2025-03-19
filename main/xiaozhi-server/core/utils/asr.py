import importlib
import os
import sys
from core.providers.asr.base import ASRProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

def create_instance(class_name: str, *args, **kwargs) -> ASRProviderBase:
    """
    工厂方法创建ASR实例
    """

    if os.path.exists(os.path.join('core', 'providers', 'asr', f'{class_name}.py')):
        lib_name = f'core.providers.asr.{class_name}'
        # 通过动态导入python模块，创建实例
        # sys.modules 是一个字典，包含了当前Python解释器中已经导入的所有模块
        if lib_name not in sys.modules:
            sys.modules[lib_name] = importlib.import_module(f'{lib_name}')
        # 调用该类的构造函数并传入 *args 和 **kwargs 创建一个实例
        return sys.modules[lib_name].ASRProvider(*args, **kwargs)

    raise ValueError(f"不支持的ASR类型: {class_name}，请检查该配置的type是否设置正确")