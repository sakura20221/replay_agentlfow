#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/5 22:59
@Author  : alexanderwu
@File    : __init__.py
"""

# --- shared-layer shim: tolerant provider imports ---
import importlib as _importlib

_PROVIDERS = [('google_gemini_api', 'GeminiLLM'), ('ollama_api', 'OllamaLLM'), ('openai_api', 'OpenAILLM'), ('zhipuai_api', 'ZhiPuAILLM'), ('azure_openai_api', 'AzureOpenAILLM'), ('metagpt_api', 'MetaGPTLLM'), ('human_provider', 'HumanProvider'), ('spark_api', 'SparkLLM'), ('qianfan_api', 'QianFanLLM'), ('dashscope_api', 'DashScopeLLM'), ('anthropic_api', 'AnthropicLLM'), ('bedrock_api', 'BedrockLLM'), ('ark_api', 'ArkLLM')]

for _module_name, _class_name in _PROVIDERS:
    try:
        _module = _importlib.import_module(f"maas.provider.{_module_name}")
        globals()[_class_name] = getattr(_module, _class_name)
    except Exception:
        # Backend not installed or not importable on this Python; the
        # bake-off does not use it, so skip rather than fail the package.
        pass

__all__ = [_class_name for _, _class_name in _PROVIDERS if _class_name in globals()]
