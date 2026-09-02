#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/19 17:26
@Author  : alexanderwu
@File    : llm_provider_registry.py
"""
from maas.configs.llm_config import LLMConfig, LLMType
from maas.provider.base_llm import BaseLLM


class LLMProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: LLMType):
        """get provider instance according to the enum"""
        return self.providers[enum]


def register_provider(keys):
    """register provider to registry"""

    def decorator(cls):
        if isinstance(keys, list):
            for key in keys:
                LLM_REGISTRY.register(key, cls)
        else:
            LLM_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_llm_instance(config: LLMConfig) -> BaseLLM:
    """get the default llm provider"""

    # --- shared-layer shim (agent_wf_v2) --- namespace override v1
    # The proxy reads its accounting namespace off the URL path, and only the
    # launching process knows which dataset and phase this job is. See
    # shims/namespace_patch.py.
    import os as _shim_ns_os

    _shim_ns_url = _shim_ns_os.getenv("SHIM_BASE_URL")
    if _shim_ns_url:
        try:
            config.base_url = _shim_ns_url
        except Exception:  # noqa: BLE001 - a frozen config must not stop the run
            pass
    llm = LLM_REGISTRY.get_provider(config.api_type)(config)
    if llm.use_system_prompt and not config.use_system_prompt:
        # for models like o1-series, default openai provider.use_system_prompt is True, but it should be False for o1-*
        llm.use_system_prompt = config.use_system_prompt
    return llm


# Registry instance
LLM_REGISTRY = LLMProviderRegistry()
