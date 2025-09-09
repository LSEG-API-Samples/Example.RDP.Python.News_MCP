"""
LLM Configuration Module for MCP News Server

This module provides guidance for integrating various LLM providers with the MCP news server.
Users should implement their preferred LLM provider based on their requirements and access.

For a complete list of supported LLM providers, see:
https://python.langchain.com/docs/integrations/chat/
"""

from typing import Any


def get_default_chat_llm() -> Any:
    """
    Get the default chat LLM instance.
    
    This function should be implemented by the user with their preferred LLM provider.
    See the module docstring for examples and available options.
    
    Raises:
        NotImplementedError: This is a placeholder that must be implemented by the user.
        
    Example implementation:
        from langchain_openai import ChatOpenAI
        def get_default_chat_llm():
            return ChatOpenAI(model="gpt-4", temperature=0.7)
    """
    raise NotImplementedError(
        "Please implement get_default_chat_llm() with your preferred LLM provider. "
        "See the module docstring for examples and https://python.langchain.com/docs/integrations/chat/ "
        "for available providers."
    )


def get_default_judge_llm() -> Any:
    """
    Get the default judge LLM instance for evaluations.
    
    This function should be implemented by the user with their preferred LLM provider.
    For evaluation tasks, consider using models optimized for reasoning and analysis.
    
    Raises:
        NotImplementedError: This is a placeholder that must be implemented by the user.
        
    Example implementation:
        from langchain_anthropic import ChatAnthropic
        def get_default_judge_llm():
            return ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.1)
    """
    raise NotImplementedError(
        "Please implement get_default_judge_llm() with your preferred LLM provider. "
        "See the module docstring for examples and https://python.langchain.com/docs/integrations/chat/ "
        "for available providers. Consider using models with strong reasoning capabilities for evaluation tasks."
    )
