from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .types import ModelRun, Tool, ToolCallLimitExceeded


class CallableBackend:
    """Adapter for any inference library that can implement the benchmark loop."""

    def __init__(
        self,
        name: str,
        function: Callable[..., ModelRun | str | Awaitable[ModelRun | str]],
        *,
        temperature: float | None = 1.0,
        max_output_tokens: int = 8192,
    ) -> None:
        if (
            not isinstance(name, str)
            or not 1 <= len(name) <= 200
            or any(character in name for character in "\x00\r\n")
        ):
            raise ValueError("Invalid model name")
        if not callable(function):
            raise TypeError("function must be callable")
        _validate_model_options(name, temperature, max_output_tokens, maximum=2)
        self.name = name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._function = function
        try:
            parameters = inspect.signature(function).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        self._generation_arguments = {
            name
            for name in ("temperature", "max_output_tokens")
            if accepts_kwargs or name in parameters
        }

    async def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[Tool],
        max_tool_calls: int,
    ) -> ModelRun:
        arguments = {
            "system": system,
            "prompt": prompt,
            "tools": tools,
            "max_tool_calls": max_tool_calls,
        }
        if "temperature" in self._generation_arguments:
            arguments["temperature"] = self.temperature
        if "max_output_tokens" in self._generation_arguments:
            arguments["max_output_tokens"] = self.max_output_tokens
        if inspect.iscoroutinefunction(self._function):
            result = self._function(**arguments)
        else:
            result = await asyncio.to_thread(self._function, **arguments)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, str):
            return ModelRun(answer=result)
        if not isinstance(result, ModelRun):
            raise TypeError("Callable backend must return str or ModelRun")
        return result


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 8192,
    ) -> None:
        _validate_model_options(model, temperature, max_output_tokens, maximum=2)
        if temperature is not None and re.match(
            r"^gpt-5(?:$|-(?:mini|nano)(?:-|$)|-20\d{2}|\.1(?:-|$))", model
        ):
            raise ValueError(f"{model} does not accept temperature")
        self.name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if api_key is not None:
            _validate_api_key(api_key)
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as error:
                raise ImportError(
                    "Install CorporateBench with the 'openai' extra"
                ) from error
            options: dict[str, Any] = {"timeout": 60.0, "max_retries": 2}
            if api_key is not None:
                options["api_key"] = api_key
            client = AsyncOpenAI(**options)
        self._client = client

    async def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[Tool],
        max_tool_calls: int,
    ) -> ModelRun:
        definitions = [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
                "strict": False,
            }
            for tool in tools
        ]
        tool_map = {tool.name: tool for tool in tools}
        input_items: list[Any] = [{"role": "user", "content": prompt}]
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0

        while True:
            request: dict[str, Any] = {
                "model": self.name,
                "instructions": system,
                "input": input_items,
                "tools": definitions,
                "parallel_tool_calls": False,
                "max_output_tokens": self.max_output_tokens,
                "store": False,
                "include": ["reasoning.encrypted_content"],
            }
            if self.temperature is not None:
                request["temperature"] = self.temperature
            response = await self._client.responses.create(**request)
            usage = getattr(response, "usage", None)
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            output = list(getattr(response, "output", []))
            calls = [
                item
                for item in output
                if getattr(item, "type", None) == "function_call"
            ]
            if not calls:
                return ModelRun(
                    answer=str(getattr(response, "output_text", "") or ""),
                    tool_calls=tool_call_count,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            if tool_call_count + len(calls) > max_tool_calls:
                raise ToolCallLimitExceeded(
                    f"Tool call limit of {max_tool_calls} exceeded"
                )

            input_items.extend(_model_dump(item) for item in output)
            for call in calls:
                arguments = _json_object(getattr(call, "arguments", "{}"))
                result = await _call_tool(tool_map, str(call.name), arguments)
                tool_call_count += 1
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(call.call_id),
                        "output": result,
                    }
                )

    async def run_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> ModelRun:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        request: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "knowledge_extraction",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        response = await self._client.chat.completions.create(**request)
        choices = list(getattr(response, "choices", []))
        if not choices:
            raise ValueError("OpenAI returned no structured output")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            raise ValueError("OpenAI returned invalid structured output")
        usage = getattr(response, "usage", None)
        return ModelRun(
            answer=content,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class AnthropicBackend:
    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 8192,
    ) -> None:
        _validate_model_options(model, temperature, max_tokens, maximum=1)
        version = re.search(r"-(\d+)-(\d+)(?:-|$)", model)
        if version and tuple(map(int, version.groups())) >= (4, 6) and temperature != 1:
            raise ValueError("Anthropic 4.6+ models require temperature=1")
        self.name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if api_key is not None:
            _validate_api_key(api_key)
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as error:
                raise ImportError(
                    "Install CorporateBench with the 'anthropic' extra"
                ) from error
            options: dict[str, Any] = {"timeout": 60.0, "max_retries": 2}
            if api_key is not None:
                options["api_key"] = api_key
            client = AsyncAnthropic(**options)
        self._client = client

    async def run(
        self,
        *,
        system: str,
        prompt: str,
        tools: Sequence[Tool],
        max_tool_calls: int,
    ) -> ModelRun:
        definitions = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        tool_map = {tool.name: tool for tool in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0

        while True:
            request: dict[str, Any] = {
                "model": self.name,
                "system": system,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            request["tools"] = definitions
            response = await self._client.messages.create(**request)
            usage = getattr(response, "usage", None)
            input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            content = list(getattr(response, "content", []))
            calls = [
                block for block in content if getattr(block, "type", None) == "tool_use"
            ]
            if not calls:
                answer = "".join(
                    str(getattr(block, "text", ""))
                    for block in content
                    if getattr(block, "type", None) == "text"
                )
                return ModelRun(
                    answer=answer,
                    tool_calls=tool_call_count,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            if tool_call_count + len(calls) > max_tool_calls:
                raise ToolCallLimitExceeded(
                    f"Tool call limit of {max_tool_calls} exceeded"
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": [_model_dump(block) for block in content],
                }
            )
            tool_results = []
            for call in calls:
                arguments = getattr(call, "input", {})
                if not isinstance(arguments, dict):
                    arguments = {}
                result = await _call_tool(tool_map, str(call.name), arguments)
                tool_call_count += 1
                is_error = result.startswith("Tool error:")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(call.id),
                        "content": result,
                        "is_error": is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_results})


async def _call_tool(
    tools: dict[str, Tool], name: str, arguments: dict[str, Any]
) -> str:
    tool = tools.get(name)
    if tool is None:
        return "Tool error: unknown tool."
    try:
        return await tool.call(arguments)
    except ToolCallLimitExceeded:
        raise
    except Exception:
        return "Tool error: invalid request."


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    raise TypeError("Unsupported SDK response object")


def _validate_model_options(
    model: str,
    temperature: float | None,
    max_output_tokens: int,
    *,
    maximum: float,
) -> None:
    if (
        not isinstance(model, str)
        or not 1 <= len(model) <= 200
        or any(character in model for character in "\x00\r\n")
    ):
        raise ValueError("Invalid model name")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not 0 <= temperature <= maximum
    ):
        raise ValueError(f"Temperature must be between 0 and {maximum:g}")
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or not 1 <= max_output_tokens <= 100_000
    ):
        raise ValueError("Invalid output token limit")


def _validate_api_key(api_key: object) -> None:
    if (
        not isinstance(api_key, str)
        or not 1 <= len(api_key) <= 10_000
        or any(character in api_key for character in "\x00\r\n")
    ):
        raise ValueError("Invalid API key")
