"""
OpenAI-compatible client backed by Tinker sampling.

Implements OpenAI client semantics for:
- chat.completions.create(...)
- completions.create(...)

Returns OpenAI types (ChatCompletion / Completion) constructed from sampled tokens.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, overload

import tinker
from openai import AsyncOpenAI
from openai._streaming import AsyncStream
from openai.resources.chat import AsyncChat as OpenAIAsyncChat
from openai.resources.chat.completions import AsyncCompletions as OpenAIAsyncChatCompletions
from openai.resources.completions import AsyncCompletions as OpenAIAsyncCompletions
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer

_THINK_BLOCK_RE = re.compile(r"(<think>)(.*?)(</think>)", flags=re.DOTALL)
_THINK_TAG_RE = re.compile(r"<think\b[^>]*>|</think\s*>", flags=re.IGNORECASE)


def _parse_max_thinking_tokens(value: Any) -> int:
    if value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _truncate_thinking_blocks_text(
    text: str,
    tokenizer: Tokenizer,
    max_thinking_tokens: int,
) -> tuple[str, bool]:
    if max_thinking_tokens <= 0:
        return text, False

    changed = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal changed
        opening, body, closing = match.groups()
        body_ids = tokenizer.encode(body, add_special_tokens=False)
        if len(body_ids) <= max_thinking_tokens:
            return match.group(0)
        truncated_body = tokenizer.decode(body_ids[:max_thinking_tokens], skip_special_tokens=False)
        changed = True
        return f"{opening}{truncated_body}{closing}"

    new_text = _THINK_BLOCK_RE.sub(_replace, text)
    return new_text, changed


def _truncate_and_close_thinking_blocks_text(
    text: str,
    tokenizer: Tokenizer,
    max_thinking_tokens: int,
) -> tuple[str, bool]:
    """
    Bound thinking tokens and force-close an unterminated <think> block.

    If a response ends inside <think>...</think> without a closing tag, truncate the
    remaining thinking content to max_thinking_tokens and append </think>.
    """
    bounded_text, changed = _truncate_thinking_blocks_text(
        text=text,
        tokenizer=tokenizer,
        max_thinking_tokens=max_thinking_tokens,
    )
    if max_thinking_tokens <= 0:
        return bounded_text, changed

    depth = 0
    unmatched_open_body_start: int | None = None
    for match in _THINK_TAG_RE.finditer(bounded_text):
        token = match.group(0).strip().lower()
        if token.startswith("</think"):
            if depth > 0:
                depth -= 1
            if depth == 0:
                unmatched_open_body_start = None
            continue

        if depth == 0:
            unmatched_open_body_start = match.end()
        depth += 1

    if depth <= 0 or unmatched_open_body_start is None:
        return bounded_text, changed

    trailing_body = bounded_text[unmatched_open_body_start:]
    trailing_ids = tokenizer.encode(trailing_body, add_special_tokens=False)
    if len(trailing_ids) > max_thinking_tokens:
        trailing_body = tokenizer.decode(
            trailing_ids[:max_thinking_tokens],
            skip_special_tokens=False,
        )
    closed_text = f"{bounded_text[:unmatched_open_body_start]}{trailing_body}</think>"
    return closed_text, True


def _answer_text_after_think_blocks(response_text: str) -> str:
    if not response_text:
        return ""
    last_close: re.Match[str] | None = None
    for match in re.finditer(r"</think\s*>", response_text, flags=re.IGNORECASE):
        last_close = match
    if last_close is not None:
        return response_text[last_close.end() :].strip()
    if re.search(r"<think\b", response_text, flags=re.IGNORECASE) is not None:
        return ""
    return response_text.strip()


class TinkerAsyncOpenAIClient(AsyncOpenAI):
    """
    OpenAI-compatible async client that routes calls to a Tinker SamplingClient.
    """

    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__(api_key="tinker", base_url="http://localhost")
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer

    def set_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        self.sampling_client = sampling_client

    @property
    def chat(self) -> OpenAIAsyncChat:
        return TinkerAsyncChat(self)

    @property
    def completions(self) -> OpenAIAsyncCompletions:
        return TinkerCompletions(self)


class TinkerChatCompletions(OpenAIAsyncChatCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Any]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(self, *args: Any, stream: bool, **kwargs: Any) -> ChatCompletion: ...

    async def create(self, *args: Any, **kwargs: Any) -> ChatCompletion | AsyncStream[Any]:
        model = kwargs.get("model", "tinker")
        messages = kwargs.get("messages", [])
        if kwargs.get("tools"):
            raise NotImplementedError("Tool calling is not yet supported by this model's renderer.")
        if kwargs.get("stream", False):
            raise ValueError("stream=True not supported by TinkerAsyncOpenAIClient")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "tools")}
        max_thinking_tokens = _parse_max_thinking_tokens(sampling_args.get("max_thinking_tokens"))
        force_answer_max_tokens = _parse_max_thinking_tokens(
            sampling_args.get("force_answer_max_tokens", 32 if max_thinking_tokens > 0 else 0)
        )
        force_answer_temperature_raw = sampling_args.get("force_answer_temperature", 0.2)
        try:
            force_answer_temperature = float(force_answer_temperature_raw)
        except (TypeError, ValueError):
            force_answer_temperature = 0.2

        stop = sampling_args.get("stop", self._parent.renderer.get_stop_sequences())
        max_tokens = sampling_args.get("max_tokens") or sampling_args.get("max_completion_tokens")
        max_tokens_int = int(max_tokens or 128)
        reserved_answer_tokens = 0
        initial_max_tokens = max_tokens_int
        if max_thinking_tokens > 0 and force_answer_max_tokens > 0 and max_tokens_int > 1:
            reserved_answer_tokens = min(force_answer_max_tokens, max_tokens_int - 1)
            initial_max_tokens = max_tokens_int - reserved_answer_tokens

        model_input = self._parent.renderer.build_generation_prompt(messages)
        prompt_token_ids: List[int] = model_input.to_ints()

        async def _recompute_completion_logprobs(
            completion_ids: list[int],
        ) -> list[float] | None:
            full_input = tinker.ModelInput.from_ints(prompt_token_ids + completion_ids)
            full_logprobs = await self._parent.sampling_client.compute_logprobs_async(full_input)
            completion_logprobs = full_logprobs[
                len(prompt_token_ids) : len(prompt_token_ids) + len(completion_ids)
            ]
            if len(completion_logprobs) != len(completion_ids):
                return None
            if any(lp is None for lp in completion_logprobs):
                return None
            return [float(lp) for lp in completion_logprobs if lp is not None]

        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=initial_max_tokens,
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
                stop=stop,
            ),
        )
        seq = sample.sequences[0]
        completion_token_ids: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)

        if max_thinking_tokens > 0 and completion_token_ids:
            completion_text = self._parent.tokenizer.decode(
                completion_token_ids, skip_special_tokens=False
            )
            bounded_text, was_truncated = _truncate_and_close_thinking_blocks_text(
                completion_text,
                tokenizer=self._parent.tokenizer,
                max_thinking_tokens=max_thinking_tokens,
            )
            if was_truncated:
                bounded_token_ids = self._parent.tokenizer.encode(
                    bounded_text, add_special_tokens=False
                )
                try:
                    bounded_logprobs = await _recompute_completion_logprobs(bounded_token_ids)
                    if bounded_logprobs is not None:
                        completion_token_ids = bounded_token_ids
                        logprobs = bounded_logprobs
                except Exception:
                    # If recomputing logprobs fails, keep original sampled output untouched.
                    pass

        if max_thinking_tokens > 0 and reserved_answer_tokens > 0 and completion_token_ids:
            completion_text = self._parent.tokenizer.decode(
                completion_token_ids, skip_special_tokens=False
            )
            post_think_answer = _answer_text_after_think_blocks(completion_text)
            has_think_tag = _THINK_TAG_RE.search(completion_text) is not None
            if has_think_tag and not post_think_answer:
                continuation_input = tinker.ModelInput.from_ints(prompt_token_ids + completion_token_ids)
                continuation_sample = await self._parent.sampling_client.sample_async(
                    prompt=continuation_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        temperature=max(0.0, force_answer_temperature),
                        max_tokens=reserved_answer_tokens,
                        top_p=float(sampling_args.get("top_p", 1.0)),
                        top_k=int(sampling_args.get("top_k", -1)),
                        stop=stop,
                    ),
                )
                continuation_tokens = continuation_sample.sequences[0].tokens
                if continuation_tokens:
                    completion_token_ids = list(completion_token_ids) + list(continuation_tokens)
                    try:
                        recomputed = await _recompute_completion_logprobs(completion_token_ids)
                        if recomputed is not None:
                            logprobs = recomputed
                    except Exception:
                        pass

        assistant_message, parse_success = self._parent.renderer.parse_response(
            completion_token_ids
        )
        finish_reason = "stop" if parse_success else "length"

        # Convert list content to string for OpenAI compatibility
        openai_content = renderers.format_content_as_string(assistant_message["content"])

        # Build OpenAI-compatible message
        openai_message: Dict[str, Any] = {
            "role": "assistant",
            "content": openai_content,
        }
        # Include tool_calls if present
        if "tool_calls" in assistant_message:
            openai_message["tool_calls"] = [
                {
                    "id": tc.id or f"call_{i}",
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for i, tc in enumerate(assistant_message["tool_calls"])
            ]

        response_dict: Dict[str, Any] = {
            "id": "tinker-chatcmpl",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": openai_message,
                    "finish_reason": finish_reason,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{tid}", "logprob": lp, "top_logprobs": []}
                            for tid, lp in zip(completion_token_ids, logprobs)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_token_ids),
                "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
            },
        }
        response = ChatCompletion.model_validate(response_dict)

        setattr(response, "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", completion_token_ids)

        return response


class TinkerCompletions(OpenAIAsyncCompletions):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @overload
    async def create(
        self, *args: Any, stream: Literal[True], **kwargs: Any
    ) -> AsyncStream[Completion]: ...

    @overload
    async def create(
        self, *args: Any, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    async def create(
        self, *args: Any, stream: bool, **kwargs: Any
    ) -> Completion | AsyncStream[Completion]: ...

    async def create(self, *args: Any, **kwargs: Any) -> Completion | AsyncStream[Completion]:
        stream = bool(kwargs.get("stream", False))
        model = kwargs.get("model", "tinker")
        prompt = kwargs.get("prompt", "")
        sampling_args = {k: v for k, v in kwargs.items() if k not in ("model", "prompt")}

        prompt_token_ids: List[int] = self._parent.tokenizer.encode(prompt, add_special_tokens=True)
        model_input = tinker.ModelInput.from_ints(prompt_token_ids)

        sample = await self._parent.sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=float(sampling_args.get("temperature", 1.0)),
                max_tokens=int(sampling_args.get("max_tokens", 128)),
                top_p=float(sampling_args.get("top_p", 1.0)),
                top_k=int(sampling_args.get("top_k", -1)),
            ),
        )
        seq = sample.sequences[0]
        completion_token_ids: List[int] = seq.tokens
        logprobs: List[float] = seq.logprobs or [0.0] * len(completion_token_ids)

        text = self._parent.tokenizer.decode(completion_token_ids)
        tokens_str = [f"token_id:{tid}" for tid in completion_token_ids]
        response_dict: Dict[str, Any] = {
            "id": "tinker-cmpl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": {
                        "tokens": tokens_str,
                        "token_logprobs": logprobs,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_token_ids),
                "completion_tokens": len(completion_token_ids),
                "total_tokens": len(prompt_token_ids) + len(completion_token_ids),
            },
        }
        response = Completion.model_validate(response_dict)

        setattr(response.choices[0], "prompt_token_ids", prompt_token_ids)
        setattr(response.choices[0], "token_ids", completion_token_ids)

        if stream:
            return TinkerAsyncCompletionStream(response)
        return response


class TinkerAsyncChat(OpenAIAsyncChat):
    def __init__(self, parent: TinkerAsyncOpenAIClient) -> None:
        self._parent = parent

    @property
    def completions(self) -> OpenAIAsyncChatCompletions:
        return TinkerChatCompletions(self._parent)


class TinkerAsyncCompletionStream(AsyncStream[Completion]):
    def __init__(self, final: Completion) -> None:
        self._final = final

    def __aiter__(self):
        self._done = True
        return self

    async def __anext__(self) -> Completion:
        raise StopAsyncIteration

    def __await__(self):
        async def _await_final():
            return self._final

        return _await_final().__await__()

    async def get_final_response(self) -> Completion:
        return self._final
