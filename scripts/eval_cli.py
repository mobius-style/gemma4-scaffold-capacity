#!/usr/bin/env python3
"""swegemma eval with scorer-like context management (HARNESS_README §7.2) and a tolerant tool-call parser.

D1: the stock CLI leaves events_compaction_config / context_cache_config unset, so contexts overflow the 32,768
    window on long trajectories (observed: prompt_tokens 32,742 -> ContextWindowExceededError). The Kaggle scorer
    (scripts/inference.py) sets compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5
    and ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10). We set the same here.
D2: ADK's LiteLlm converter does json.loads(tool_call.function.arguments) unguarded; a malformed/truncated
    argument string (llama-server tool parser on cut-off output) raised JSONDecodeError and aborted the task
    ("Sandbox execution error: Unterminated string ..."). On the scorer the model receives a structured tool error
    and continues. We wrap the converter so malformed arguments become {"__malformed_arguments__": raw}, which the
    harness ToolErrorPlugin turns into an error dict returned to the model (same recoverable path).
Usage: identical to `swegemma eval ...`.
"""
import json, sys
from google.adk.models import lite_llm as _ll
from google.genai import types as _types

_orig = _ll._message_to_generate_content_response


def _safe_message_to_generate_content_response(message, *, is_partial=False, model_version=None, thought_parts=None):
    tool_calls = getattr(message, 'tool_calls', None) or []
    for tc in tool_calls:
        fn = getattr(tc, 'function', None)
        if fn is None:
            continue
        raw = fn.arguments
        if raw in (None, ''):
            continue
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            fn.arguments = json.dumps({'__malformed_arguments__': str(raw)[:4000], '__parse_error__': str(e)[:200]})
            sys.stderr.write(f'[eval_cli] malformed tool-call arguments for {getattr(fn, "name", "?")}: {str(e)[:120]}\n')
    return _orig(message, is_partial=is_partial, model_version=model_version, thought_parts=thought_parts)


_ll._message_to_generate_content_response = _safe_message_to_generate_content_response

# D5: llama-server answers HTTP 500 "The model produced output that does not match the expected peg-gemma4 format"
# when the model's text cannot be parsed by llama.cpp's Gemma 4 output grammar (observed on 12B: 25/104 tasks in
# one cell, 0/208 on 26B, 0/60 on E4B pilots). litellm retries, then raises InternalServerError and the task dies.
# On the scorer (vLLM) an unparseable output is returned as plain text and the harness nudges the model. We
# reproduce that path: the failed turn becomes a text-only model response, counted in stderr for reporting.
from google.adk.models.llm_response import LlmResponse as _LlmResponse

_orig_gen = _ll.LiteLlm.generate_content_async
_PEG_MSG = ('[harness] The previous model output could not be parsed as a tool call by the inference server '
            '(format error). Re-emit a single, well-formed tool call, or call submit_patch if the work is complete.')


async def _tolerant_generate_content_async(self, llm_request, stream=False):
    try:
        async for r in _orig_gen(self, llm_request, stream=stream):
            yield r
    except Exception as e:  # litellm.InternalServerError / openai.InternalServerError wrappers
        if 'peg-gemma4' in str(e) or 'does not match the expected' in str(e):
            sys.stderr.write('[eval_cli] peg-gemma4 parse failure converted to text turn\n')
            yield _LlmResponse(content=_types.Content(role='model', parts=[_types.Part.from_text(text=_PEG_MSG)]))
        else:
            raise


_ll.LiteLlm.generate_content_async = _tolerant_generate_content_async

from swegemma import cli as _cli  # noqa: E402
from swegemma.config import EvalConfig as _EvalConfig  # noqa: E402

try:
    from google.adk.apps._configs import EventsCompactionConfig
except Exception:  # pragma: no cover
    from google.adk.apps.app import EventsCompactionConfig
from google.adk.agents.context_cache_config import ContextCacheConfig

COMPACTION = dict(compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5)
CACHE = dict(min_tokens=2048, ttl_seconds=1800, cache_intervals=10)


class _PatchedEvalConfig(_EvalConfig):
    def __init__(self, *a, **kw):
        kw.setdefault('events_compaction_config', EventsCompactionConfig(**COMPACTION))
        kw.setdefault('context_cache_config', ContextCacheConfig(**CACHE))
        super().__init__(*a, **kw)


_cli.EvalConfig = _PatchedEvalConfig

if __name__ == '__main__':
    sys.stderr.write(f'[eval_cli] compaction={COMPACTION} cache={CACHE} tolerant-tool-args=on\n')
    sys.exit(_cli.main())
