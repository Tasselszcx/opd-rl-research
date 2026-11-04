"""Replay G1's vLLM V0 text decoding at an earlier generated-token budget.

This is a paired prefix of one sampled trajectory, not a second sampling run.
The stop policy is the frozen G1 policy: default EOS, three stop strings,
no additional stop-token IDs, min_tokens=0, and excluded stop strings.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
import inspect
from numbers import Integral
from pathlib import Path

STOP_STRINGS = ('</s>', '<|im_end|>', '<|endoftext|>')


class PrefixReplayError(RuntimeError):
    """A prefix could not be reconstructed without guessing its text."""


@lru_cache(maxsize=1)
def _v0_backend():
    # Import only when replay is needed; no model is loaded or sampled. vLLM
    # imports can probe the platform, so standalone CPU tests mask CUDA devices.
    from vllm.transformers_utils import detokenizer_utils
    from vllm.engine.output_processor.stop_checker import StopChecker
    convert = detokenizer_utils.convert_prompt_ids_to_tokens
    step = detokenizer_utils.detokenize_incrementally
    required = {'tokenizer', 'all_input_ids', 'prev_tokens', 'prefix_offset',
                'read_offset', 'skip_special_tokens', 'spaces_between_special_tokens'}
    if not required.issubset(inspect.signature(step).parameters):
        raise PrefixReplayError('Installed vLLM V0 detokenizer interface is incompatible')
    paths = {
        'detokenizer_utils': Path(inspect.getfile(detokenizer_utils)),
        'stop_checker': Path(inspect.getfile(StopChecker)),
    }
    provenance = {
        'vllm_version': importlib.metadata.version('vllm'),
        'v0_source_sha256': {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
        },
    }
    return convert, step, StopChecker.check_stop_strings, provenance


def _integer_ids(values, label, vocab_size):
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise PrefixReplayError(label + ' contains a non-integer token ID')
        value = int(value)
        if not 0 <= value < vocab_size:
            raise PrefixReplayError(label + ' contains an out-of-vocabulary token ID')
        result.append(value)
    return result


def prefix_text_with_metadata(tokenizer, token_ids, final_text, limit=2048,
                              prompt_token_ids=None):
    """Return ``(text, audit_metadata)`` using the frozen G1 stopping policy.

    Finished samples of at most ``limit`` tokens retain the exact returned text.
    Longer samples require actual prompt IDs and replay *only* the first
    ``limit`` generated IDs with the installed V0 incremental decoder. An
    unfinished UTF-8 character may remain buffered, and a stop-string fragment
    crossing the cutoff is not removed using later IDs.

    Errors are explicit. Callers must retain the raw generated record before
    aborting/failing an audit; this function never fabricates or drops a sample.
    """
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError('limit must be a positive integer')
    if not isinstance(final_text, str):
        raise TypeError('final_text must be the original string returned by vLLM')
    count = len(token_ids)
    metadata = {
        'method': 'original_finished_text' if count <= limit else 'vllm_v0_incremental_replay',
        'limit': int(limit), 'sample_tokens': count,
        'replayed_tokens': 0, 'uses_future_tokens_for_decoding': False,
    }
    if count <= limit:
        return final_text, metadata
    if prompt_token_ids is None:
        raise PrefixReplayError('Long-sample replay requires the actual prompt_token_ids')
    try:
        convert, step, check_stop_strings, provenance = _v0_backend()
        metadata.update(provenance)
        all_ids = _integer_ids(prompt_token_ids, 'Prompt', len(tokenizer))
        prefix_ids = _integer_ids(token_ids[:limit], 'Generated prefix', len(tokenizer))
        previous_tokens, prefix_offset, read_offset = convert(
            tokenizer=tokenizer, prompt_ids=all_ids, skip_special_tokens=True)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if not isinstance(eos_id, Integral):
            raise PrefixReplayError('Frozen G1 replay requires a single known tokenizer EOS ID')
        text = ''
        empty_decode_steps = 0
        for position, token_id in enumerate(prefix_ids, 1):
            all_ids.append(token_id)
            new_tokens, new_text, prefix_offset, read_offset = step(
                tokenizer=tokenizer, all_input_ids=all_ids,
                prev_tokens=previous_tokens, prefix_offset=prefix_offset,
                read_offset=read_offset, skip_special_tokens=True,
                spaces_between_special_tokens=True)
            previous_tokens.extend(new_tokens)
            text += new_text
            empty_decode_steps += not bool(new_text)
            # G1 has no explicit stop_token_ids. Its stop strings must not
            # silently become new token stops, including special spellings.
            stop_string = check_stop_strings(
                output_text=text, new_char_count=len(new_text),
                stop=list(STOP_STRINGS), include_in_output=False)
            if token_id == eos_id or stop_string is not None:
                # A sample genuinely longer than the cutoff cannot have
                # already stopped inside its prefix under the same policy.
                reason = 'EOS' if token_id == eos_id else repr(stop_string[0])
                raise PrefixReplayError(
                    f'Long sample contains an earlier {reason} stop at generated token '
                    f'{position}; original sampling policy/record must be checked')
        metadata.update({
            'replayed_tokens': len(prefix_ids),
            'empty_decode_steps': empty_decode_steps,
            'decoder_has_buffered_suffix': read_offset < len(previous_tokens),
            'contains_replacement_character': '\ufffd' in text,
            # False is permitted when a stop string begins before the cutoff
            # but is completed and removed only later in the full sample.
            'is_prefix_of_final_visible_text': final_text.startswith(text),
        })
        return text, metadata
    except PrefixReplayError:
        raise
    except Exception as exc:
        raise PrefixReplayError(
            'vLLM V0 prefix replay failed; retain raw IDs/text and do not score '
            'a guessed prefix: ' + repr(exc)) from exc


def prefix_text(tokenizer, token_ids, final_text, limit=2048, prompt_token_ids=None):
    """String-only interface; use prefix_text_with_metadata for audit provenance."""
    return prefix_text_with_metadata(
        tokenizer, token_ids, final_text, limit, prompt_token_ids)[0]
