"""CPU tests against the installed vLLM V0 incremental decoder, no model load."""
import unittest
from unittest.mock import patch

from length_audit.prefix_scoring import (
    PrefixReplayError, prefix_text, prefix_text_with_metadata,
)


class ByteTokenizer:
    """Byte-fallback tokenizer exposing the real decoder's HF interface."""
    eos_token_id = 256
    is_fast = True

    def __len__(self):
        return 257

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [chr(i) if i < 256 else '<eos>' for i in ids
                if not (skip_special_tokens and i == self.eos_token_id)]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens).encode('latin1').decode('utf-8', errors='replace')

    def get_added_vocab(self):
        return {}


class ContextTokenizer(ByteTokenizer):
    def convert_tokens_to_string(self, tokens):
        # Standalone decoding would incorrectly remove a continuation's space.
        return super().convert_tokens_to_string(tokens).lstrip(' ')


def ids(text):
    return list(text.encode('utf-8'))


class PrefixScoringTest(unittest.TestCase):
    def test_early_stop_keeps_original_visible_text(self):
        self.assertEqual(prefix_text(ByteTokenizer(), [65, 256], 'A', 3), 'A')

    def test_exact_limit_keeps_original_detokenization(self):
        self.assertEqual(prefix_text(ByteTokenizer(), [65, 256], 'A', 2), 'A')

    def test_answer_after_limit_is_not_leaked(self):
        value = 'reasoning<answer>yes</answer>'
        self.assertEqual(prefix_text(ByteTokenizer(), ids(value), value, 9, ids('Q ')), 'reasoning')

    def test_partial_answer_tag_is_not_repaired(self):
        value = '<answer>yes</answer>'
        limit = len('<answer>yes</ans')
        result = prefix_text(ByteTokenizer(), ids(value), value, limit, ids('Q '))
        self.assertEqual(result, '<answer>yes</ans')
        self.assertNotIn('</answer>', result)

    def test_incomplete_utf8_at_cutoff_stays_buffered(self):
        value = 'A\u4e2dZ'
        text, meta = prefix_text_with_metadata(ByteTokenizer(), ids(value), value, 2, ids('Q '))
        self.assertEqual(text, 'A')
        self.assertTrue(meta['decoder_has_buffered_suffix'])
        self.assertNotIn('\ufffd', text)

    def test_complete_utf8_at_cutoff_is_emitted(self):
        value = 'A\u4e2dZ'
        self.assertEqual(prefix_text(ByteTokenizer(), ids(value), value, 4, ids('Q ')), 'A\u4e2d')

    def test_prompt_context_affects_detokenization(self):
        value = ' xZ'
        self.assertEqual(prefix_text(ContextTokenizer(), ids(value), value, 2, ids('Q')), ' x')

    def test_stop_string_crossing_cutoff_is_not_removed_early(self):
        text, meta = prefix_text_with_metadata(ByteTokenizer(), ids('abc</s>'), 'abc', 5, ids('Q '))
        self.assertEqual(text, 'abc</')
        self.assertFalse(meta['is_prefix_of_final_visible_text'])

    def test_complete_stop_inside_long_sample_is_inconsistent(self):
        with self.assertRaisesRegex(PrefixReplayError, 'earlier.*stop'):
            prefix_text(ByteTokenizer(), ids('a</s>tail'), 'a', 6, ids('Q '))

    def test_eos_inside_long_sample_is_inconsistent(self):
        with self.assertRaisesRegex(PrefixReplayError, 'earlier EOS'):
            prefix_text(ByteTokenizer(), [65, 256, 66, 67], 'A', 3, ids('Q '))

    def test_future_tokens_do_not_change_prefix(self):
        a = prefix_text(ByteTokenizer(), ids('ABCtail'), 'ABCtail', 3, ids('Q '))
        b = prefix_text(ByteTokenizer(), ids('ABC<answer>no</answer>'), 'different final text', 3, ids('Q '))
        self.assertEqual(a, b)
        self.assertEqual(a, 'ABC')

    def test_long_sample_requires_actual_prompt_ids(self):
        with self.assertRaisesRegex(PrefixReplayError, 'actual prompt_token_ids'):
            prefix_text(ByteTokenizer(), ids('abc'), 'abc', 2)

    def test_backend_failure_is_not_silently_scored(self):
        with patch('length_audit.prefix_scoring._v0_backend', side_effect=ImportError('missing')):
            with self.assertRaisesRegex(PrefixReplayError, 'do not score a guessed prefix'):
                prefix_text(ByteTokenizer(), ids('abc'), 'abc', 2, [])

    def test_invalid_token_is_not_silently_decoded(self):
        with self.assertRaisesRegex(PrefixReplayError, 'out-of-vocabulary'):
            prefix_text(ByteTokenizer(), [65, 999, 66], 'AB', 2, [])

    def test_invalid_limit_rejected(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                prefix_text(ByteTokenizer(), ids('abc'), 'abc', limit, [])


if __name__ == '__main__':
    unittest.main()
