"""CPU-only tests of policy mechanics; these are not measured RL results."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from meta_teaching.policy import TeachingPolicy, pair_reward


class TeachingPolicyTests(unittest.TestCase):
    def test_observed_updates_change_greedy_selection(self):
        policy = TeachingPolicy(["plain", "full"], [], exploration=0, seed=14)
        policy.update({}, "plain", 0.4)
        policy.update({}, "full", -0.2)
        self.assertEqual(policy.select({}), "plain")
        for _ in range(20):
            policy.update({}, "full", 0.9)
        self.assertEqual(policy.select({}), "full")
        self.assertGreater(policy.predict({}, "full"), policy.predict({}, "plain"))
        self.assertEqual(policy.total_updates, 22)
        self.assertEqual(policy.arm_counts, {"plain": 1, "full": 21})

    def test_different_contexts_can_choose_different_actions(self):
        policy = TeachingPolicy(["plain", "full"], ["difficulty"], exploration=0)
        # Deterministic test labels exercise context learning, not real training.
        for _ in range(40):
            for difficulty in (0.0, 1.0):
                policy.update({"difficulty": difficulty}, "plain", 1 - difficulty)
                policy.update({"difficulty": difficulty}, "full", difficulty)
        self.assertEqual(policy.select({"difficulty": 0.0}), "plain")
        self.assertEqual(policy.select({"difficulty": 1.0}), "full")

    def test_checkpoint_preserves_predictions_updates_and_rng_sequence(self):
        policy = TeachingPolicy(["plain", "partial", "full"], ["budget"], exploration=0.6)
        for index in range(12):
            context = {"budget": index / 12}
            action = policy.select(context)
            policy.update(context, action, index / 100)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "subdir" / "policy.json"
            policy.save(checkpoint)
            restored = TeachingPolicy.load(checkpoint)
            self.assertEqual(policy.state_dict(), restored.state_dict())
            self.assertFalse(list(checkpoint.parent.glob("*.tmp")))
            for index in range(100):
                context = {"budget": (index % 10) / 10}
                action = policy.select(context)
                self.assertEqual(action, restored.select(context))
                for candidate in policy.action_ids:
                    self.assertEqual(policy.predict(context, candidate), restored.predict(context, candidate))
                policy.update(context, action, 0.1)
                restored.update(context, action, 0.1)
            self.assertEqual(policy.state_dict(), restored.state_dict())

    def test_cold_ties_are_random_not_first_action_biased(self):
        policy = TeachingPolicy(["plain", "partial", "full"], [], exploration=0, seed=9)
        counts = dict.fromkeys(policy.action_ids, 0)
        for _ in range(3000):
            counts[policy.select({})] += 1
        self.assertTrue(all(850 < count < 1150 for count in counts.values()), counts)
        self.assertEqual(policy.arm_counts, {"plain": 0, "partial": 0, "full": 0})

    def test_eligible_actions_and_missing_features(self):
        policy = TeachingPolicy(["plain", "full"], ["budget"], exploration=1)
        self.assertEqual(policy.predict({}, "plain"), policy.predict({"budget": 0}, "plain"))
        for _ in range(20):
            self.assertEqual(policy.select({}, ["full"]), "full")
        for invalid in ([], ["unknown"], ["plain", "plain"], "plain"):
            with self.subTest(eligible=invalid), self.assertRaises(ValueError):
                policy.select({}, invalid)

    def test_nonfinite_invalid_and_unknown_input_rejected_without_mutation(self):
        policy = TeachingPolicy(["plain"], ["budget"])
        original = policy.state_dict()
        for value in (float("nan"), float("inf"), -float("inf"), True, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    policy.update({"budget": value}, "plain", 0.1)
                with self.assertRaises(ValueError):
                    policy.update({}, "plain", value)
                with self.assertRaises(ValueError):
                    policy.select({"budget": value})
                self.assertEqual(original, policy.state_dict())
        with self.assertRaises(ValueError):
            policy.predict({"typo_budget": 1.0}, "plain")
        with self.assertRaises(ValueError):
            policy.predict({}, "unknown")
        policy.update({"budget": 1e300}, "plain", 0.1)
        self.assertTrue(math.isfinite(policy.predict({"budget": -1e300}, "plain")))

    def test_state_is_independent_and_invalid_state_rejected(self):
        policy = TeachingPolicy(["plain"], ["budget"])
        snapshot = policy.state_dict()
        snapshot["arms"]["plain"]["xty"][0] = 100
        self.assertEqual(policy.predict({}, "plain"), 0)
        with self.assertRaises(ValueError):
            TeachingPolicy.from_state_dict(snapshot)
        for field, value in (("schema_version", 99), ("feature_transform", "unknown")):
            state = policy.state_dict()
            state[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                TeachingPolicy.from_state_dict(state)
        malformed = policy.state_dict()
        malformed["arms"]["plain"]["xtx"][0][0] = -1
        with self.assertRaises(ValueError):
            TeachingPolicy.from_state_dict(malformed)
        # JSON encoding/decoding preserves the plain-dict API as well.
        decoded = json.loads(json.dumps(policy.state_dict()))
        self.assertEqual(TeachingPolicy.from_state_dict(decoded).state_dict(), policy.state_dict())

    def test_configuration_validation(self):
        for actions in ([], ["plain", "plain"], [""]):
            with self.assertRaises(ValueError):
                TeachingPolicy(actions)
        for keyword in ({"ridge": 0}, {"exploration": 1.1}, {"ridge": float("nan")}):
            with self.assertRaises(ValueError):
                TeachingPolicy(["plain"], **keyword)

    def test_cost_penalty_converts_gpu_seconds_to_gpu_hours(self):
        self.assertAlmostEqual(pair_reward(0.7, 0.5, 7200, 3600, 0.1), 0.1)
        self.assertAlmostEqual(pair_reward(0.5, 0.5, 3600, 7200, 0.1), 0.1)
        self.assertAlmostEqual(pair_reward(0.7, 0.5, 7200, 3600), 0.2)
        # Eight GPUs for one extra wall-clock hour cost eight GPU hours.
        self.assertAlmostEqual(pair_reward(0.7, 0.5, 8 * 7200, 8 * 3600, 0.01), 0.12)
        for arguments in (
            (float("nan"), 0.5, 0, 0, 0),
            (0.5, 0.5, -1, 0, 0),
            (0.5, 0.5, 0, 0, -0.1),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                pair_reward(*arguments)


if __name__ == "__main__":
    unittest.main()
