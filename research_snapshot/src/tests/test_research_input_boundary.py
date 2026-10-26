"""R01 artificial requests only; no positive real-world construction."""
import builtins
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock, patch
from research.input_boundary import ResearchInputPolicy, build_environment

FIXTURE = dict(year=2033, family="w1_access_fixture", group="G00",
               split="train", fixture=True)


class ResearchInputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.policy = ResearchInputPolicy([FIXTURE.copy()])

    def deny(self, request):
        factory = Mock(side_effect=AssertionError("factory must not be called"))
        with self.assertRaises((PermissionError, ValueError)):
            self.policy.create_store(request, factory=factory)
        factory.assert_not_called()

    def test_approved_artificial_request_calls_fake_factory_once(self):
        fake = Mock(return_value="artificial-result")
        self.assertEqual(self.policy.create_store(FIXTURE.copy(), factory=fake),
                         "artificial-result")
        fake.assert_called_once_with(**FIXTURE)

    def test_test_and_confirmation_splits_rejected_before_factory(self):
        for split in ("test", "confirmation", "sealed", "holdout"):
            with self.subTest(split=split):
                self.deny(dict(FIXTURE, split=split))

    def test_train_alias_cannot_generate_unapproved_world(self):
        for field, value in (("family", "future_confirm_family"),
                             ("year", 2040), ("group", "G11"),
                             ("fixture", False)):
            with self.subTest(field=field):
                self.deny(dict(FIXTURE, **{field: value}))

    def test_mixed_csv_path_and_unknown_keys_rejected(self):
        self.deny(dict(FIXTURE, path="data/mixed_train_test.csv"))
        self.deny(dict(FIXTURE, model_path="weights.zip"))
        self.deny("data/mixed_train_test.csv")

    def test_missing_and_invalid_types_are_not_coerced(self):
        bad = FIXTURE.copy()
        bad.pop("fixture")
        self.deny(bad)
        for field, value in (("year", True), ("fixture", 1),
                             ("family", "../data"), ("group", None)):
            with self.subTest(field=field):
                self.deny(dict(FIXTURE, **{field: value}))

    def test_policy_cannot_silently_grant_research_or_validation(self):
        for spec in (dict(FIXTURE, fixture=False),
                     dict(FIXTURE, split="validation")):
            with self.assertRaises(PermissionError):
                ResearchInputPolicy([spec])

    def test_empty_authority_denied(self):
        with self.assertRaises(ValueError):
            ResearchInputPolicy([])

    def test_policy_and_input_alias_are_immutable(self):
        supplied = FIXTURE.copy()
        policy = ResearchInputPolicy([supplied])
        supplied["year"] = 2040
        self.assertEqual(policy.validate(FIXTURE)["year"], 2033)
        with self.assertRaises(FrozenInstanceError):
            policy.approved_keys = frozenset()

    def test_factory_receives_copy_not_original_request(self):
        request = FIXTURE.copy()
        def fake(**spec):
            spec["year"] = 2040
            return spec
        self.assertEqual(self.policy.create_store(request, factory=fake)["year"], 2040)
        self.assertEqual(request, FIXTURE)
        self.assertEqual(self.policy.validate(request), FIXTURE)

    def test_denied_environment_request_precedes_project_import(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.startswith(("v31", "v3", "torch", "stable_baselines3")):
                raise AssertionError("project/model import happened before denial")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            with self.assertRaises(PermissionError):
                build_environment([dict(FIXTURE, split="test")],
                                  policy=self.policy, level="upper")

    def test_every_request_checked_before_environment_import(self):
        with patch("builtins.__import__",
                   side_effect=AssertionError("must reject before import")):
            with self.assertRaises(PermissionError):
                build_environment([FIXTURE, dict(FIXTURE, family="unapproved")],
                                  policy=self.policy, level="upper")

    def test_invalid_level_seed_and_empty_requests_refused(self):
        for kwargs in (dict(level="bad"), dict(level="upper", seed=True)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    build_environment([FIXTURE], policy=self.policy, **kwargs)
        with self.assertRaises(ValueError):
            build_environment([], policy=self.policy, level="upper")


if __name__ == "__main__":
    unittest.main()
