import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from model import train

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


class TitleTests(unittest.TestCase):
    def test_valid_titles(self):
        self.assertEqual(train.parse_title("train: blue"), ("train", "blue"))
        self.assertEqual(train.parse_title("  TRAIN : Orange  "), ("train", "orange"))
        self.assertEqual(train.parse_title("train:orange"), ("train", "orange"))
        self.assertEqual(train.parse_title("Predict"), ("predict", None))

    def test_unknown_label_is_flagged(self):
        self.assertEqual(train.parse_title("train: purple"), ("train", None))
        self.assertEqual(train.parse_title("train: blue please"), ("train", None))

    def test_unrelated_titles(self):
        for title in ("Typo in README", "training question", "predict the weather", "", None):
            self.assertIsNone(train.parse_title(title), title)


class BodyTests(unittest.TestCase):
    def test_valid_bodies(self):
        self.assertEqual(train.parse_body("x=3.2, y=1.7"), ((3.2, 1.7), None))
        self.assertEqual(train.parse_body("X: 4\nY : 6\nthanks!"), ((4.0, 6.0), None))
        self.assertEqual(train.parse_body("x=1.23456, y=0"), ((1.23, 0.0), None))
        self.assertEqual(train.parse_body("x=10, y=10"), ((10.0, 10.0), None))

    def test_hint_comment_is_ignored(self):
        body = "x=5.0, y=5.0\n\n<!-- Change the numbers (0–10) and press Submit -->"
        self.assertEqual(train.parse_body(body), ((5.0, 5.0), None))
        self.assertEqual(train.parse_body("<!-- x=1, y=1 -->")[1], ("format",))

    def test_bad_format(self):
        for body in ("", None, "somewhere in the middle", "x=3", "max=3, may=4", "x=abc, y=1"):
            self.assertEqual(train.parse_body(body), (None, ("format",)), body)

    def test_out_of_range(self):
        self.assertEqual(train.parse_body("x=12, y=3"), (None, ("range", 12.0, 3.0)))
        self.assertEqual(train.parse_body("x=-1, y=3"), (None, ("range", -1.0, 3.0)))


class EventTests(unittest.TestCase):
    def run_fixture(self, name):
        event = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        return train.handle_event(event, train.seed_rows(), NOW)

    def test_fixtures(self):
        expected = {
            "train_blue.json": "trained",
            "train_orange.json": "trained",
            "predict.json": "predicted",
            "bad_format.json": "rejected",
            "out_of_range.json": "rejected",
            "unrelated.json": "ignored",
            "bot_user.json": "ignored",
        }
        for name, status in expected.items():
            self.assertEqual(self.run_fixture(name)["status"], status, name)

    def test_train_adds_one_point(self):
        result = self.run_fixture("train_orange.json")
        new = result["rows"][-1]
        self.assertEqual((new["x"], new["y"], new["label"], new["user"], new["issue"]), (3.2, 1.7, "orange", "octocat", 12))
        self.assertEqual(len(result["rows"]), len(train.seed_rows()) + 1)
        self.assertEqual(result["commit"], "train: +1 orange point by @octocat (#12)")

    def test_predict_changes_nothing(self):
        result = self.run_fixture("predict.json")
        self.assertNotIn("rows", result)
        self.assertIn("the model says **orange**", result["comment"])

    def test_issue_text_is_never_echoed(self):
        event = {"issue": {"number": 3, "title": "train: blue", "body": "@everyone <script>x=99, y=1",
                           "user": {"login": "octocat", "type": "User"}}}
        comment = train.handle_event(event, train.seed_rows(), NOW)["comment"]
        self.assertNotIn("<script>", comment)
        self.assertNotIn("@everyone", comment)

    def test_bad_login_or_number_is_ignored(self):
        base = {"number": 3, "title": "predict", "body": "x=1, y=1", "user": {"login": "octocat", "type": "User"}}
        bad = [
            {**base, "user": {"login": "evil; rm -rf /", "type": "User"}},
            {**base, "number": "3; ls"},
            {**base, "user": None},
        ]
        for issue in bad:
            self.assertEqual(train.handle_event({"issue": issue}, train.seed_rows(), NOW)["status"], "ignored")
        self.assertEqual(train.handle_event({}, train.seed_rows(), NOW)["status"], "ignored")


if __name__ == "__main__":
    unittest.main()
