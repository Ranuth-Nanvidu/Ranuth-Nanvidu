import unittest
from datetime import datetime, timedelta, timezone

from model import config, readme, render, train

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)


def row(x, y, label="blue", user="alice", issue=1, at=NOW):
    return {"x": x, "y": y, "label": label, "user": user, "issue": issue, "timestamp": train.format_time(at)}


def train_event(user, x, y, number=99):
    return {"issue": {"number": number, "title": "train: blue", "body": f"x={x}, y={y}",
                      "user": {"login": user, "type": "User"}}}


class RateLimitTests(unittest.TestCase):
    def test_fourth_point_in_24h_is_rejected_with_reset_time(self):
        rows = train.seed_rows() + [
            row(1, 1, at=NOW - timedelta(hours=20)),
            row(2, 2, at=NOW - timedelta(hours=5)),
            row(3, 3, at=NOW - timedelta(hours=1)),
        ]
        result = train.handle_event(train_event("alice", 8, 1), rows, NOW)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("You can add another in 4h 0m.", result["comment"])

    def test_old_points_dont_count(self):
        rows = train.seed_rows() + [
            row(1, 1, at=NOW - timedelta(hours=30)),
            row(2, 2, at=NOW - timedelta(hours=5)),
            row(3, 3, at=NOW - timedelta(hours=1)),
        ]
        self.assertIsNone(train.rate_limit_reset(rows, "alice", NOW))
        self.assertEqual(train.handle_event(train_event("alice", 8, 1), rows, NOW)["status"], "trained")

    def test_limit_is_per_user_and_case_insensitive(self):
        rows = [row(i, i, user="Alice") for i in range(3)]
        self.assertIsNotNone(train.rate_limit_reset(rows, "alice", NOW))
        self.assertIsNone(train.rate_limit_reset(rows, "bob", NOW))

    def test_owner_is_exempt(self):
        rows = [row(i, i, user=config.OWNER) for i in range(5)]
        self.assertIsNone(train.rate_limit_reset(rows, config.OWNER, NOW))


class DuplicateTests(unittest.TestCase):
    def test_near_duplicate_from_same_user_is_rejected(self):
        rows = train.seed_rows() + [row(4.0, 4.0, at=NOW - timedelta(hours=30))]
        result = train.handle_event(train_event("alice", 4.03, 4.03), rows, NOW)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("almost the same spot", result["comment"])

    def test_other_users_and_far_points_are_fine(self):
        rows = [row(4.0, 4.0)]
        self.assertIsNone(train.find_duplicate(rows, "bob", 4.0, 4.0))
        self.assertIsNone(train.find_duplicate(rows, "alice", 4.1, 4.0))
        self.assertIsNotNone(train.find_duplicate(rows, "alice", 4.05, 4.0))


class CapTests(unittest.TestCase):
    def test_oldest_visitor_points_dropped_seeds_kept(self):
        seeds = train.seed_rows()
        visitors = [row(1, 1, user=f"u{i}", issue=i) for i in range(config.MAX_VISITOR_POINTS + 2)]
        kept = train.apply_cap(seeds[:2] + visitors[:1] + seeds[2:] + visitors[1:])
        kept_visitors = [r for r in kept if not train.is_seed(r)]
        self.assertEqual(len(kept_visitors), config.MAX_VISITOR_POINTS)
        self.assertEqual(kept_visitors[0]["issue"], 2)
        self.assertEqual(sum(train.is_seed(r) for r in kept), len(seeds))

    def test_under_cap_untouched(self):
        rows = train.seed_rows() + [row(1, 1)]
        self.assertEqual(train.apply_cap(rows), rows)


class ReadmeTests(unittest.TestCase):
    def test_replaces_only_between_markers(self):
        text = f"# Hi\n\nintro\n\n{readme.START}\nold stuff\n{readme.END}\n\nfooter\n"
        out = readme.replace_block(text, "NEW")
        self.assertEqual(out, f"# Hi\n\nintro\n\n{readme.START}\nNEW\n{readme.END}\n\nfooter\n")
        self.assertEqual(readme.replace_block(out, "NEWER").replace("NEWER", "NEW"), out)

    def test_appends_when_markers_missing(self):
        out = readme.replace_block("# Hi\nfooter\n", "NEW")
        self.assertEqual(out, f"# Hi\nfooter\n\n{readme.START}\nNEW\n{readme.END}\n")

    def test_keeps_crlf_line_endings(self):
        text = f"a\r\n{readme.START}\r\nold\r\n{readme.END}\r\nb\r\n"
        out = readme.replace_block(text, "one\ntwo")
        self.assertEqual(out, f"a\r\n{readme.START}\r\none\r\ntwo\r\n{readme.END}\r\nb\r\n")

    def test_block_content(self):
        block = readme.build_block(30, 4, "alice", 0.912)
        self.assertIn("Trained by 4 visitors · last update by @alice · accuracy 91%", block)
        self.assertIn("model-dark.svg?v=30", block)
        self.assertIn("title=train%3A%20orange&amp;body=x%3D5.0%2C%20y%3D5.0", block)
        self.assertNotIn("\n\n", block)


class RenderTests(unittest.TestCase):
    def test_svg_size_and_accessibility(self):
        rows = train.seed_rows() + [row(1, 9, "orange", user=f"u{i}", issue=i) for i in range(5)]
        points = train.as_points(rows)
        for theme in config.THEMES:
            svg = render.render(points, theme, newest=points[-1], visitors=5)
            self.assertLess(len(svg.encode()), 100_000)
            self.assertIn('role="img"', svg)
            self.assertIn("trained by 5 visitors", svg)
            self.assertIn('class="ring"', svg)
            self.assertIn("prefers-reduced-motion", svg)

    def test_runs_are_merged(self):
        runs = render.region_runs([(1, 1, "blue")])
        self.assertEqual(len(runs), config.GRID_ROWS)
        self.assertTrue(all(length == config.GRID_COLS for _, _, length, _, _ in runs))


if __name__ == "__main__":
    unittest.main()
