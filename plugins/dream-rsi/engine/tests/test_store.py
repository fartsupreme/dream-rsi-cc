import json
import tempfile
import unittest
from pathlib import Path

from drsi.store import Campaign, Tree, make_node


class TreeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tree.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _chain(self):
        t = Tree(self.path)
        t.add(make_node(id="1", parent=None, proposal="root attempt"))
        t.add(make_node(id="2", parent="1", proposal="refine"))
        t.add(make_node(id="3", parent="1", proposal="sibling"))
        t.add(make_node(id="4", parent="2", proposal="deeper"))
        return t

    def test_add_persists_and_reloads_in_order(self):
        self._chain()
        again = Tree(self.path)
        self.assertEqual([n["id"] for n in again.nodes()], ["1", "2", "3", "4"])

    def test_duplicate_id_rejected(self):
        t = self._chain()
        with self.assertRaises(ValueError):
            t.add(make_node(id="2", parent="1"))

    def test_unknown_parent_rejected(self):
        t = Tree(self.path)
        with self.assertRaises(ValueError):
            t.add(make_node(id="9", parent="missing"))

    def test_children_ancestors_leaves_roots(self):
        t = self._chain()
        self.assertEqual([n["id"] for n in t.children("1")], ["2", "3"])
        self.assertEqual([n["id"] for n in t.ancestors("4")], ["2", "1"])
        self.assertEqual(sorted(n["id"] for n in t.leaves()), ["3", "4"])
        self.assertEqual([n["id"] for n in t.roots()], ["1"])

    def test_update_rewrites_fields_and_survives_reload(self):
        t = self._chain()
        t.update("3", fingerprint={"family": "F1"}, score=0.5)
        again = Tree(self.path)
        self.assertEqual(again.get("3")["fingerprint"], {"family": "F1"})
        self.assertEqual(again.get("3")["score"], 0.5)

    def test_update_many_single_rewrite(self):
        t = self._chain()
        t.update_many({"1": {"score": 1.0}, "4": {"score": 4.0}})
        again = Tree(self.path)
        self.assertEqual(again.get("1")["score"], 1.0)
        self.assertEqual(again.get("4")["score"], 4.0)

    def test_two_instances_do_not_lose_each_others_writes(self):
        a = self._chain()
        b = Tree(self.path)                      # loaded before a's next write
        a.add(make_node(id="5", parent="4"))     # e.g. the live loop records an attempt
        b.update("1", score=9.0)                 # e.g. a concurrent `drsi sync` fingerprints
        b.add_many([make_node(id="6", parent="2")])
        again = Tree(self.path)
        self.assertEqual([n["id"] for n in again.nodes()], ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(again.get("1")["score"], 9.0)

    def test_torn_last_line_is_ignored_on_load(self):
        self._chain()
        with open(self.path, "a") as fh:
            fh.write('{"id": "7", "parent": "4", "prop')  # a writer caught mid-append
        self.assertEqual(len(Tree(self.path)), 4)

    def test_make_node_fills_schema_defaults(self):
        n = make_node(id="x", parent=None)
        for key in ("branch", "seq", "source", "created", "proposal", "text", "fingerprint",
                    "gates", "score", "valid", "fail_class", "artifacts", "worker", "ext"):
            self.assertIn(key, n)
        self.assertNotIn("cost", n)  # spending is not tracked
        self.assertEqual(n["source"], "live")

    def test_make_node_rejects_unknown_fields(self):
        with self.assertRaises(TypeError):
            make_node(id="x", parent=None, bogus=1)

    def test_branch_inherited_from_parent_when_unset(self):
        t = Tree(self.path)
        t.add(make_node(id="a", parent=None, branch=0))
        t.add(make_node(id="b", parent="a"))
        self.assertEqual(t.get("b")["branch"], 0)
        self.assertEqual(t.get("b")["seq"], 1)


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_writes_config_and_layout(self):
        c = Campaign.create("demo", {"goal": "go fast"}, home=self.home)
        self.assertTrue((c.root / "campaign.json").exists())
        cfg = json.loads((c.root / "campaign.json").read_text())
        self.assertEqual(cfg["goal"], "go fast")
        self.assertEqual(cfg["name"], "demo")
        for sub in ("policy", "trace_pool", "logs", "work"):
            self.assertTrue((c.root / sub).is_dir(), sub)

    def test_create_refuses_existing(self):
        Campaign.create("demo", {}, home=self.home)
        with self.assertRaises(FileExistsError):
            Campaign.create("demo", {}, home=self.home)

    def test_open_by_name_and_by_path(self):
        c = Campaign.create("demo", {}, home=self.home)
        self.assertEqual(Campaign.open("demo", home=self.home).root, c.root)
        self.assertEqual(Campaign.open(str(c.root), home=self.home).root, c.root)

    def test_open_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            Campaign.open("nope", home=self.home)

    def test_name_must_be_safe(self):
        with self.assertRaises(ValueError):
            Campaign.create("../escape", {}, home=self.home)

    def test_config_defaults_present(self):
        c = Campaign.create("demo", {}, home=self.home)
        cfg = c.config
        self.assertEqual(cfg["search"]["W"], 4)
        self.assertEqual(cfg["search"]["K1"], 6)
        self.assertEqual(cfg["dream"]["M"], 3)
        self.assertIn("model", cfg["llm"])
        # operator directive 2026-09-22: spending is neither capped nor tracked
        self.assertNotIn("budget", cfg)


if __name__ == "__main__":
    unittest.main()
