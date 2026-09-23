import tempfile
import unittest
from pathlib import Path

from drsi.fingerprint import FP_SCHEMA, fingerprint_nodes
from drsi.importer import import_jsonl
from drsi.store import Tree
from tests.helpers import ScriptedLLM, fp_for, ids_in_block

FIXTURE = Path(__file__).parent / "fixtures" / "attempts_sample.jsonl"


class FingerprintTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tree = Tree(Path(self.tmp.name) / "tree.jsonl")
        import_jsonl(self.tree, FIXTURE)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fills_every_node_and_persists(self):
        llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
        stats = fingerprint_nodes(self.tree, llm, goal="find X", batch=4, workers=2)
        self.assertEqual(stats["done"], 9)
        again = Tree(self.tree.path)
        for n in again.nodes():
            self.assertEqual(n["fingerprint"]["mechanism"], f"mechanism of {n['id']}")

    def test_batches_respect_size(self):
        llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
        fingerprint_nodes(self.tree, llm, goal="g", batch=4, workers=1)
        self.assertEqual(sorted(len(ids_in_block(p)) for p in llm.prompts), [1, 4, 4])

    def test_skips_already_fingerprinted(self):
        self.tree.update("1", fingerprint={"mechanism": "kept"})
        llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
        stats = fingerprint_nodes(self.tree, llm, goal="g", batch=20, workers=1)
        self.assertEqual(stats["done"], 8)
        self.assertEqual(self.tree.get("1")["fingerprint"], {"mechanism": "kept"})

    def test_missing_ids_retried_alone_then_marked_error(self):
        def answer(p, s):
            ids = ids_in_block(p)
            return {"items": [fp_for(i) for i in ids if i != "5"]}
        llm = ScriptedLLM(answer)
        stats = fingerprint_nodes(self.tree, llm, goal="g", batch=20, workers=1)
        self.assertEqual(stats["done"], 8)
        self.assertEqual(stats["failed"], 1)
        self.assertIn("error", self.tree.get("5")["fingerprint"])
        # the retry for id 5 was a batch of one
        self.assertIn(["5"], [ids_in_block(p) for p in llm.prompts])

    def test_foreign_ids_in_answer_ignored(self):
        llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)] + [fp_for("999")]})
        fingerprint_nodes(self.tree, llm, goal="g", batch=20, workers=1)
        self.assertNotIn("999", self.tree)

    def test_prompt_carries_trimmed_fields_and_goal(self):
        llm = ScriptedLLM(lambda p, s: {"items": [fp_for(i) for i in ids_in_block(p)]})
        fingerprint_nodes(self.tree, llm, goal="THE GOAL", batch=20, workers=1)
        prompt = llm.prompts[0]
        self.assertIn("THE GOAL", prompt)
        self.assertIn("SHELLSORT-GAPS-SW", prompt)
        self.assertNotIn("v" * 1200, prompt)  # 5000-char verdict is cut for the classifier

    def test_llm_exception_marks_batch_failed_without_crashing(self):
        def boom(p, s):
            raise RuntimeError("down")
        stats = fingerprint_nodes(self.tree, ScriptedLLM(boom), goal="g", batch=20, workers=1)
        self.assertEqual(stats["done"], 0)
        self.assertEqual(stats["failed"], 9)

    def test_schema_requires_outcome_enum(self):
        item = FP_SCHEMA["properties"]["items"]["items"]
        self.assertIn("outcome", item["required"])
        self.assertIn("refuted", item["properties"]["outcome"]["enum"])


if __name__ == "__main__":
    unittest.main()
