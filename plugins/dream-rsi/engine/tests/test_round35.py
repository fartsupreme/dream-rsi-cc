"""Round 35: the cross-vendor review of rounds 31 to 34 (Grok), each finding reproduced here first.

- `drsi prune` re-parents an attempt that continued from a removed one but keeps its cell, which names the removed
  attempt: replay then cannot reach it, and a world's best drops to what is left. The dream still compared on such a
  world. On four of them a revision that opens one root a batch deployed over one that opens four at once (replay
  0.5278 against 0.5270), while on the same attempts as live recorded them the wide policy is ahead (0.2118 against
  0.2029). The dream now compares only on worlds whose cells are exactly what live records: each root in a
  root:<n> slot, each other attempt opened from its parent.
- A root slot cell was read with str.isdigit and int, so "root:00" and an Arabic-Indic "root:\\u0660" were slot 0,
  and a second attempt in that slot was hidden. A slot is "root:" and a plain decimal number now, as live writes it.
- The stream's result was the last whole result line anywhere in the file, so a line appended after an error result
  (by anything that kept the file open after the call) turned the call into a success. Claude Code ends a session
  with exactly one result event as its last line; any other shape is now a failure. Every line is read as JSON, so
  the reading does not depend on how "type" is spelled.
- Transcript names were the UTC time to the microsecond: two developer calls in one microsecond collided, and the
  second call's FileExistsError ended the dream. Names now carry a random suffix after the time.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drsi import cli
from drsi.agent import AgentResult, ClaudeAgent
from drsi.dream import SEED_POLICY, run_dream
from drsi.question import ReplayQuestion
from drsi.store import DEFAULT_CONFIG, Campaign
from drsi.worlds import comparable
from tests.helpers import with_block
from tests.test_dream import Dev
from tests.test_round33 import RESULT, stream
from tests.test_round34 import FileOnlyRunner

WIDE = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        if question.rounds:
            return []
        return question.legal_roots()[:4]
    # EVOLVE-BLOCK-END
"""
SERIAL = """    # EVOLVE-BLOCK-START
    def select_batch(self, question, closed, state):
        if question.rounds >= 4:
            return []
        return question.legal_roots()[:1]
    # EVOLVE-BLOCK-END
"""


def pruned(n):
    """Live opened b (a failed attempt) from root 0, then c (score 1.0) from b; a prune removed b and re-parented c to
    root 0, keeping c's cell, which names b."""
    nodes = [{"id": f"r{n}_{i}", "parent": None, "cell": f"root:{i}", "score": s + 0.0001 * n, "valid": True,
              "fail_class": "ok"} for i, s in enumerate((0.51, 0.50, 0.50, 0.50))]
    nodes.append({"id": f"c{n}", "parent": f"r{n}_0", "cell": f"b{n}", "score": 1.0, "valid": True,
                  "fail_class": "ok"})
    return {"id": f"iter{n + 1:04d}", "baseline": 0.0, "nodes": nodes}


class PrunedWorldTest(unittest.TestCase):
    def test_a_world_a_prune_rewrote_is_not_compared_on(self):
        self.assertEqual(comparable([pruned(0)]), [])

    def test_a_root_in_no_slot_is_not_what_live_records(self):
        w = pruned(0)
        w["nodes"] = [n for n in w["nodes"] if n["parent"] is None] + [
            {"id": "x", "parent": None, "cell": "r0_0", "score": 0.2, "valid": True}]
        self.assertEqual(comparable([w]), [])

    def test_a_world_as_live_records_it_is_still_compared_on(self):
        w = pruned(0)
        w["nodes"][-1] = dict(w["nodes"][-1], cell="r0_0")  # opened from its parent, as live records
        self.assertEqual(len(comparable([w])), 1)

    def test_the_revision_that_gains_only_on_pruned_worlds_is_not_deployed(self):
        seed = SEED_POLICY.read_text()
        with tempfile.TemporaryDirectory() as d:
            pdir = Path(d) / "policy"
            pdir.mkdir()
            (pdir / "method.py").write_text(with_block(WIDE)(seed))
            cfg = {"search": {"W": 4, "K1": 6}, "dream": dict(DEFAULT_CONFIG["dream"], M=1)}
            rep = run_dream(pdir, [pruned(n) for n in range(4)], Dev(with_block(SERIAL)), cfg, Path(d) / "logs")
        self.assertFalse(rep["deployed"], rep)
        self.assertTrue(rep["skipped"])


class SlotTest(unittest.TestCase):
    def world(self, cell):
        return {"id": "w", "baseline": 0.0, "nodes": [
            {"id": "a", "parent": None, "cell": "root:0", "score": 0.1, "valid": True},
            {"id": "b", "parent": None, "cell": cell, "score": 1.0, "valid": True}]}

    def test_only_a_plain_decimal_slot_is_a_slot(self):
        for cell in ("root:00", "root:01", "root:٠", "root:١", "root:+1", "root: 1", "root:1\n"):
            q = ReplayQuestion(self.world(cell), 2, max_probes=4)
            self.assertEqual(sorted(q._slot), [0], cell)  # b has no slot: nothing reaches it
            self.assertEqual(comparable([self.world(cell)]), [], cell)
        q = ReplayQuestion(self.world("root:1"), 2, max_probes=4)
        self.assertEqual(sorted(q._slot), [0, 1])
        self.assertEqual(len(comparable([self.world("root:1")])), 1)

    def test_two_roots_in_one_slot_are_not_what_live_records(self):
        self.assertEqual(comparable([self.world("root:0")]), [])


class ResultShapeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.n = 0

    def tearDown(self):
        self.tmp.cleanup()

    def run_stream(self, text, rc=0):
        self.n += 1
        agent = ClaudeAgent(model="opus", tools="Read", runner=FileOnlyRunner(text, rc=rc))
        return agent.run("/tmp/ws", "p", transcript=Path(self.tmp.name) / f"{self.n}.jsonl")

    def test_a_success_appended_after_an_error_is_a_failure(self):
        bad = dict(RESULT, subtype="error_max_turns", is_error=True)
        self.assertFalse(self.run_stream(stream({"type": "system"}, bad, RESULT)).ok)

    def test_anything_after_the_result_is_a_failure(self):
        self.assertFalse(self.run_stream(stream(RESULT, {"type": "assistant"})).ok)
        self.assertFalse(self.run_stream(stream(RESULT) + "trailing words\n").ok)

    def test_one_result_as_the_last_line_is_read_however_type_is_spelled(self):
        self.assertTrue(self.run_stream(stream({"type": "system"}, RESULT)).ok)
        spaced = json.dumps(RESULT).replace('"type": "result"', '"type" : "result"')
        self.assertTrue(self.run_stream(stream({"type": "system"}) + spaced + "\n").ok)
        escaped = json.dumps(RESULT).replace('"type"', '"\\u0074ype"', 1)
        self.assertTrue(self.run_stream(stream({"type": "system"}) + escaped + "\n").ok)

    def test_a_result_inside_another_event_is_not_the_result(self):
        inner = {"type": "user", "message": {"content": [{"type": "result", "subtype": "success"}]}}
        self.assertFalse(self.run_stream(stream(inner)).ok)


class NameTest(unittest.TestCase):
    def test_two_calls_in_one_microsecond_get_two_transcripts(self):
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("t", {}, home=Path(d))
            seen = []

            def fake_run(agent, cwd, prompt, add_dirs=(), transcript=None):
                seen.append(Path(transcript))
                Path(transcript).parent.mkdir(parents=True, exist_ok=True)
                with open(transcript, "x"):
                    pass
                return AgentResult(ok=True)
            with mock.patch.object(ClaudeAgent, "run", fake_run), mock.patch.object(cli.time, "time",
                                                                                   return_value=1700000000.5):
                dev = cli.make_developer(camp)
                dev(Path(d), "one")
                dev(Path(d), "two")
            self.assertEqual(len({p.name for p in seen}), 2)
            self.assertTrue(all(p.name.startswith("20231114T221320500000Z") for p in seen))


if __name__ == "__main__":
    unittest.main()
