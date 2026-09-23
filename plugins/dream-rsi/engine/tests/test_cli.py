import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from drsi import cli
from tests.helpers import ScriptedLLM, fp_for, ids_in_block

FIXTURE = Path(__file__).parent / "fixtures" / "attempts_sample.jsonl"


def fake_llm(prompt, schema):
    props = schema.get("properties", {})
    if "items" in props and "mechanism" in json.dumps(props):
        return {"items": [fp_for(i, family_hint="iso") for i in ids_in_block(prompt)]}
    if "families" in props:
        return {"families": [{"id": "F01", "name": "shellsort", "description": "d", "boundary": "b"}]}
    if "items" in props:
        return {"items": [{"id": i, "family": "F01"} for i in re.findall(r"^([^|\s]+)\|", prompt, re.M)]}
    if "directions" in props:
        return {"directions": [{"direction": "go elsewhere", "rationale": "r", "avoids": ["F01"]}]}
    if "verdict" in props:
        return {"verdict": "duplicate", "family": "F01", "nearest_ids": ["1"], "what_differs": "",
                "targets_gate": "speed", "addresses_stopper": False, "doubts": "", "rationale": "same thing"}
    raise AssertionError(f"unexpected schema {schema}")


class CLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = {"DRSI_HOME": self.tmp.name}
        self._old = {k: os.environ.get(k) for k in ("DRSI_HOME", "DRSI_CAMPAIGN")}
        os.environ["DRSI_HOME"] = self.tmp.name
        os.environ.pop("DRSI_CAMPAIGN", None)
        self.llm = ScriptedLLM(fake_llm)
        cli.LLM_FACTORY = lambda cfg, role: self.llm

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cli.LLM_FACTORY = None
        self.tmp.cleanup()

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_full_layer1_flow(self):
        code, out = self.run_cli("init", "demo", "--goal", "break the wall")
        self.assertEqual(code, 0)
        code, out = self.run_cli("import", "-c", "demo", str(FIXTURE))
        self.assertEqual(code, 0)
        self.assertIn("9 new attempts", out)
        code, out = self.run_cli("fingerprint", "-c", "demo")
        self.assertIn("9 fingerprinted", out)
        code, out = self.run_cli("families", "-c", "demo")
        self.assertEqual(code, 0)
        code, out = self.run_cli("map", "-c", "demo")
        self.assertIn("9 attempts", out)
        self.assertIn("go elsewhere", out)
        root = Path(self.tmp.name) / "campaigns" / "demo"
        self.assertTrue((root / "map.md").exists())
        code, out = self.run_cli("check", "-c", "demo", "a shellsort gap sequence again")
        self.assertEqual(code, 4)
        self.assertIn("DUPLICATE", out)
        self.assertEqual(len((root / "logs" / "checks.jsonl").read_text().splitlines()), 1)

    def test_check_json_output(self):
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(FIXTURE))
        self.run_cli("fingerprint", "-c", "demo")
        self.run_cli("families", "-c", "demo")
        code, out = self.run_cli("check", "-c", "demo", "--json", "again")
        self.assertEqual(json.loads(out)["verdict"], "duplicate")

    def test_sync_reimports_recorded_sources_and_indexes_new_rows(self):
        grown = Path(self.tmp.name) / "ledger.jsonl"
        grown.write_text(FIXTURE.read_text())
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(grown))
        self.run_cli("fingerprint", "-c", "demo")
        self.run_cli("families", "-c", "demo")
        with open(grown, "a") as fh:
            fh.write(json.dumps({"id": 10, "supersedes": [9], "candidate": "NEW", "falsifiable": "f",
                                 "verdict": "v", "check_cmd": "c", "next": "n"}) + "\n")
        code, out = self.run_cli("sync", "-c", "demo")
        self.assertEqual(code, 0)
        self.assertIn("1 new attempts", out)
        tree_lines = (Path(self.tmp.name) / "campaigns" / "demo" / "tree.jsonl").read_text().splitlines()
        last = json.loads(tree_lines[-1])
        self.assertEqual(last["id"], "10")
        self.assertEqual(last["fingerprint"]["family"], "F01")

    def test_single_campaign_is_default(self):
        self.run_cli("init", "only", "--goal", "g")
        code, out = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("only", out)

    def test_status_reports_counts(self):
        self.run_cli("init", "demo", "--goal", "g")
        self.run_cli("import", "-c", "demo", str(FIXTURE))
        code, out = self.run_cli("status", "-c", "demo")
        self.assertIn("attempts: 9", out)
        self.assertIn("fingerprinted: 0", out)


class CLILayer2Test(unittest.TestCase):
    def setUp(self):
        from tests.test_live import SCORER, stub_worker
        from tests.test_scorer_workspace import make_repo
        self.tmp = tempfile.TemporaryDirectory()
        self._old = {k: os.environ.get(k) for k in ("DRSI_HOME", "DRSI_CAMPAIGN")}
        os.environ["DRSI_HOME"] = self.tmp.name
        os.environ.pop("DRSI_CAMPAIGN", None)
        self.src = make_repo(Path(self.tmp.name))
        cli.LLM_FACTORY = lambda cfg, role: ScriptedLLM(fake_llm)
        cli.WORKER_FACTORY = lambda camp: stub_worker()
        from drsi.agent import AgentResult
        cli.DEVELOPER_FACTORY = lambda camp: (lambda sb, prompt: AgentResult(ok=True))
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["init", "toy", "--goal", "maximise value", "--scorer", SCORER, "--repo", str(self.src)])
            cli.main(["config", "-c", "toy", "--set", 'workspace.mutable=["value.txt"]', "--set", "search.W=2",
                      "--set", "search.K1=2", "--set", "dream.M=1", "--set", "live.require_check=false"])

    def tearDown(self):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cli.LLM_FACTORY = cli.WORKER_FACTORY = cli.DEVELOPER_FACTORY = None
        self.tmp.cleanup()

    def run_cli(self, *argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_config_set_parses_json_values(self):
        cfg = json.loads((Path(self.tmp.name) / "campaigns" / "toy" / "campaign.json").read_text())
        self.assertEqual(cfg["workspace"]["mutable"], ["value.txt"])
        self.assertEqual(cfg["search"]["W"], 2)
        self.assertIs(cfg["live"]["require_check"], False)

    def test_baseline(self):
        code, out = self.run_cli("baseline", "-c", "toy")
        self.assertEqual(code, 0)
        self.assertIn("baseline: 1.0", out)

    def test_run_then_replay_then_dream(self):
        code, out = self.run_cli("run", "-c", "toy", "--rounds", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("iter0001", out)
        code, out = self.run_cli("replay", "-c", "toy")
        self.assertEqual(code, 0, out)
        self.assertIn("reward", out)
        code, out = self.run_cli("dream", "-c", "toy")
        self.assertEqual(code, 0, out)
        self.assertIn("kept the incumbent", out)

    def test_replay_without_worlds_says_so(self):
        code, out = self.run_cli("replay", "-c", "toy")
        self.assertEqual(code, 1)
        self.assertIn("no replay worlds", out)


class PendingTest(unittest.TestCase):
    def test_pending_checks_are_recent_passing_and_unrecorded(self):
        from drsi.store import Campaign, make_node, utcnow
        with tempfile.TemporaryDirectory() as d:
            camp = Campaign.create("p", {}, home=Path(d))
            camp.tree.add(make_node(id="n1", parent=None, artifacts={"novelty_ticket": "recorded0000"}))
            camp.checks_path.parent.mkdir(parents=True, exist_ok=True)
            rows = [{"ticket": "recorded0000", "verdict": "novel", "checked": utcnow(), "proposal": "a", "node": "n1"},
                    {"ticket": "fresh0000000", "verdict": "variant", "checked": utcnow(), "proposal": "b", "node": "n2"},
                    {"ticket": "dup000000000", "verdict": "duplicate", "checked": utcnow(), "proposal": "c", "node": "n3"},
                    {"ticket": "old000000000", "verdict": "novel", "checked": "2020-01-01T00:00:00Z", "proposal": "d",
                     "node": "n4"},
                    {"ticket": "session00000", "verdict": "novel", "checked": utcnow(), "proposal": "e"}]
            camp.checks_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            got = cli.pending_checks(camp, hours=12)
            self.assertEqual([p["ticket"] for p in got], ["fresh0000000"])  # in-session checks are not claims


class CheckSerialisationTest(unittest.TestCase):
    def test_concurrent_checks_see_earlier_claims(self):
        import threading
        import time as _t
        tmp = tempfile.TemporaryDirectory()
        old = os.environ.get("DRSI_HOME")
        os.environ["DRSI_HOME"] = tmp.name
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["init", "s", "--goal", "g"])
            spans, prompts, lock = [], [], threading.Lock()

            def judge(prompt, schema):
                t0 = _t.time()
                _t.sleep(0.3)
                with lock:
                    spans.append((t0, _t.time()))
                    prompts.append(prompt)
                return {"verdict": "novel", "family": "F00", "nearest_ids": [], "what_differs": "",
                        "targets_gate": "", "addresses_stopper": True, "doubts": "", "rationale": "r"}
            cli.LLM_FACTORY = lambda cfg, role: ScriptedLLM(judge)
            # one prior attempt so the judge is consulted
            from drsi.store import Campaign, make_node
            Campaign.open("s").tree.add(make_node(id="1", parent=None, proposal="old idea"))

            def run(text, node):
                with redirect_stdout(io.StringIO()):
                    cli.main(["check", "-c", "s", "--node", node, text])
            ts = [threading.Thread(target=run, args=(f"idea number {i}", f"iter0001-00{i}")) for i in (1, 2)]
            for th in ts:
                th.start()
            for th in ts:
                th.join()
            # judged concurrently, but whichever claimed second saw the first claim as in flight
            self.assertEqual(sum(1 for p in prompts if "pending:iter0001-00" in p), 1)
            rows = [json.loads(l) for l in Campaign.open("s").checks_path.read_text().splitlines()]
            self.assertEqual(sorted(r["verdict"] for r in rows), ["claim", "claim", "novel", "novel"])
        finally:
            cli.LLM_FACTORY = None
            if old is None:
                os.environ.pop("DRSI_HOME", None)
            else:
                os.environ["DRSI_HOME"] = old
            tmp.cleanup()


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("DRSI_HOME")
        os.environ["DRSI_HOME"] = self.tmp.name
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli.main(["init", "g", "--goal", "x"])
        self.checks = Path(self.tmp.name) / "campaigns" / "g" / "logs" / "checks.jsonl"
        self.checks.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("DRSI_HOME", None)
        else:
            os.environ["DRSI_HOME"] = self._old
        self.tmp.cleanup()

    def gate(self, *extra):
        buf = io.StringIO()
        with redirect_stdout(buf):
            return cli.main(["gate", "-c", "g", *extra])

    def write(self, verdict, when):
        with open(self.checks, "a") as fh:
            fh.write(json.dumps({"ticket": "t", "verdict": verdict, "checked": when}) + "\n")

    def test_no_checks_blocks(self):
        self.assertEqual(self.gate(), 2)

    def test_recent_novel_check_passes(self):
        from drsi.store import utcnow
        self.write("novel", utcnow())
        self.assertEqual(self.gate(), 0)

    def test_duplicate_only_blocks(self):
        from drsi.store import utcnow
        self.write("duplicate", utcnow())
        self.assertEqual(self.gate(), 2)

    def test_stale_check_blocks(self):
        self.write("novel", "2020-01-01T00:00:00Z")
        self.assertEqual(self.gate("--max-age-hours", "6"), 2)


if __name__ == "__main__":
    unittest.main()
