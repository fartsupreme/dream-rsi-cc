import unittest

from drsi.bm25 import BM25, tokenize


class TokenizeTest(unittest.TestCase):
    def test_lowercases_and_drops_stopwords_and_punctuation(self):
        self.assertEqual(tokenize("The Shellsort, of the RUNS!"), ["shellsort", "runs"])

    def test_keeps_attempt_refs_and_numbers_with_units(self):
        toks = tokenize("see #42 at 2,290 us and 2^61-1")
        self.assertIn("#42", toks)
        self.assertIn("2^61", toks)

    def test_hyphenated_terms_kept_whole_and_split(self):
        toks = tokenize("short-gap tail")
        self.assertIn("short-gap", toks)
        self.assertIn("short", toks)
        self.assertIn("gap", toks)


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.docs = {
            "a": "shellsort gap sequence short gap tail below comparison floor",
            "b": "radix bucket width and cache line density sweep",
            "c": "galloping merge law run boundary linear bound",
            "d": "shellsort index stability separate pass decoupled tail target",
        }
        self.index = BM25(self.docs)

    def test_top_hit_is_most_relevant(self):
        hits = self.index.top_k("galloping merge of the run", 2)
        self.assertEqual(hits[0][0], "c")

    def test_scores_descending_and_k_respected(self):
        hits = self.index.top_k("shellsort tail", 3)
        self.assertLessEqual(len(hits), 3)
        scores = [s for _, s in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual({h[0] for h in hits[:2]}, {"a", "d"})

    def test_no_overlap_returns_empty(self):
        self.assertEqual(self.index.top_k("zzzz qqqq", 3), [])

    def test_empty_corpus(self):
        self.assertEqual(BM25({}).top_k("anything", 5), [])


if __name__ == "__main__":
    unittest.main()
