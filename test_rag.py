import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from rag_minimal import RAG, Chunk, chunk_text, load_documents, tokenize
from rag_minimal.text import _stem

PROJECT_DIR = Path(__file__).resolve().parent
CLI = PROJECT_DIR / "cli.py"

SAMPLE_WATERING = (
    "# Cactus\n\nWater the cactus every two weeks in summer.\n\n"
    "# Ferns\n\nFerns need constant humidity indoors.\n"
)


class TestTokenizer(unittest.TestCase):
    def test_lowercase_and_diacritics(self):
        self.assertIn("intencion", tokenize("Intención"))

    def test_stopwords_removed(self):
        self.assertNotIn("de", tokenize("el costo de la cosa"))

    def test_bigrams_bridge_stopwords(self):
        tokens = tokenize("riego del cactus")
        self.assertIn("cactu", tokens)
        self.assertIn("rieg_cactu", tokens)

    def test_stemming_unifies_plurals(self):
        self.assertEqual(tokenize("las plantas"), tokenize("planta"))

    def test_stemming_unifies_english_plurals(self):
        for singular, plural in (("plant", "plants"), ("water", "waters"), ("week", "weeks")):
            with self.subTest(plural=plural):
                self.assertIn(_stem(plural), tokenize(plural))
                self.assertEqual(tokenize(plural), tokenize(singular))

    def test_short_words_are_protected(self):
        self.assertIn("agua", tokenize("agua"))
        self.assertIn("less", tokenize("less"))

    def test_unigrams_only_mode(self):
        self.assertNotIn("rieg_cactu", tokenize("riego del cactus", ngram_max=1))


class TestChunker(unittest.TestCase):
    def test_splits_and_numbers_chunks(self):
        text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6))
        chunks = chunk_text(text, "doc.md", max_chars=400, overlap=80)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c.id for c in chunks], list(range(len(chunks))))

    def test_heading_is_metadata_not_body(self):
        chunks = chunk_text("# Guide\n\nBody one.\n\n## Section\n\nBody two.", "g.md")
        self.assertEqual(chunks[0].heading, "Guide")
        self.assertEqual(chunks[1].heading, "Section")
        self.assertNotIn("#", " ".join(c.text for c in chunks))

    def test_hash_that_is_not_a_heading_stays_in_the_body(self):
        chunks = chunk_text("keep me\n\n#1 best deal today\n\ntail", "g.md")
        self.assertEqual(chunks[0].heading, "")
        self.assertIn("#1 best deal today", chunks[0].text)

    def test_empty_text(self):
        self.assertEqual(chunk_text("   \n\n  ", "vacio.md"), [])


class TestVectors(unittest.TestCase):
    def test_vectors_are_l2_normalized(self):
        index = RAG.from_chunks([Chunk(0, "a.md", "", "el gato bebe leche cada mañana")])
        vec = index.vectors[0]
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vec.values())), 1.0, places=9)

    def test_idf_downweights_common_terms(self):
        chunks = [Chunk(i, "a.md", "", f"palabra comun numero {i}") for i in range(4)]
        chunks[3] = Chunk(3, "a.md", "", "palabra comun numero 3 y tambien untermRaro")
        index = RAG.from_chunks(chunks)
        self.assertLess(index.idf["comun"], index.idf["untermrar"])


class TestSearch(unittest.TestCase):
    def setUp(self):
        chunks = [
            Chunk(0, "riego.md", "Cactus", "El cactus necesita poca agua. Conviene regar el cactus cada dos semanas en verano."),
            Chunk(1, "riego.md", "Helechos", "Los helechos son plantas de interior que necesitan humedad constante."),
            Chunk(2, "riego.md", "Suculentas", "Las suculentas almacenan agua en sus hojas. El sustrato debe ser drenante."),
            Chunk(3, "perros.md", "Mascotas", "Perros venenosos: cactus con espinas. Gatos odian el cactus."),
        ]
        self.index = RAG.from_chunks(chunks)

    def test_ranks_relevant_chunk_first(self):
        hits = self.index.search("cada cuanto regar el cactus")
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.id, 0)

    def test_scores_sorted_descending(self):
        scores = [h.score for h in self.index.search("sustrato drenante para suculentas", top_k=4)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLessEqual(max(scores), 1.0 + 1e-9)

    def test_top_k_respected(self):
        self.assertEqual(len(self.index.search("agua", top_k=2)), 2)

    def test_out_of_vocabulary_returns_nothing(self):
        self.assertEqual(self.index.search("zzzz qqqq"), [])

    def test_min_score_filters(self):
        self.assertEqual(self.index.search("agua", top_k=5, min_score=0.9), [])

    def test_hit_metadata(self):
        hit = self.index.search("helechos humedad")[0]
        self.assertEqual(hit.source, "riego.md")
        self.assertEqual(hit.heading, "Helechos")
        self.assertEqual(hit.location(), "riego.md > Helechos")


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip_preserves_ranking(self):
        chunks = [Chunk(i, "a.md", f"H{i}", f"contenido numero {i} sobre riego") for i in range(5)]
        index = RAG.from_chunks(chunks)
        before = [h.chunk.id for h in index.search("riego")]

        with tempfile.TemporaryDirectory() as tmp:
            path = index.save(Path(tmp) / "index.json")
            self.assertTrue(path.exists())
            loaded = RAG.load(path)

        self.assertEqual(len(loaded), len(index))
        self.assertEqual([h.chunk.id for h in loaded.search("riego")], before)

    def test_load_rejects_future_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            path.write_text('{"version": 999, "chunks": []}', encoding="utf-8")
            with self.assertRaises(ValueError):
                RAG.load(path)


class TestLoadDocuments(unittest.TestCase):
    def test_reads_txt_and_md_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notas.txt").write_text("uno", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub" / "guia.md").write_text("dos", encoding="utf-8")
            (root / "ignorado.pdf").write_bytes(b"%PDF")

            found = dict(load_documents(root))

        self.assertEqual(found, {"notas.txt": "uno", "sub/guia.md": "dos"})

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "solo.md"
            path.write_text("hola", encoding="utf-8")
            self.assertEqual(load_documents(path), [("solo.md", "hola")])


class CLITestCase(unittest.TestCase):
    def run_cli(self, *args, stdin=None, cwd=None):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            capture_output=True,
            text=True,
            input=stdin,
            cwd=cwd,
            timeout=120,
        )

    @contextmanager
    def sample_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "watering.md").write_text(SAMPLE_WATERING, encoding="utf-8")
            self.assertEqual(self.run_cli("index", str(root), stdin="").returncode, 0)
            yield root


class TestCLI(CLITestCase):
    def test_index_then_ask(self):
        with self.sample_index() as root:
            self.assertTrue((root / "index.json").exists())

            asked = self.run_cli("ask", "how often to water the cactus", str(root), "--json")
            self.assertEqual(asked.returncode, 0, asked.stderr)
            hits = json.loads(asked.stdout)["hits"]
            self.assertTrue(hits)
            self.assertIn("cactus", hits[0]["text"].lower())
            self.assertGreater(hits[0]["score"], 0)

    def test_ask_without_index_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            asked = self.run_cli("ask", "something", tmp)
            self.assertEqual(asked.returncode, 1)
            self.assertIn("No index at", asked.stderr)

    def test_index_on_empty_directory_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            indexed = self.run_cli("index", tmp)
            self.assertEqual(indexed.returncode, 1)
            self.assertIn("Empty index", indexed.stderr)

    def test_module_entry_point(self):
        with self.sample_index() as root:
            run = subprocess.run(
                [sys.executable, "-m", "rag_minimal", "ask", "cactus", str(root)],
                capture_output=True,
                text=True,
                cwd=PROJECT_DIR,
                timeout=120,
            )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("watering.md", run.stdout)


class TestChat(CLITestCase):
    def test_answers_a_question(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="how often to water the cactus\n/quit\n")
            self.assertEqual(chat.returncode, 0, chat.stderr)
            self.assertIn("Question: how often to water the cactus", chat.stdout)
            self.assertIn("watering.md", chat.stdout)
            self.assertIn("Bye.", chat.stdout)

    def test_ctrl_d_exits_cleanly(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="")
            self.assertEqual(chat.returncode, 0, chat.stderr)
            self.assertIn("Bye.", chat.stdout)

    def test_help_and_sources(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="/help\n/sources\n/quit\n")
            self.assertIn("/sources", chat.stdout)
            self.assertIn("watering.md  (2 chunks)", chat.stdout)

    def test_top_command_limits_results(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="/top 1\nwater cactus humidity\n/quit\n")
            self.assertIn("top_k = 1", chat.stdout)
            self.assertEqual(chat.stdout.count("score="), 1)

    def test_bad_command_arguments_do_not_crash(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="/top abc\n/min-score 5\n/nope\ncactus\n/quit\n")
            self.assertEqual(chat.returncode, 0, chat.stderr)
            self.assertIn("top must be a number", chat.stdout)
            self.assertIn("min-score must be between", chat.stdout)
            self.assertIn("Unknown command: /nope", chat.stdout)
            self.assertIn("score=", chat.stdout)

    def test_no_match_does_not_exit_the_loop(self):
        with self.sample_index() as root:
            chat = self.run_cli("chat", str(root), stdin="zzzz qqqq\ncactus\n/quit\n")
            self.assertIn("No matches.", chat.stdout)
            self.assertIn("Bye.", chat.stdout)

    def test_chat_without_index_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = self.run_cli("chat", tmp, stdin="/quit\n")
            self.assertEqual(chat.returncode, 1)
            self.assertIn("No index at", chat.stderr)


class TestExecutableEntryPoint(CLITestCase):
    def test_shebang_and_exec_bit(self):
        self.assertTrue(os.access(CLI, os.X_OK), "cli.py should be executable")
        self.assertTrue(CLI.read_text(encoding="utf-8").startswith("#!/usr/bin/env -S uv run --script"))

    def test_runs_as_an_executable_from_any_directory(self):
        indexed = self.run_cli("index", str(PROJECT_DIR / "docs"))
        self.assertEqual(indexed.returncode, 0, indexed.stderr)

        elsewhere = tempfile.mkdtemp()
        try:
            run = subprocess.run(
                [str(CLI), "ask", "how often to water the cactus", "-k", "1"],
                capture_output=True,
                text=True,
                cwd=elsewhere,
                timeout=120,
            )
        finally:
            shutil.rmtree(elsewhere, ignore_errors=True)

        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("watering.md", run.stdout)


if __name__ == "__main__":
    unittest.main()