"""Tests con unittest (stdlib). Ejecuta: python -m unittest -v"""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rag import RAG, Chunk, chunk_text, load_documents, tokenize


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

    def test_short_words_are_protected(self):
        self.assertIn("agua", tokenize("agua"))

    def test_unigrams_only_mode(self):
        self.assertNotIn("rieg_cactu", tokenize("riego del cactus", ngram_max=1))


class TestChunker(unittest.TestCase):
    def test_splits_and_numbers_chunks(self):
        text = "\n\n".join(f"Parrafo {i} " + "palabra " * 40 for i in range(6))
        chunks = chunk_text(text, "doc.md", max_chars=400, overlap=80)
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c.id for c in chunks], list(range(len(chunks))))

    def test_heading_is_metadata_not_body(self):
        chunks = chunk_text("# Guia\n\nCuerpo uno.\n\n## Seccion\n\nCuerpo dos.", "g.md")
        self.assertEqual(chunks[0].heading, "Guia")
        self.assertEqual(chunks[1].heading, "Seccion")
        self.assertNotIn("#", " ".join(c.text for c in chunks))

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


class TestCLI(unittest.TestCase):
    """Flujo completo: indexar por subprocess y consultar."""

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(Path(__file__).parent / "cli.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_index_then_ask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "docs"
            root.mkdir()
            (root / "riego.md").write_text("# Cactus\n\nRegar el cactus cada dos semanas en verano.\n", encoding="utf-8")

            indexed = self.run_cli("index", str(root))
            self.assertEqual(indexed.returncode, 0, indexed.stderr)
            self.assertTrue((root / "index.json").exists())

            asked = self.run_cli("ask", "cada cuanto regar el cactus", str(root), "--json")
            self.assertEqual(asked.returncode, 0, asked.stderr)
            hits = json.loads(asked.stdout)["hits"]
            self.assertTrue(hits)
            self.assertIn("cactus", hits[0]["text"].lower())
            self.assertGreater(hits[0]["score"], 0)

    def test_ask_without_index_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            asked = self.run_cli("ask", "algo", tmp)
            self.assertEqual(asked.returncode, 1)
            self.assertIn("No existe el índice", asked.stderr)


if __name__ == "__main__":
    unittest.main()