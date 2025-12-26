#!/usr/bin/env python3
"""
Generate HuggingFace BERT tokenizer benchmark data.

This script generates the expected tokenization results from HuggingFace's
BertWordPieceTokenizer for use in compatibility benchmarks.

Requirements:
    pip install tokenizers

Usage:
    python scripts/generate_hf_benchmark_data.py

Output:
    hf_benchmark_data.json - JSON file with expected tokenization results
"""

import json
import sys
from pathlib import Path

try:
    from tokenizers import BertWordPieceTokenizer
except ImportError:
    print("ERROR: tokenizers package not installed.")
    print("Run: pip install tokenizers")
    sys.exit(1)


SINGLE_ENCODING_TESTS = [
    ("Simple greeting", "Hello, world!"),
    ("Classic pangram", "The quick brown fox jumps over the lazy dog."),
    ("Subword tokenization", "tokenization"),
    ("Complex subword", "unbelievable"),
    ("Accented text (café)", "café"),
    ("Accented text (naïve)", "naïve"),
    ("Accented text (résumé)", "résumé"),
    ("Numbers", "12345"),
    ("Multiple punctuation", "!!??"),
    ("BERT model name", "BERT is a transformer-based model."),
    ("NLP text", "I love NLP!"),
    ("Contraction (I'm)", "I'm happy"),
    ("Contraction (don't)", "don't"),
    ("Hyphenated word", "state-of-the-art"),
    ("Machine learning sentence", "Machine learning is amazing!"),
    ("Subword (preprocessing)", "preprocessing"),
    ("Subword (playing)", "playing"),
    ("Subword (transformers)", "transformers"),
    ("Question sentence", "Hello, how are you?"),
    ("Ellipsis in text", "test...test"),
    ("Year in sentence", "The year is 2024."),
    ("Framework names", "TensorFlow and PyTorch are popular."),
    ("Long NLP sentence", "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."),
    ("Mixed symbols", "Hello @user #hashtag $100 50%"),
    ("Email in sentence", "Email: test@example.com"),
    ("Complex URL", "https://www.example.com/path?query=value"),
    ("German umlaut (Zürich)", "Zürich"),
    ("Portuguese (São Paulo)", "São Paulo"),
    ("German umlaut (München)", "München"),
    ("Multiple accented words", "naïve café résumé"),
    ("French accent (fiancée)", "fiancée"),
    ("Spanish tilde (piñata)", "piñata"),
    ("Very long word 1", "antidisestablishmentarianism"),
    ("Very long word 2", "supercalifragilisticexpialidocious"),
    ("Very long word 3", "internationalization"),
    ("Scientific term", "deoxyribonucleic"),
    ("Numbers in sentence", "I have 3 cats and 2 dogs."),
    ("Temperature with degree", "The temperature is 72.5°F."),
    ("Date format", "2023-12-25"),
    ("Time format", "10:30 AM"),
    ("Formatted number", "1,234,567.89"),
    ("Contraction (won't)", "won't"),
    ("Contraction (it's)", "it's"),
    ("Contraction (they're)", "they're"),
    ("Contraction (we've)", "we've"),
    ("Hyphenated (well-known)", "well-known"),
    ("Hyphenated (self-driving)", "self-driving"),
    ("Hyphenated (twenty-first)", "twenty-first"),
    ("Exclamation with question", "Really?!"),
    ("Trailing ellipsis", "Wait..."),
    ("Multiple exclamations", "Hello!!!"),
    ("Mixed punctuation", "What?!?!"),
    ("Double quoted", '"Hello"'),
    ("Single quoted", "'Hello'"),
    ("Quote in sentence", "He said 'hello'"),
    ("Underscore identifier", "function_name"),
    ("camelCase", "camelCase"),
    ("PascalCase", "PascalCase"),
    ("snake_case", "snake_case"),
    ("SCREAMING_CASE", "SCREAMING_CASE"),
    ("Greek letter beta", "β-testing"),
    ("Greek letter alpha", "α-version"),
    ("Greek letter mu", "µ-controller"),
    ("Repeated character", "a" * 50),
    ("Repeated word", "test " * 10),
    ("Punctuation only", ".,!?;:"),
]

PAIR_ENCODING_TESTS = [
    ("Simple QA pair", "What is NLP?", "Natural language processing."),
    ("Question-Answer pair", "Who created BERT?", "Google AI researchers."),
    ("Short pair", "Hello", "World"),
]

EDGE_CASE_TESTS = [
    ("Empty string", ""),
    ("Whitespace only", "   "),
    ("Tab and newline", "\t\n"),
    ("Single character", "a"),
    ("Single punctuation", "."),
    ("Multiple spaces between words", "hello    world"),
    ("All caps", "HELLO WORLD"),
    ("Mixed case", "HeLLo WoRLd"),
    ("URL-like text", "https://example.com"),
    ("Email-like text", "test@example.com"),
    ("Currency symbols", "$100"),
    ("Percentage", "50%"),
]

OFFSET_TESTS = [
    ("Simple word offsets", "hello"),
    ("Two words offsets", "hello world"),
    ("Subword offsets", "tokenization"),
]


def main():
    vocab_path = Path("vocab.txt")
    if not vocab_path.exists():
        print("ERROR: vocab.txt not found in current directory.")
        sys.exit(1)

    print("Loading HuggingFace BertWordPieceTokenizer...")
    tokenizer = BertWordPieceTokenizer(
        str(vocab_path),
        lowercase=True,
        strip_accents=True,
    )

    results = {
        "single_encoding": [],
        "pair_encoding": [],
        "edge_cases": [],
        "offset_tests": [],
    }

    print("\n--- Single Encoding Tests ---")
    for name, text in SINGLE_ENCODING_TESTS:
        encoding = tokenizer.encode(text)
        result = {
            "name": name,
            "input": text,
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask,
            "offsets": encoding.offsets,
        }
        results["single_encoding"].append(result)
        print(f"  {name}: {encoding.tokens}")

    print("\n--- Pair Encoding Tests ---")
    for name, text_a, text_b in PAIR_ENCODING_TESTS:
        encoding = tokenizer.encode(text_a, text_b)
        result = {
            "name": name,
            "input_a": text_a,
            "input_b": text_b,
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask,
            "offsets": encoding.offsets,
        }
        results["pair_encoding"].append(result)
        print(f"  {name}: {encoding.tokens}")

    print("\n--- Edge Case Tests ---")
    for name, text in EDGE_CASE_TESTS:
        encoding = tokenizer.encode(text)
        result = {
            "name": name,
            "input": text,
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "type_ids": encoding.type_ids,
            "attention_mask": encoding.attention_mask,
            "offsets": encoding.offsets,
        }
        results["edge_cases"].append(result)
        display = repr(text) if len(text) < 20 else f"{text[:17]}..."
        print(f"  {name}: {encoding.tokens}")

    print("\n--- Offset Tests ---")
    for name, text in OFFSET_TESTS:
        encoding = tokenizer.encode(text)
        result = {
            "name": name,
            "input": text,
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "offsets": encoding.offsets,
        }
        results["offset_tests"].append(result)
        print(f"  {name}: offsets={encoding.offsets}")

    output_path = Path("hf_benchmark_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_path}")

    print("\n--- Dart Test Case Generation ---")
    print("Copy the following to update benchmark test cases:\n")

    _generate_dart_code(results)


def _generate_dart_code(results):
    """Generate Dart code for test cases."""

    print("// Single Encoding Test Cases")
    print("const _singleEncodingTestCases = [")
    for r in results["single_encoding"]:
        print(f"  SingleEncodingTestCase(")
        print(f"    name: '{r['name']}',")
        print(f"    input: {repr(r['input'])},")
        print(f"    expectedTokens: {r['tokens']},")
        print(f"    expectedIds: {r['ids']},")
        print(f"    expectedTypeIds: {r['type_ids']},")
        print(f"    expectedAttentionMask: {r['attention_mask']},")
        print(f"  ),")
    print("];")
    print()

    print("// Pair Encoding Test Cases")
    print("const _pairEncodingTestCases = [")
    for r in results["pair_encoding"]:
        print(f"  PairEncodingTestCase(")
        print(f"    name: '{r['name']}',")
        print(f"    inputA: {repr(r['input_a'])},")
        print(f"    inputB: {repr(r['input_b'])},")
        print(f"    expectedTokens: {r['tokens']},")
        print(f"    expectedIds: {r['ids']},")
        print(f"    expectedTypeIds: {r['type_ids']},")
        print(f"  ),")
    print("];")
    print()

    print("// Edge Case Test Cases")
    print("const _edgeCaseTestCases = [")
    for r in results["edge_cases"]:
        print(f"  EdgeCaseTestCase(")
        print(f"    name: '{r['name']}',")
        print(f"    input: {repr(r['input'])},")
        print(f"    expectedTokens: {r['tokens']},")
        print(f"    expectedIds: {r['ids']},")
        print(f"  ),")
    print("];")
    print()

    print("// Offset Test Cases")
    print("const _offsetTestCases = [")
    for r in results["offset_tests"]:
        offsets_dart = [f"({o[0]}, {o[1]})" for o in r["offsets"]]
        print(f"  OffsetTestCase(")
        print(f"    name: '{r['name']}',")
        print(f"    input: {repr(r['input'])},")
        print(f"    expectedOffsets: [{', '.join(offsets_dart)}],")
        print(f"  ),")
    print("];")


if __name__ == "__main__":
    main()
