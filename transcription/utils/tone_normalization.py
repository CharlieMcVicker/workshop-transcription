#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script and module to perform tone normalization on Cherokee phonetic transcripts.
"""

import csv
import re
import unicodedata
import sys
import os


def respell_consonants(s: str) -> str:
    # Rewrite rules for aspiration marking
    # Order matters: t->th before d->t, k->kh before g->k
    # Exception: ts should stay ts (not become ths)

    # We want to replace 't' with 'th' only if it's not followed by 's'
    s = re.sub(r"t(?!s)", "th", s)

    rules = [
        # ("t", "th"), # Handled by regex above to allow for ts exception
        ("d", "t"),
        ("k", "kh"),
        ("g", "k"),
        ("j", "ts"),
        ("ch", "tsh"),
        ("hn", "nh"),
        ("hl", "lh"),
        ("hy", "yh"),
        ("hw", "wh"),
        ("?", "'"),
        ("’", "'"),
    ]
    for old, new in rules:
        s = s.replace(old, new)

    s = re.sub(r"sl(?=[aeiouv])", "slh", s)
    s = re.sub(r"([^ht])s", r"\1hs", s)

    return s


VOWELS = set("aeiouvAEIOUV")

# Dropped marks (occurred only once in the dataset)
DROPPED_MARKS = ["\u0302\u003a\u0301", "\u003a\u003a", "\u0307"]  # ̂:́  # ::  # ̇

# Tone replacement dictionary
# Rules:
# - Replace entries with a colon first to avoid partial matches
# - "COMBINING CIRCUMFLEX ACCENT" with no colon same as with colon
# - "COMBINING GRAVE ACCENT" without colon same as with colon
# - "COMBINING DOUBLE ACUTE ACCENT" without colon same as with colon
TONE_DICT = {
    # Colons first
    "\u0300\u003a": "21",  # ̀: (Grave with colon)
    "\u030b\u003a": "34",  # ̋: (Double acute with colon)
    "\u0302\u003a": "32",  # ̂: (Circumflex with colon)
    "\u0301\u003a": "33",  # ́: (Acute with colon)
    "\u030c\u003a": "23",  # ̌: (Caron with colon)
    "\u003a": "22",  # : (Colon alone)
    # Without colons (mapped to same numbers where specified)
    "\u0300": "21",  # ̀ (Grave without colon)
    "\u030b": "34",  # ̋ (Double acute without colon)
    "\u0302": "32",  # ̂ (Circumflex without colon)
    "\u0301": "3",  # ́ (Acute without colon)
    "\u030c": "23",  # ̌ (Caron without colon)
}

# Non-word-final vowel with no mark gets '2'
UNMARKED_VAL = "2"


def replace_tones(text):
    """
    Applies Unicode NFD normalization and replaces tone/diacritic sequences
    following vowels with their mapped numerical values.

    Returns:
        (normalized_text, should_drop)
        - normalized_text: str (or None if dropped)
        - should_drop: bool (True if text contains low-frequency dropped marks)
    """
    if not isinstance(text, str):
        return "", False

    nfd_text = unicodedata.normalize("NFD", text)

    # Check for dropped marks
    for mark in DROPPED_MARKS:
        if mark in nfd_text:
            return None, True

    # Split by spaces to preserve word structure
    text_normaled_all = respell_consonants(nfd_text)
    words = text_normaled_all.split(" ")
    new_words = []

    for word in words:
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            if char in VOWELS:
                # Scan for following diacritics/colons
                j = i + 1
                seq = []
                while j < n:
                    next_char = word[j]
                    is_combining = unicodedata.category(next_char).startswith("M")
                    if is_combining or next_char in [":", "ː"]:
                        seq.append(next_char)
                        j += 1
                    else:
                        break

                seq_str = "".join(seq)
                is_non_final = any(c.isalpha() for c in word[j:])

                replacement = ""
                if seq_str:
                    # Look up sequence in replacement dictionary
                    if seq_str in TONE_DICT:
                        replacement = TONE_DICT[seq_str]
                    else:
                        # Fallback for unexpected sequences
                        replacement = seq_str
                else:
                    # Both non-word-final and word-final vowels with no mark get UNMARKED_VAL
                    replacement = UNMARKED_VAL

                new_word.append(char + replacement)
                i = j
            else:
                new_word.append(char)
                i += 1
        new_words.append("".join(new_word))

    return " ".join(new_words), False


def remove_tones_and_double_vowels(text):
    """
    Normalizes Cherokee text:
    1. Unicode NFD normalization.
    2. Respell consonants.
    3. For vowels (aeiouvAEIOUV):
       - If followed by a colon (':' or 'ː'), double the vowel and remove the colon and all combining diacritics.
       - If not followed by a colon, keep the vowel single and remove all combining diacritics.
    """
    if not isinstance(text, str):
        return "", False

    nfd_text = unicodedata.normalize("NFD", text)

    # Check for dropped marks
    for mark in DROPPED_MARKS:
        if mark in nfd_text:
            return None, True

    # Respell consonants
    text_normaled_all = respell_consonants(nfd_text)
    words = text_normaled_all.split(" ")
    new_words = []

    for word in words:
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            char = word[i]
            if char in VOWELS:
                # Scan for following diacritics/colons
                j = i + 1
                seq = []
                while j < n:
                    next_char = word[j]
                    is_combining = unicodedata.category(next_char).startswith("M")
                    if is_combining or next_char in [":", "ː"]:
                        seq.append(next_char)
                        j += 1
                    else:
                        break

                seq_str = "".join(seq)
                if ":" in seq_str or "ː" in seq_str:
                    new_word.append(char * 2)
                else:
                    new_word.append(char)
                i = j
            else:
                new_word.append(char)
                i += 1
        new_words.append("".join(new_word))

    return " ".join(new_words), False





def main():
    if len(sys.argv) < 3:
        print("Usage: python tone_normalize.py <input_csv> <output_csv> [text_column]")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    text_col = sys.argv[3] if len(sys.argv) > 3 else "phonetic"

    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        sys.exit(1)

    print(f"Processing '{input_csv}' -> '{output_csv}' (column: '{text_col}')...")

    processed_count = 0
    dropped_count = 0

    with open(input_csv, mode="r", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        fieldnames = reader.fieldnames

        rows_to_write = []
        for row in reader:
            val = row.get(text_col, "")
            norm_val, should_drop = replace_tones(val)
            if should_drop or norm_val is None:
                dropped_count += 1
                continue

            row[text_col] = norm_val
            rows_to_write.append(row)
            processed_count += 1

    with open(output_csv, mode="w", encoding="utf-8", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)

    print(
        f"Done! Processed {processed_count} rows. Dropped {dropped_count} rows containing rare marks."
    )


if __name__ == "__main__":
    main()
