#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset for TypoTransformer
"""

import os
import re
import csv
import torch
import multiprocessing
import random
import json
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader, IterableDataset

from utils.mimic_tools import MIMICPseudonymizer
from scripts.utils import sanitize_text, clean_text

from fastDamerauLevenshtein import damerauLevenshtein

DEFAULT_MAX_CHARACTER_POSITIONS = 64
char_tokens = list(r"0123456789abcdefghijklmnopqrstuvwxyz+-*/^.,;:=!?'()[]{}&")


###############################################################################
# CharTokenizer
###############################################################################
class CharTokenizer(object):
    """
    A simple character-level tokenizer with optional BOS/EOS/pad tokens.
    """
    def __init__(self, max_length=DEFAULT_MAX_CHARACTER_POSITIONS):
        self.max_length = max_length
        self.bos, self.pad, self.eos, self.unk = ['<s>', '<pad>', '</s>', '<unk>']
        self.bos_index, self.pad_index, self.eos_index, self.unk_index = 0, 1, 2, 3

        # Build char_to_id / id_to_char for special tokens + characters.
        self.char_to_id = {}
        self.id_to_char = {}
        special_tokens = [self.bos, self.pad, self.eos, self.unk]
        for i, c in enumerate(special_tokens + char_tokens):
            self.char_to_id[c] = i
            self.id_to_char[i] = c

    def tokenize(self, text, eos_bos=True, padding_end=False, max_length=None, output_token_ids=False):
        """
        Tokenize text at the character level.
        - eos_bos: whether to prepend <s> and append </s>.
        - padding_end: whether to pad the sequence to a fixed length.
        - max_length: override default if needed.
        - output_token_ids: return numeric IDs if True; else return token strings and attention mask.
        """
        assert isinstance(text, str)
        if max_length is None:
            max_length = self.max_length
        max_seq_len = max_length - 2 if eos_bos else max_length
        tokens = []
        attention_mask = []

        # Convert characters up to max_seq_len
        for char in text[:max_seq_len]:
            tokens.append(char if char in self.char_to_id else self.unk)
            attention_mask.append(1)

        # Optionally add BOS/EOS tokens
        if eos_bos:
            tokens.insert(0, self.bos)
            tokens.append(self.eos)
            attention_mask.insert(0, 1)
            attention_mask.append(1)

        # Optionally pad sequence
        if padding_end:
            pad_len = max_length - len(tokens)
            tokens.extend([self.pad] * pad_len)
            attention_mask.extend([0] * pad_len)

        if output_token_ids:
            return self.convert_tokens_to_ids(tokens), attention_mask
        return tokens, attention_mask

    def convert_tokens_to_ids(self, tokens):
        return [self.char_to_id[t] for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().detach().tolist()
        return [self.id_to_char[i] for i in ids]


###############################################################################
# Map-style Dataset
###############################################################################
class TypoDataset(Dataset):
    """
    This dataset gives examples of (typo, context, correction).
    """
    def __init__(self, tsv_file, bert_tokenizer, typo_tokenizer, num_process=None, max_rows=None):
        assert os.path.exists(tsv_file), f'{tsv_file} does not exist'
        self.tsv_file = tsv_file
        self.bert_tokenizer = bert_tokenizer
        self.typo_tokenizer = typo_tokenizer

        print(f"Read file {tsv_file}... ", end="", flush=True)
        self.csv_rows = []
        with open(self.tsv_file, "r", encoding="utf-8", errors="replace") as fd:
            reader = csv.reader(fd, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:  # skip header
                    continue
                self.csv_rows.append(row)
                if max_rows is not None and len(self.csv_rows) >= max_rows:
                    break
        print(f"{len(self.csv_rows)} rows")

        # Determine the number of processes to use.
        if num_process is None:
            num_process = multiprocessing.cpu_count()
        num_process = min(num_process, len(self.csv_rows))
        print(f"Parsing rows using {num_process} process{'es' if num_process > 1 else ''}")

        self.examples = []
        fixed_chunksize = 100

        if num_process == 1:
            for row in tqdm(self.csv_rows, total=len(self.csv_rows)):
                self.examples.append(self._parse_row(row))
        else:
            pool = multiprocessing.Pool(num_process)
            for example in tqdm(pool.imap(self._parse_row, self.csv_rows, chunksize=fixed_chunksize),
                                total=len(self.csv_rows)):
                self.examples.append(example)
            pool.close()
            pool.join()

    def _make_sentence(self, tokens_left, tokens_right, seq_length=128):
        len_left = len(tokens_left)
        len_right = len(tokens_right)
        cut_len = len_left + len_right - (seq_length - 1)
        if cut_len > 0:
            cut_left = len_left - seq_length // 2
            cut_right = len_right - (seq_length - 1) // 2
            if cut_left < 0:
                cut_left, cut_right = 0, cut_left + cut_right
            elif cut_right < 0:
                cut_left, cut_right = cut_left + cut_right, 0
        else:
            cut_left, cut_right = 0, 0

        tokens_left = tokens_left[cut_left:]
        tokens_right = tokens_right[:len(tokens_right) - cut_right]
        tokens = tokens_left + [self.bert_tokenizer.mask_token] + tokens_right
        attention_mask = [1] * len(tokens_left) + [1] + [1] * len(tokens_right)

        if len(tokens) < seq_length:
            pad_len = seq_length - len(tokens)
            tokens.extend([self.bert_tokenizer.pad_token] * pad_len)
            attention_mask.extend([0] * pad_len)
        return tokens, attention_mask

    def _parse_row(self, row):
        """
        Convert a CSV row to an example.
        Expected row format: [ex_id, note_id, typo, left, right, correct]
        """
        ex_id, note_id, typo, left, right, correct = row
        tokens_left = self.bert_tokenizer.tokenize(left)[0]  # assuming tokenize() returns (tokens, mask)
        tokens_right = self.bert_tokenizer.tokenize(right)[0]
        context_tokens, context_attention_mask = self._make_sentence(tokens_left, tokens_right)
        context_token_ids = self.bert_tokenizer.convert_tokens_to_ids(context_tokens)

        typo_token_ids, typo_attention_mask = self.typo_tokenizer.tokenize(
            typo, eos_bos=True, padding_end=False, output_token_ids=True)
        correct_token_ids, correct_attention_mask = self.typo_tokenizer.tokenize(
            correct, eos_bos=True, padding_end=False, output_token_ids=True)

        return {
            'example_id': int(ex_id),
            'note_id': int(note_id),
            'context_tokens': context_token_ids,
            'context_attention_mask': context_attention_mask,
            'typo': typo,
            'typo_token_ids': typo_token_ids,
            'typo_attention_mask': typo_attention_mask,
            'correct': correct,
            'correct_token_ids': correct_token_ids,
            'correct_attention_mask': correct_attention_mask
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def get_collate_fn(self, stack=True):
        def _collate_fn(examples):
            max_typo_len = max(len(e['typo_token_ids']) for e in examples)
            max_correct_len = max(len(e['correct_token_ids']) for e in examples)
            common_len = max(max_typo_len, max_correct_len) + 1
            for e in examples:
                e['typo_token_ids'] = e['typo_token_ids'] + [self.typo_tokenizer.pad_index] * (common_len - len(e['typo_token_ids']))
                e['typo_attention_mask'] = e['typo_attention_mask'] + [0] * (common_len - len(e['typo_attention_mask']))
                e['correct_token_ids'] = e['correct_token_ids'] + [self.typo_tokenizer.pad_index] * (common_len - len(e['correct_token_ids']))
                e['correct_attention_mask'] = e['correct_attention_mask'] + [0] * (common_len - len(e['correct_attention_mask']))
            if stack:
                batch = {}
                for key in examples[0].keys():
                    # Do not tensor-ify raw text fields.
                    if key in ['typo', 'correct', 'left_context', 'right_context']:
                        batch[key] = [e[key] for e in examples]
                    else:
                        batch[key] = torch.tensor([e[key] for e in examples])
                return batch
            else:
                return examples
        return _collate_fn


###############################################################################
# Iterable (Streaming) Dataset
###############################################################################
class TypoOnlineDataset(IterableDataset):
    """
    This streaming dataset reads from multiple CSV files at random.
    It implements its own __len__ by summing the rows in the CSV files.
    """
    def __init__(self, csv_dir, dict_file, bert_tokenizer, typo_tokenizer,
                 max_word_corruptions=2, do_substitution=True, do_transposition=True,
                 no_corruption_prob=0.0, min_word_len=3):
        super().__init__()
        self.csv_dir = csv_dir
        self.csv_fnames = [f for f in os.listdir(self.csv_dir)
                           if f.startswith('NOTEEVENTS') and f.endswith('.csv')]
        self.dict_file = dict_file
        with open(self.dict_file, 'r') as fd:
            self.dictionary = set(json.load(fd))
        self.bert_tokenizer = bert_tokenizer
        self.typo_tokenizer = typo_tokenizer

        self.max_word_corruptions = max_word_corruptions
        self.do_substitution = do_substitution
        self.do_transposition = do_transposition
        self.word_corrupter = WordCorrupter(self.max_word_corruptions, self.do_substitution, self.do_transposition)
        self.no_corruption_prob = no_corruption_prob
        self.min_word_len = min_word_len

        mimic_tools_dir = 'scripts/mimic-tools/lists'
        self.mimic_pseudo = MIMICPseudonymizer(mimic_tools_dir)

        # Compute total number of samples in the CSV files.
        self.total_samples = self._count_total_rows()

    def get_collate_fn(self, stack=True):
        # Reuse collate function from TypoDataset.
        return TypoDataset.get_collate_fn(self)

    def _count_total_rows(self):
        total = 0
        for fname in self.csv_fnames:
            csv_path = os.path.join(self.csv_dir, fname)
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                total += len(df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
        return total

    def __len__(self):
        return self.total_samples

    def _load_random_csv(self):
        self.csv_fname = random.choice(self.csv_fnames)
        self.csv_path = os.path.join(self.csv_dir, self.csv_fname)
        self.df_note = pd.read_csv(self.csv_path, low_memory=False)
        self.note_iterrows = self.df_note.iterrows()

    def _random_word_context(self, text, max_trial=10):
        puncs = list("[]!\"#$%&'()*+,./:;<=>?@\\^_`{|}~-")
        words = text.split()
        trial = 0
        done = False
        while trial < max_trial and not done:
            trial += 1
            w_idx = random.randint(0, len(words) - 1)
            word, left_res, right_res = words[w_idx], [], []
            if (len(word) >= self.min_word_len and
                word.lower() in self.dictionary and
                len(word) < DEFAULT_MAX_CHARACTER_POSITIONS - 4):
                done = True
            else:
                if word and word[0] in puncs:
                    word, left_res = word[1:], [word[0]]
                if not word:
                    continue
                if word and word[-1] in puncs:
                    word, right_res = word[:-1], [word[-1]]
                if not word:
                    continue
                if (len(word) < self.min_word_len or
                    (word.lower() not in self.dictionary) or
                    len(word) >= DEFAULT_MAX_CHARACTER_POSITIONS - 4):
                    continue
                right_snip = ' '.join(words[w_idx+1:w_idx+5])
                if '**]' in right_snip and '[**' not in right_snip:
                    continue
                left_snip = ' '.join(words[max(0, w_idx-4):w_idx])
                if '[**' in left_snip and '**]' not in left_snip:
                    continue
                done = True

        if done:
            return word, ' '.join(words[:w_idx] + left_res), ' '.join(right_res + words[w_idx+1:])
        else:
            raise ValueError('failed to choose word')

    def _process_note(self, note):
        note = re.sub('\n', ' ', note)
        note = re.sub('\t', ' ', note)
        return sanitize_text(clean_text(note))

    def _parse_row(self, row):
        """
        Updated _parse_row for streaming examples.
        Expected row format: [ex_id, note_id, typo, left, right, correct]
        """
        ex_id, note_id, typo, left, right, correct = row
        typo_token_ids, typo_attention_mask = self.typo_tokenizer.tokenize(
            typo, eos_bos=True, padding_end=False, output_token_ids=True)
        correct_token_ids, correct_attention_mask = self.typo_tokenizer.tokenize(
            correct, eos_bos=True, padding_end=False, output_token_ids=True)
        return {
            "example_id": ex_id,
            "note_id": note_id,
            "typo": typo,
            "typo_token_ids": typo_token_ids,
            "typo_attention_mask": typo_attention_mask,
            "left_context": left,
            "right_context": right,
            "correct": correct,
            "correct_token_ids": correct_token_ids,
            "correct_attention_mask": correct_attention_mask
        }

    def __iter__(self):
        self._load_random_csv()
        return self

    def __next__(self):
        # Loop until a valid note is found.
        while True:
            try:
                _, row = next(self.note_iterrows)
            except StopIteration:
                self._load_random_csv()
                _, row = next(self.note_iterrows)
            note_id = int(row.ROW_ID)
            note = row.TEXT.strip()
            if len(note) < 2000:
                continue
            try:
                correct, left, right = self._random_word_context(note)
            except Exception:
                continue
            break

        correct = correct.lower()
        if random.uniform(0, 1) >= self.no_corruption_prob:
            typo = self.word_corrupter.corrupt_word(correct)
        else:
            typo = correct

        left = self.mimic_pseudo.pseudonymize(left)
        left = self._process_note(left)
        left = ' '.join(left.split(' ')[-128:])
        right = self.mimic_pseudo.pseudonymize(right)
        right = self._process_note(right)
        right = ' '.join(right.split(' ')[:128])
        temp_csv_row = [-1, note_id, typo, left, right, correct]
        return self._parse_row(temp_csv_row)


###############################################################################
# Utility: WordCorrupter
###############################################################################
class WordCorrupter(object):
    def __init__(self, max_corruptions=2, do_substitution=True, do_transposition=True):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.max_corruptions = max_corruptions
        self.do_substitution = do_substitution
        self.do_transposition = do_transposition
        self.operation_list = ['ins', 'del']
        if self.do_substitution:
            self.operation_list.append('sub')
        if self.do_transposition:
            self.operation_list.append('tra')

    def random_alphabet(self):
        return random.choice(self.alphabet)

    def single_corruption(self, word):
        while True:
            oper = random.choice(self.operation_list)
            if oper == "del":
                if len(word) == 1:
                    continue
                cidx = random.randint(0, len(word) - 1)
                ret = word[:cidx] + word[cidx+1:]
                break
            elif oper == "ins":
                cidx = random.randint(0, len(word))
                ret = word[:cidx] + self.random_alphabet() + word[cidx:]
                break
            elif oper == "sub":
                cidx = random.randint(0, len(word) - 1)
                while True:
                    c = self.random_alphabet()
                    if c != word[cidx]:
                        ret = word[:cidx] + c + word[cidx+1:]
                        break
                break
            elif oper == "tra":
                if len(word) == 1:
                    continue
                cidx = random.randint(0, len(word) - 2)
                if word[cidx+1] == word[cidx]:
                    continue
                ret = word[:cidx] + word[cidx+1] + word[cidx] + word[cidx+2:]
                break
            else:
                raise ValueError(f'Wrong operation {oper}')
        return ret

    def corrupt_word(self, word_original):
        num_corruption = random.randint(1, self.max_corruptions)
        while True:
            word = word_original
            for i in range(num_corruption):
                word = self.single_corruption(word)
            if word_original != word:
                break
        return word

print("Character Language Model and related classes defined successfully.")
