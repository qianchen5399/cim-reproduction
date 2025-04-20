# CIM-Transformer: Context-Sensitive Spelling Correction in Clinical Text  
**Reproduction of CHIL 2022 Paper**  
_"Context-Sensitive Spelling Correction of Clinical Text via Conditional Independence"_

---

## 📌 Overview

This repository contains a full reproduction of the CIM model described in the CHIL 2022 paper, focused on correcting misspellings in MIMIC-III clinical notes. It combines:

- ✅ Character-level language modeling
- ✅ BERT-based context encoding
- ✅ Edit-distance–aware decoding via beam search
- ✅ Custom pretraining, evaluation, and checkpoint visualization

All scripts have been restructured and adapted for use with MIMIC-III v1.4 on a local machine.

### 📥 Required Downloads (Before Running Anything)

Please download the following external resources and place them into the specified folders:

1. **MIMIC-III Clinical Notes**
   - Register and download from [PhysioNet](https://physionet.org/)
   - Required file: `NOTEEVENTS.csv`
   - Place under:
     ```
     data/mimic3/NOTEEVENTS.csv
     ```

2. **UMLS SPECIALIST Lexicon (2018AB)**
   - Source: [LSG Website](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/)
   - Download the following:
     - `LRWD` file  
     - `prevariants` file
   - Place them under:
     ```
     data/umls/LRWD
     data/umls/prevariants
     ```

3. **English Word List (DWYL)**
   - GitHub (specific commit):  
     https://github.com/dwyl/english-words/tree/7cb484d
   - File: `words.txt`
   - Place under:
     ```
     data/english_words/words.txt
     ```

4. **BlueBERT (PubMed + MIMIC-III)**
   - GitHub: [ncbi-nlp/bluebert](https://github.com/ncbi-nlp/bluebert)
   - Download: **"BlueBERT-Base, Uncased, PubMed+MIMIC-III"**
   - Files needed:
     - `bert_config.json`
     - `bert_model.ckpt`
     - `vocab.txt`
   - Place under:
     ```
     bert/ncbi_bert_base/
     ```

---





---

## 🧪 Preprocessing Pipeline

| Notebook                         | Purpose                                                              |
|----------------------------------|----------------------------------------------------------------------|
| `my_build_vocab_corpus.ipynb`   | Builds `lexicon.json` from UMLS + DWYL and splits MIMIC notes        |
| `my_preprocessing_clinspell.ipynb` | Aligns real annotated typos (Fivez et al., 2017) with MIMIC text     |
| `my_preprocessing_synthetic.ipynb` | Creates synthetic misspellings with context and saves TSV datasets   |

> 🔒 All patient note contexts are pseudonymized using `mimic-tools`.

---

## ⚙️ Training Pipeline

| Component                        | Description                                                         |
|----------------------------------|---------------------------------------------------------------------|
| `cim-reproduction.ipynb`        | Full pipeline for training, checkpointing, and evaluation           |
| `TypoDataset`, `TypoOnlineDataset` | Offline and streaming dataloaders for synthetic and real data        |
| `CharacterLanguageModel`        | BERT encoder + BART-style decoder for character-level modeling      |
| `generate_with_edit_distance`   | Beam search decoding with edit distance re-ranking (Eq. 8)          |

🧠 The model uses a combination of:
- Transformer-based encoder-decoder architecture
- Character-level decoding
- Edit distance loss during inference

---

## 📦 Setup

```bash
git clone https://github.com/your-username/cim-transformer-replica.git
cd cim-transformer-replica
pip install -r requirements.txt
