# 📚 BiTimeBERT Pretraining Guide

This guide walks you through the data preparation and tokenization steps required to pretrain the **BiTimeBERT** model using the **New York Times Annotated Corpus**.

---

## 🗂️ Step 1: Download the Raw Dataset

Download the **New York Times Annotated Corpus** from the official source:

🔗 [https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/GZC6PL](https://abacus.library.ubc.ca/dataset.xhtml?persistentId=hdl:11272.1/AB2/GZC6PL)

> 💡 **Tip**: After downloading, extract the archive and note the path to the raw data directory (referred to as `<RAW_DATA_PATH>` below).

---

## ⚙️ Step 2: Run Preprocessing Scripts

Navigate to the preprocessing directory and execute the scripts **in order**.

✅ **Expected output files**:
```
├── 0_corpus_train.txt
├── 0_corpus_val.txt
├── 0_corpus_test.txt
├── 0_docid2timestamp.pickle
└── docid2sentidx2temptokenizeinfor_dict.pickle
```
> ⚠️ **Important**: Ensure all scripts complete successfully before proceeding. Missing any intermediate file will cause tokenization or training to fail.

---

## 🔤 Step 3: Tokenize the Processed Corpus

Use the provided preprocessing script to convert text files into model-ready tokenized format:

```bash
python BiTimeBERT_Pretraining/preprocess.py \
  --trainpref <PATH>/0_corpus_train.txt \
  --validpref <PATH>/0_corpus_val.txt \
  --testpref <PATH>/0_corpus_test.txt \
  --destdir <OUTPUT_DIR> \
  --only-source \
  --srcdict dict.txt \
  --padding-factor 1 \
  --workers 12
```

### 📌 Parameter Explanation

| Argument | Description | Example |
|----------|-------------|---------|
| `--trainpref` | Path prefix for training corpus | `./data/0_corpus_train` |
| `--validpref` | Path prefix for validation corpus | `./data/0_corpus_val` |
| `--testpref` | Path prefix for test corpus | `./data/0_corpus_test` |
| `--destdir` | **Output directory** for tokenized binaries | `./binarized_corpus` |
| `--only-source` | Process source text only (no target) | *(flag, no value)* |
| `--srcdict` | Path to source vocabulary dictionary | `dict.txt` |
| `--padding-factor` | Memory alignment factor for efficiency | `1` |
| `--workers` | Number of parallel processing threads | `12` |

> 💡 **Tip**: Replace `<PATH>` with the actual path to your preprocessed files (e.g., `Pretraining_Preprocessing/2_Preprocessed_Data/2_corpus_titlecontent_withdocid/`), and `<OUTPUT_DIR>` with your desired output location.

---

## 🚀 Next Steps: Pretraining

Once tokenization is complete, you can start pretraining:

```bash
python 3_TempBERT_MaskTemp/train.py \
  --data <OUTPUT_DIR> \
  --save_dir <CHECKPOINT_DIR> \
  --docid2timestamp_dir <PATH>/0_docid2timestamp.pickle \
  --docid2sentidx2temptokenizeinfor_dict_file <PATH>/docid2sentidx2temptokenizeinfor_dict.pickle \
  --model_type mask_tempbert \
  --temp_granularity Month \
  --no_nsp \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1
```

📖 See `train.py --help` for a full list of training options.

---

> ℹ️ **Note**: All paths in this guide are relative to the project root unless specified as absolute. Adjust according to your local setup.

✅ **You're all set!** Once tokenization finishes, your data is ready for BiTimeBERT pretraining. 🎉
