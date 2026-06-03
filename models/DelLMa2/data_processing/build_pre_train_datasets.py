import hashlib
import time
from pathlib import Path
import random

from shard_writer import ShardWriter

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

TRAIN_DIR = "DELLMA_2_DATA/TRAIN"
TOKENIZER_TARIN_DIR = "DELLMA_2_DATA/TOK_TRAIN"
VALIDATION_DIR = "DELLMA_2_DATA/VALIDATION"

DATASONE_DIR = "CLEANED_DATA/CLEANED_DATASONE"
OIMASIRUTEXT_DIR = "CLEANED_DATA/CLEANED_OIMASIRUTEXT"
DS_FILES = [
    f"{OIMASIRUTEXT_DIR}/cleaned_data_1.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_2.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_3.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_4.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_5.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_6.txt",
    f"{OIMASIRUTEXT_DIR}/cleaned_data_7.txt",
    # f"{DATASONE_DIR}/cleaned_articles_1.txt",
    # f"{DATASONE_DIR}/cleaned_articles_2.txt",
    # f"{DATASONE_DIR}/cleaned_articles_3.txt",
    # f"{DATASONE_DIR}/cleaned_articles_4.txt",
    # f"{DATASONE_DIR}/cleaned_articles_5.txt",
    # f"{DATASONE_DIR}/cleaned_articles_6.txt",
    # f"{DATASONE_DIR}/cleaned_articles_7.txt",
    # f"{DATASONE_DIR}/cleaned_articles_8.txt",
    # f"{DATASONE_DIR}/cleaned_articles_9.txt",
    # f"{DATASONE_DIR}/cleaned_articles_10.txt",
]


def _get_total_size():
    datasone = Path(DATASONE_DIR)
    oimasirutext = Path(OIMASIRUTEXT_DIR)
    datasone_size = sum(f.stat().st_size for f in datasone.iterdir() if f.is_file())
    oimasirutext_size = sum(f.stat().st_size for f in oimasirutext.iterdir() if f.is_file())
    total_size = datasone_size + oimasirutext_size
    return total_size


def main(
        val_ratio=0.05,
        tok_train_size_gb=0.5,
        max_shard_size_gb=1.0,
        start_tag="<startoftext>",
        end_tag="<endoftext>",
):
    train_writer = ShardWriter(TRAIN_DIR, "train", max_shard_size_gb)
    val_writer = ShardWriter(VALIDATION_DIR, "validation", max_shard_size_gb)
    tok_train_writer = ShardWriter(TOKENIZER_TARIN_DIR, "tok_train", max_shard_size_gb)
    total_size = _get_total_size()
    tok_train_ratio = (tok_train_size_gb * 1024 ** 3) / total_size

    seen_hashes = set()  # D5-хеши уникальных текстов
    current_doc_lines = []
    in_doc = False
    processed_bytes = 0
    progress = 0

    stats = {"train": 0, "val":0, "tokenizer_train":0, "duplicates": 0}

    start_time = time.time()
    for file in DS_FILES:
        with open(file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                progress += 1
                processed_bytes += len(line.encode("utf-8", errors="ignore"))

                if line == start_tag:
                    in_doc = True
                    current_doc_lines = []
                    continue

                if line == end_tag:
                    in_doc = False
                    block = "\n".join(current_doc_lines + [f"{end_tag}\n"])
                    block_hash = hashlib.md5(block[:3000].encode("utf-8")).hexdigest()

                    # Детектор Дубликатов
                    if block_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    seen_hashes.add(block_hash)

                    # Решаем куда отправлять блок - train или validation.
                    # Дополнительно решаем добавлять ли блок в датасет
                    # для обучения токенизатора
                    is_val = True if random.random() < val_ratio else False
                    is_tok_train = True if random.random() < tok_train_ratio else False
                    if is_val:
                        stats["val"] += 1
                        val_writer.write_doc(block)
                    else:
                        stats["train"] += 1
                        train_writer.write_doc(block)
                    if is_tok_train:
                        stats["tokenizer_train"] += 1
                        tok_train_writer.write_doc(block)
                    continue

                if in_doc:
                    current_doc_lines.append(line)

                if progress % 100_000 == 0:
                    t = time.time() - start_time
                    v = processed_bytes / t
                    print(
                        f"\rProgress: {(processed_bytes / total_size) * 100:.1f}% | "
                        f"Time: {t:.1f} s | "
                        f"Time Left: {(total_size - processed_bytes) / v:.2f} s | "
                        f"Speed: {v / (1024 ** 2):.2f} Mb/s |"
                        f"train: {stats['train']} | "
                        f"val: {stats['val']} | "
                        f"duplicates: {stats["duplicates"]} |"
                        f"tokenizer_train: {stats['tokenizer_train']} | ",
                        end="",
                        flush=True
                    )

    train_writer.flush()
    val_writer.flush()
    tok_train_writer.flush()

if __name__ == '__main__':
    main()
