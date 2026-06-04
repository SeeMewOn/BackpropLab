import os
import time
from pathlib import Path
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    trainers,
)


TOK_TRAIN_DIR = "../../../data/DELLMA_2_DATA/tok_train"
SAVE_DIR = "saved_tokenizer"

VOCAB_SIZE = 16000

SPECIAL_TOKENS = ["<|pad|>", "<|endoftext|>", "<|user|>", "<|assistant|>"]

def get_training_corpus(batch_size=1000):
	path = Path(TOK_TRAIN_DIR)
	current_batch = []
	shard_files = sorted(list(path.glob("*.txt")))
	print(f"Найдено {len(shard_files)} шардов для обучения токенизатора.")

	for file_path in shard_files:
		with open(file_path, "r", encoding="utf-8") as f:
			content = f.read()
			docs = content.split("<endoftext>\n")

			for doc in docs:
				doc = doc.strip()
				if not doc:
					continue

				# Поскольку .split() съел наш тег конца текста,
				# мы возвращаем его на место, чтобы токенизатор видел структуру
				full_text = f"{doc}\n<|endoftext|>"
				current_batch.append(full_text)

				# Отдаем батч, когда он накопился
				if len(current_batch) == batch_size:
					yield current_batch
					current_batch = []

	# Отдаем остатки
	if current_batch:
		yield current_batch

if __name__ == '__main__':
	tokenizer = Tokenizer(model=models.BPE())

	# Собираем последовательность пре-токенизаторов.
	# Первым шагом Split находит ВСЕ переносы строк и изолирует их (behavior="isolated").
	# Граница изоляции гарантирует, что BPE никогда не сможет объединить \n с чем-либо еще.
	# Вторым шагом ByteLevel превращает всё в байты, как и раньше.
	tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
	    pre_tokenizers.Split(pattern="\n", behavior="isolated"),
	    pre_tokenizers.ByteLevel(add_prefix_space=False)
	])

	tokenizer.decoder = decoders.ByteLevel()
	trainer = trainers.BpeTrainer(
	    vocab_size=VOCAB_SIZE,
	    special_tokens=SPECIAL_TOKENS,
	    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
	)

	print("Начало обучения токенизатора на очищенных шардах...")
	start_time = time.time()

	tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

	print(f"Обучение завершено! Заняло: {int(time.time() - start_time)} сек.")

	# 6. Сохраняем результат
	os.makedirs(SAVE_DIR, exist_ok=True)
	tokenizer_path = os.path.join(SAVE_DIR, "tokenizer_delLMa2.json")
	tokenizer.save(tokenizer_path)
	print(f"Токенизатор успешно сохранен в '{tokenizer_path}'")