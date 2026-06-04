import os
import time
from enum import Enum

import unicodedata
from datasets import load_dataset
import re

from models.DelLMa2.data_processing.shard_writer import ShardWriter

DS_DIR = "../../../data/datasone_wiki"
CLEANED_DIR = "../../../data/CLEANED_DATA/wiki"
STATS_DIR = "../../../data/CLEANED_DATA/wiki/stats"


class ArticleType(Enum):
	""" Причины, по которым статья считается мусорной. """
	FUNDAMENTAL = "Хорошая статья для пре-трейна"
	# === Причины по названию (title) ===
	BIO = "Личности <Фамилия, Имя Отчество> или что-то имени чего-то"
	NAMED_AFTER = "имени, им."
	ROMAN_NUMS = "Римские цифры: <Гексанитрокобальтат(III) калия>, <Ричард III> и т.д."
	DATES = "Даты: <27 февраля>, <1654 год>"
	BAD_START_WORD = "Нежелательные слова в начале статьи: список, хронология и т.д."
	GEOGRAPHY = "Географические объекты"
	VALUES = "Название (значения)"
	SPORT = "Олимпийские игры, чемпионаты и тд."

	# === Причины по контенту (text) ===
	TOO_SHORT_RAW = "Статья изначально короче 3000 симв."
	TOO_SHORT_POST_CLEAN = "Статья стала короче 2000 симв. после обрезки хвостов"
	# Конкретные маркеры мусора в теле статьи
	GEOPOLITICS = "Геополитический маркер (государство, суверенный...)"
	BIO_CARD = "Биографическая карточка (писатель, царь, даты жизни...)"
	POLITICS_MILITARY = "Политики, чиновники, военный шум"
	INFRASTRUCTURE = "Статичные объекты (собор, музей, завод...)"
	GEOGRAPHY_CONTENT = "Мелкий мусор (село, река, переменная звезда...)"
	ASTRONOMY = "Звёзды, созвездия"


# -------------------------------------------------------------------------------------------------------- #
#                                        Фильтрация названий статей                                        #
# -------------------------------------------------------------------------------------------------------- #


# Все биографии людей в Википедии называются по схеме Фамилия, Имя Отчество.
# Например: «Бродский, Иосиф Александрович», «Меир, Голда».
# Если в названии есть запятая, а после неё пробел и Большая буква — это 100% человек.
RE_BIO = re.compile(r'^[А-ЯЁа-яё\-\'’]+\s*,\s+[А-ЯЁ]')  # "Фамилия, Имя" или что-то имени кого-то
RE_NAMED_AFTER = re.compile(r"\s+им\.|\sимени\s", re.IGNORECASE)
RE_ROMAN_NUMS = re.compile(r'\b(?!(?:ML|DL)\b)[IVXLCDM]+\b')  # Наличие римских цифр (Петр I, Людовик XIV) (Кроме ML/DL)
RE_DATES = re.compile(r"\b\d+\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)|"
                      r"(\b\d+(-й|-го|-е)?\s+год)", re.IGNORECASE)
RE_BAD_START_WORDS = re.compile(  # Нежелательные стартовые слова
	r"^(список|премия|хронология|история|география|население|экономика|культура|политика|санкции|герб|"
	r"флаг|гимн|календарь|словарь|атлас|фильмография|улица|район|районы|единая россия|заслуженный|награды)\s+",
	re.IGNORECASE)
RE_GEOGRAPHY = re.compile(
	r"я область|сельсовет|сельские советы|поселение|департамент|улица|переулок|"
	r"памятник|\(станция\)|\(станция метро\)|\(округ\)|административный округ|автономный округ|"
	r"\(аэропорт\)|\(регион\)|\(провинция\)|\(коммуна\)|\(остров\)|(?<![А-Яа-яёЁ])парк(?![а-яёА-ЯЁ])",
	# Никольск (Вологодская область), Горно-Бадахшанская автономная область
	re.IGNORECASE
)
RE_VALUES = re.compile(r"\(значения\)|\(фамилия\)|^\.")  # Список значений слова, фамилии, домены
RE_SPORT = re.compile(r"чемпион|олимпийск|футбол|хоккей|кёрлинг|волейбол|теннис", re.IGNORECASE)
RE_MILITARY = re.compile(
	r"батальон|(?<![А-Яа-яёЁ])полк(?![а-яёА-ЯЁ])|дивизия|армия|(?<![А-Яа-яёЁ])крейсер(?![а-яёА-ЯЁ])", re.IGNORECASE)
RE_POLITICS = re.compile(r"выборы", re.IGNORECASE)
RE_ASTRONOMY = re.compile(r"^NGC\s\d+")

TITLE_RULES_MAPPING = {
	ArticleType.BIO: RE_BIO,
	ArticleType.NAMED_AFTER: RE_NAMED_AFTER,
	ArticleType.ROMAN_NUMS: RE_ROMAN_NUMS,
	ArticleType.DATES: RE_DATES,
	ArticleType.BAD_START_WORD: RE_BAD_START_WORDS,
	ArticleType.GEOGRAPHY: RE_GEOGRAPHY,
	ArticleType.VALUES: RE_VALUES,
	ArticleType.SPORT: RE_SPORT,
	ArticleType.POLITICS_MILITARY: RE_MILITARY,
	ArticleType.GEOPOLITICS: RE_POLITICS,
	ArticleType.ASTRONOMY: RE_ASTRONOMY,
}
TITLE_RULES = [(pattern, reason) for reason, pattern in TITLE_RULES_MAPPING.items()]
# TITLE_RULES = [
# 	(RE_BIO, ArticleType.BIO),
# 	(RE_ROMAN_NUMS, ArticleType.ROMAN_NUMS),
# 	(RE_DATES, ArticleType.DATES),
# 	(RE_BAD_START_WORDS, ArticleType.BAD_START_WORD),
# 	(RE_GEOGRAPHY, ArticleType.GEOGRAPHY),
# 	(RE_VALUES, ArticleType.VALUES),
# ]

# -------------------------------------------------------------------------------------------------------- #
#                                        Фильтрация содержания статей                                      #
# -------------------------------------------------------------------------------------------------------- #


TEXT_RULES_MAPPING = {
	# ArticleType.GEOPOLITICS: [
	# 	# 1. Страны, материки, геополитика и территории
	# 	r"[—–\-] (государство|островное государство|суверенное государство|королевство|"
	# 	r"эмират|материк|континент|административный центр|муниципальное образование|муниципальный район)(?![а-яёА-ЯЁ])",
	# 	r"официальное название [—–\-] (Респу́блика|Республика|Короле́вство|Королевство|Госуда́рство|Государство)",
	# 	r"в (Юго-Восточной|Западной|Центральной|Южной|Восточной|Северной|Евроазиатской) (Африке|Азии|Америке|Европе)",
	# 	r"омывается\s+[а-яёА-ЯЁ]+\s+морем",
	# 	r"на (зимних|летних)?\s*Олимпийских играх",
	# ],
	ArticleType.GEOPOLITICS: [
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}"
		r"(?:административный центр|муниципальное образование|муниципальный район"
        r"|сельское поселение|городское поселение|городской округ|уезд|департамент|коммуна)"
		r"(?![а-яёА-ЯЁ])",
        r"на (зимних|летних)?\s*Олимпийских играх",
	],
	ArticleType.BIO_CARD: [
		# 2. Личности, правители, философы, писатели (Выжигаем био-карточки)
		# Используем \b(философ)\b, чтобы "философия" или "философский" НЕ триггерились
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}(?:философ|врач|писатель|поэт|историк|учёный|ученый|мыслитель|космолог|мифограф"
		r"|исследователь|художник|художница|архиерей|епископ|живописец|архитектор|музыкант|актёр|актер|актриса|певец|"
		r"певица|дирижёр|гравёр|спортсмен|футболист|хоккеист|теннисист|боксёр|шахматист"
		r"скульптор|композитор|режиссёр|режиссер)(?![А-Яа-яёЁ])",
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}(?:царь|король|император|императрица|принцесса|герцог|герцогиня|князь|княгиня|магистр|султан|хан|шах)(?![А-Яа-яёЁ])",
		r"из династии (Романовых|Рюриковичей|Бурбонов|Габсбургов|династии)",
		r"\((ок\.\s*)?\d+\s+(до н\. э\.|н\. э\.)?[\s\S]*?[—–\-]\s*(ок\.\s*)?\d+",
		# Даты жизни: (ок. 315 до н. э. — ок. 240 до н. э.)
		r"(?<![А-Яа-яёЁ])(род\.|родился|родилась|умер|умерла)\s+\d+",  # Даты рождения и смерти
	],
	ArticleType.POLITICS_MILITARY: [
		# 3. Политики, чиновники, военные (Весь ура-патриотический и госслужебный шум)
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}(?:полководец|генерал|адмирал|маршал|полковник|командир|офицер|военачальник|боец|бригадир)(?![А-Яа-яёЁ])",
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}(?:президент|премьер-министр|министр|депутат|революционер|политик|государственный деятель|дипломат|предприниматель|бизнесмен)(?![А-Яа-яёЁ])",
		r"(?:—|–|-) (воинское соединение|дивизия|полк|бригада|истребительный авиационный полк|эскадрилья)(?![А-Яа-яёЁ])",
		r"(руководителей СССР|первых лиц|коммунистической партии|выборы президента)",
	],
	ArticleType.INFRASTRUCTURE: [
		# 4. Постройки, музеи, соборы, замки (Статичные объекты инфраструктуры)
		r"(?:—|–|-)\s+(?:[а-яёА-ЯЁ\-]+[,\s]+){0,3}(?:собор|храм|замок|дворец|музей|кабинет|памятник архитектуры|монастырь|церковь|синагога|разрез|предприятие|завод|фабрика)(?![А-Яа-яёЁ])",
		r"(?<![А-Яа-яёЁ])(кафедральный собор|музей)(?![А-Яа-яёЁ])",
	],
	ArticleType.GEOGRAPHY_CONTENT: [
		# 5. Базовый мусор (села, реки, переменные звезды из каталогов)
		r"(?:—|–|-) (село|деревня|посёлок|поселок|коммуна|река|протока|город|озеро|остров|мыс|округ"
		r"|уезд|район|альбом|сингл|саундтрек|композиция|трек|персонаж)(?![А-Яа-яёЁ])",

	],
	ArticleType.ASTRONOMY: [
		r"(?:—|–|-) (одиночная|переменная|двойная)?\s*звезда в созвездии"
	],
}

# Компилируем один раз при старте модуля
TEXT_RULES = [
	(re.compile(pattern, re.IGNORECASE), reason)
	for reason, patterns in TEXT_RULES_MAPPING.items()
	for pattern in patterns
]

# -------------------------------------------------------------------------------------------------------- #
#                                        Удаление ненужных разделов                                        #
# -------------------------------------------------------------------------------------------------------- #


TRASH_SECTION_PATTERNS = [
	# 1. COMMON
	r"^\s*См\. также",
	# r"^Примечания\s*(?:к таблице|и ссылки|:|\.|$)",
	r"^\s*(?:примечания|ссылки|литература|источники|список|внешние|основные|дополнительная|дополнительные|использованная|использованные|рекомендуемая|рекомендуемые|библиографический)\s*"
	r"(?:и|на|к|по|для|о|об|из)?\s*"
	r"(?:источники|примечания|разделу|разделы|теме|публикации|ссылки|литература|литературы|таблице|сноски|сносок|комментарии|комментариев|списку|внешние ресурсы)?\s*"
	r"(?:$|\.|:|;)",

	# 2. Фильмы
	r"^\s*В ролях\s*(?:$|\.|:|;)",
	r"^\s*Создатели\s*(?:$|\.|:|;)",
	r"^\s*Съёмочная группа\s*(?:$|\.|:|;)",
	r"^\s*Награды\s*(?:$|\.|:|;)",
	r"^\s*Премии\s*(?:$|\.|:|;)",
	r"^\s*Номинации\s*(?:$|\.|:|;)",
	r"^\s*Награды и номинации\s*(?:$|\.|:|;)",
	r"^\s*Критика\s*(?:$|\.|:|;)",
	r"^\s*Саундтрек\s*(?:$|\.|:|;)",
	r"^\s*Кассовые сборы\s*(?:$|\.|:|;)",
	r"^\s*Экранизации\s*(?:$|\.|:|;)",

	# 3. Страны, прочее
	r"^\s*Галерея\s*(?:$|\.|:|;)",
	r"^\s*Демография\s*(?:$|\.|:|;)",
]

# Собираем всё в одну мега-регулярку для дикой скорости
# Флаг re.MULTILINE заставляет ^ триггериться на новую строку внутри текста
RE_TRUNCATE_SECTIONS = re.compile("|".join(TRASH_SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


# -------------------------------------------------------------------------------------------------------- #
#                                      Создание чистого датасета                                           #
# -------------------------------------------------------------------------------------------------------- #

def _is_fundamental(
		title: str,
		text: str = "",
		min_article_length: int = 3000,
		check_first_k: int = 500,
) -> tuple[bool, ArticleType]:
	"""
	Фундаментальная ли статья?
		- title is not None, text = None -> Проверка фундаментальности названия статьи.
		- title = None, text is not None -> Проверка фундаментальности текста статьи.
		- title & text is not None -> Проверка на фундаментальность названия и текста статьи вместе.
	:param min_article_length: Ограничение снизу на число символов в фундаментальной статье.
	Если 0, то ограничений по размеру статьи снизу нет.
	:param check_first_k: Сколько первых символов статьи проверять.
	:return: Фундаментальная ли статья и тип мусора. Если статья фундаментальна - тип мусора - None
	"""
	# Проверка названия
	if title:
		for pattern, reason in TITLE_RULES:
			if pattern.search(title):
				return False, reason
	# if RE_BIO.match(title): return False, ArticleType.BIO
	# if RE_ROMAN_NUMS.search(title): return False, ArticleType.ROMAN_NUMS
	# if RE_DATES.search(title): return False, ArticleType.DATES
	# if RE_BAD_START_WORDS.match(title): return False, ArticleType.BAD_START_WORD
	# if RE_GEOGRAPHY.search(title): return False, ArticleType.GEOGRAPHY
	# if RE_VALUES.search(title): return False, ArticleType.VALUES

	# Проверка длины статьи
	if text:
		if min_article_length > 0:
			if len(text) < min_article_length:
				return False, ArticleType.TOO_SHORT_RAW

		# Проверяем первые check_first_k символов. Там всегда сидит определение статьи
		definition_zone = text[:check_first_k]

		for pattern, reason in TEXT_RULES:
			if pattern.search(definition_zone):
				return False, reason  # Нашли мусорный маркер -> выкидываем статью

	return True, ArticleType.FUNDAMENTAL  # Статья прошла все круги ада, она чистая


def remove_all_wiki_templates(text: str) -> str:
	# Ищем {{, внутри которого НЕТ других { или } и закрываем через }}
	pattern = re.compile(r'\{\{[^{}]*}}')

	while True:
		text, count = pattern.subn('', text)
		if count == 0:  # Если за проход ничего не удалилось — мы вычистили всё
			break

	return text


def _truncate_sections(text: str) -> str:
	if not text:
		return ""

	# Ищем первое вхождение любого мусорного заголовка, стоящего на новой строке
	match = RE_TRUNCATE_SECTIONS.search(text)

	if match:
		# Отрезаем всё, что идет начиная с индекса старта этого заголовка
		return text[:match.start()].strip()

	return text.strip()


def _normalize_text(text: str):
	# Замена кривых переносов строк (LS и PS / категории Zl и Zp) на стандартный \n
	text = re.sub(r'[\u2028\u2029]', '\n', text)    # Кривые переносы строк
	text = re.sub(r'[\u200b\u200e\u200f\u202a-\u202e]', '', text)   # Категории Cc и Cf
	text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\x1a]', '', text)   # Символы (Cc), кроме \n и \t
	text = unicodedata.normalize('NFD', text)  # Разбиваем все составные буквы (и латинские é, и кириллические а́)
	text = text.replace('\u0301', '')  # Вырезаем само ударение
	text = unicodedata.normalize('NFKC', text)  # Возвращаемся обратно
	text = re.sub(r'\s*\(\s*\)', '', text)  # Удаление пустых скобок
	text = re.sub(r'\[.*?]', '', text)  # Удаление квадратных скобок и всего, что внутри них
	# приводим ВСЕ виды длинных, средних и кривых тире из Юникода к ОДНОМУ эталонному длинному тире (—)
	# При этом обычный дефис (-) мы здесь НЕ трогаем, он остается дефисом.
	text = re.sub(r"[–‒⎼⁃⎯⌲⏤▕─⸺⸻⸼―]", "—", text)

	# Убираем пробельный мусор вокруг тире, чтобы модель не путалась в их количестве,
	# но сохраняем структуру (например, пробел-тире-пробел для знака препинания)
	text = re.sub(r' +— +', ' — ', text)

	# ^[ \t]+ ловит любые пробелы и табуляции в начале КАЖДОЙ строки строки благодаря re.M
	text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
	text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE) # пробелы в КОНЦЕ строк

	text = re.sub(r'[ \t]+', ' ', text)  # Схлопываем множественные пробелы в один
	text = re.sub(r'\n\s*\n+', '\n', text)  # Удаление множественных переносов строк
	return text.strip()


def _inspect_filtered_dataset(dataset, interval: tuple):
	print("Загрузка датасета...")

	print(f"Всего статей в сыром дампе: {len(dataset)}")
	print("Начинаем стриминг и фильтрацию. Выводим качественные статьи:\n" + "=" * 60)

	printed_count = 0
	scanned_count = 0

	# Идем по датасету линейно
	for item in dataset.select(range(interval[0], interval[1])):
		scanned_count += 1
		text = item['text']
		title = item['title']

		if _is_fundamental(title, text):
			printed_count += 1
			# print(f" СЕМПЛ #{scanned_count} (Длина: {len(text)} симв.)")
			# Если есть колонка title, выведем её

			print(f"|{title}| #{scanned_count} (Длина: {len(text)} симв.)")

			# print("-" * 40)
			# Принтим первые 1200 символов статьи для оценки глубины контента
			print(text)
			print("\n")

	print(f"Сканирование завершено. Проверено статей: {scanned_count}. Найдено годноты: {printed_count}")


def _get_article_titles(
		dataset,
		fundamental_path,
		garbage_path,
		min_article_length=500,
		check_first_k=300,
		size_step=1000,
):
	""" Генерация файла со всеми названиями статей датасета """
	print(f"Всего статей в сыром дампе: {len(dataset)}")

	stats = {
		"total_articles": len(dataset),
		# "FUNDAMENTAL": 0,
		# "BIO": 0,
		# "NAMED_AFTER":0,
		# "ROMAN_NUMS": 0,
		# "DATES": 0,
		# "BAD_START_WORD": 0,
		# "GEOGRAPHY": 0,
		# "VALUES": 0,
		# "SPORT": 0,
		# "TOO_SHORT_RAW": 0,
		# "TOO_SHORT_POST_CLEAN": 0,
		# "GEOPOLITICS": 0,
		# "BIO_CARD": 0,
		# "POLITICS_MILITARY": 0,
		# "INFRASTRUCTURE": 0,
		# "GEOGRAPHY_CONTENT": 0,
		# "ASTRONOMY": 0,
	}
	sizes = {}

	def save_stats(key):
		if key in stats:
			stats[key] += 1
		else:
			stats[key] = 1

	def save_size(text_len):
		# Вычисляем нужный диапазон (округляем вверх до ближайшего шага)
		k = ((text_len + size_step - 1) // size_step) * size_step

		if k in sizes:
			sizes[k] += 1
		else:
			sizes[k] = 1

	with open(fundamental_path, "w", encoding="utf-8") as f_fun, open(garbage_path, "w", encoding="utf-8") as f_garb:
		start_time = time.time()
		progress = 0
		for item in dataset:
			title = item['title']
			text = item['text']
			text = remove_all_wiki_templates(text)
			is_fundamental, trash_type = _is_fundamental(
				title=title,
				text=text,
				min_article_length=min_article_length,
				check_first_k=check_first_k,
			)
			if is_fundamental:
				f_fun.write(f"{title} - [{len(text)}]" + "\n")
				# stats["FUNDAMENTAL"] += 1
				save_stats("FUNDAMENTAL")
				save_size(len(text))
			else:
				f_garb.write(f"{title} - {trash_type.name} - [{len(text)}]" + "\n")
				save_stats(trash_type.name)
			# 				stats[trash_type.name] += 1

			progress += 1
			if progress % 100_000 == 0:
				t = time.time() - start_time
				v = progress / t
				print(f"\rProgress: {progress / len(dataset) * 100:.2f}% | "
				      f"Time: {t:.2f} s | "
				      f"Time Left: {(len(dataset) - progress) / v:.2f} s | "
				      f"Speed: {v:.2f} Articles / s | {stats}", end="", flush=True)
		print()
		print(f"Time: {time.time() - start_time:.2f}s")

		print(f"Всего фундаментальных названий: {stats["FUNDAMENTAL"]}")
		print("STATS")
		print(*sorted(stats.items(), key=lambda x: x[1], reverse=True), sep='\n')
		print("SIZES")
		print(*sorted(sizes.items(), key=lambda x: x[0]), sep='\n')


def print_articles_by_titles(dataset, titles: list[str], first_k=500, clean: bool = False):
	for item in dataset:
		if item["title"] in titles:
			title = item["title"]
			text = item["text"][:first_k]
			if clean:
				text = _truncate_sections(text)
				text = _normalize_text(text)
			print(f"============================================ {title} ============================================")
			print(text)
			print()


def main(
		dataset,
		output_dir,
		# stats_dir,
		start_tag="<startoftext>",
		end_tag="<endoftext>",
		max_shard_size_gb=1.0,
		check_first_k=1000,
		min_dirty_article_length=500,
		min_clean_article_length=400,
		size_step=1000
):
	"""
	Пайплайн фильтрации, очистки и нарезки датасета на txt шарды
	"""
	os.makedirs(f"{output_dir}/stats", exist_ok=True)
	shard_writer = ShardWriter(output_dir, "wiki_articles", max_shard_size_gb)
	total_size = dataset.size_in_bytes
	print(total_size / (1024**3))

	total_articles = len(dataset)
	processed_bytes = 0
	progress = 0
	saved_count = 0

	f_titles = open(f"{output_dir}/stats/titles.txt", "w", encoding="utf-8")

	stats = {"total_articles": len(dataset), }
	sizes = {}

	start_time = time.time()

	def save_stats(key):
		if key in stats:
			stats[key] += 1
		else:
			stats[key] = 1

	def save_size(text_len):
		# Вычисляем нужный диапазон (округляем вверх до ближайшего шага)
		k = ((text_len + size_step - 1) // size_step) * size_step

		if k in sizes:
			sizes[k] += 1
		else:
			sizes[k] = 1

	# Формируем шарды
	print(f"Запуск пайплайна. Целевой размер шарда: {max_shard_size_gb} ГБ")
	for item in dataset:
		progress += 1
		title = item["title"]
		text = item["text"]
		processed_bytes += len(text.encode('utf-8', errors='ignore'))
		text = remove_all_wiki_templates(text)
		# Проверка на фундаментальность
		is_fundamental, trash_type = _is_fundamental(
			title=title,
			text=text,
			min_article_length=min_dirty_article_length,
			check_first_k=check_first_k)


		if not is_fundamental:
			save_stats(trash_type.name)
			f_titles.write(f"{title} [{trash_type.name}] [{len(text)}]\n")
			continue


		# Отрезаем мусорные разделы и нормализуем текст
		text = _truncate_sections(text)
		text = _normalize_text(text)


		# На всякий случай проверяем длину после обрезки хвостов
		if len(text) < min_clean_article_length:
			save_stats("TOO_SHORT_POST_CLEAN")
			f_titles.write(f"{title} [TOO_SHORT_POST_CLEAN] [{len(text)}]\n")
			continue

		# Формируем финальный блок для претрейна
		save_stats("FUNDAMENTAL")
		save_size(len(text))
		block = f"{start_tag}\n{title}\n{text}\n{end_tag}\n\n"


		shard_writer.write_doc(block)
		saved_count += 1

		if progress % 10_000 == 0:
			t = time.time() - start_time
			v_docs = progress / t
			v_mb = processed_bytes / t
			print(
				f"\rProgress: {progress / total_articles * 100:.2f}% | "
				f"Time: {t:.2f} s | "
				f"Time Left: {(total_articles - progress) / v_docs:.2f} s | "
				f"Speed: {v_mb / (1024 ** 2):.2f} Mb/s ({v_docs:.1f} art/s)",
				end="",
				flush=True
			)

	# Не забываем закрыть последний файл
	shard_writer.flush()
	f_titles.flush()
	f_titles.close()

	# Пишем статистику в файлы
	with open(f"{output_dir}/stats/stats.txt", "w", encoding="utf-8") as f_stats, \
			open(f"{output_dir}/stats/fundamental_sizes.txt", "w", encoding="utf-8") as f_sizes:
		for name, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
			f_stats.write(f"{name}: {count}\n")
		for symbols, count in sorted(sizes.items(), key=lambda x: x[0]):
			f_sizes.write(f"{symbols}: {count}\n")

	print("\n" + "=" * 60)
	print(f"Пайплайн успешно завершен за {time.time() - start_time:.1f} сек!")
	# print(f"Всего проверено статей: {processed_count}")
	print(f"Всего сохранено в чистый датасет: {saved_count}")
	print(f"Итоговое количество шардов: {shard_writer.shard_index}")


if __name__ == "__main__":
	# Запускаем просмотр (выведет первые 10 статей, прошедших фильтр)
	# inspect_filtered_dataset(HF_DATASET_PATH, max_to_print=1000)

	ds = load_dataset(
		"arrow",
		data_files=os.path.join(DS_DIR, "wikipedia_ru-train-*.arrow"),
		split="train"
	)

	main(
		dataset=ds,
		output_dir=CLEANED_DIR
	)

	# _get_article_titles(
	# 	ds,
	# 	"fundamental_wiki_article_titles.txt",
	# 	"garbage_wiki_article_titles.txt",
	# 	min_article_length=500,
	# 	check_first_k=1000,
	# )

# print_articles_by_titles(
# 	ds,
# 	[
# 		"Советско-германские соглашения (1939—1941)",
# 		"Великая Отечественная война",
# 		"Украинцы в США",
# 		"Ногай давысы",
# 		"Венгрия на зимних Олимпийских играх 2010",
# 		"Украинское сельское поселение",
# 		"9 Июля (Санта-Фе)",
# 		"Ваханский хребет",
# 		"Парламентские выборы в Латвии(2006)",
# 		"Старый Крым (санаторий)",
# 		"Американская чёрная кряква",
# 		"Франческо Сальвиати",
# 		"Арвальские братья",
# 		"147-я стрелковая дивизия (2-го формирования)",
# 		"Площадь Тургенева",
# 		"Мясная улица",
# 		"Агасси(тауншип, Миннесота)",
# 		"Агдер(тауншип, Миннесота)",
# 	],
# 	clean=False,
# 	first_k=300
# )
# print_articles_by_titles(
# 	ds,
# 	[
# 		".rw",
# 		".sa",
# 		".sc",
# 		".sd",
# 		".se",
# 		".sg",
# 		".sh",
# 		".si",
# 		".sj",
# 		".sk",
# 		".sl",
# 		".sm",
# 		".sn",
# 		".so",
# 		".sr",
# 		".st",
# 		".sv",
# 	],
# 	clean=True,
# 	first_k=1_000_000
# )
