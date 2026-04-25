import numpy as np
from numpy.lib.stride_tricks import as_strided
import time
from PIL import Image

start = 0
stop = 12400
# for i in range(start, stop):
# 	try:
# 		Image.open(f"archive/PetImages/Cat/{i}.jpg").convert("RGB")
# 	except:
# 		print(f"archive/PetImages/Cat/{i}.jpg")


print()

for j in range(start, stop):
	try:
		# img = Image.open(f"archive/PetImages/Dog/{j}.jpg")
		# img.verify()
		with Image.open(f"data/PetImages/Dog/{j}.jpg") as img:
			img.verify()  # Проверка целостности структуры файла

		# Дополнительная проверка: попробуем реально прочитать пиксели
		# Иногда verify проходит, а load() падает на битых данных
		with Image.open(f"data/PetImages/Dog/{j}.jpg") as img:
			img.load()
	except(UserWarning):
		print(f"archive/PetImages/Dog/{j}.jpg")


	# X.append(im2tensor(f"archive/PetImages/Dog/{j}.jpg", target_size))
	# print(f"\rПёсель #{j}", end="")