# Рабочая тетрадь основного варианта.
1. Для файла Procept_tfidf.py тетрадь [Procept_final_accuracy](https://github.com/SDI84/Prosept/blob/main/Drafts/Prosept_final_accuracy) - проверка метрик финального скрипта, использующего для векторизации названий TF-IDF, была изменена функция предварительной очистки названий. Accuracy@5:0.9321 Accuracy@1:0.7397 MRR:0.8164  

## Черновики

Здесь собраны тетради с разными вариантами подготовки данных/признаков. Оставили только рабочие варианты.
1. Тетрадь [Prosept_TF-IDF_LaBSE](Prosept_TF-IDF_LaBSE.ipynb) c EDA и предобработкой данных, разными вариантами очистки текста, анализом разных параметров векторайзера и использования разных имен в качестве названий заказчика.
2. Тетрадь [Prosept_CatBoost](Procept_LaBSE_CatBoost.ipynb), проверили влияние на точность предсказаний товара заказчика добавления к вычислению ближайших k названий реранжирования с помощью обучения классификатора на k ближайших именах и выбора 5 (или другого указанного числа) лучших на основании предсказания модели.
3. Тетрадь [Procept_TF-IDF_LaBSE_4_12](https://github.com/SDI84/Prosept/blob/main/Drafts/Procept_TF-IDF_LaBSE_4_12.ipynb) - добавлена новая функция очистки текста clean_texts, тестирование модели LaBSE и clean_texts, улучшены итоговые результаты метрик: Accuracy@5 = 0.9403, 	Accuracy@1 = 0.7457, 	MRR = 0.8272
5. Тетрадь [Procept_TF-IDF_LaBSE_7_12.ipynb](https://github.com/SDI84/Prosept/blob/main/Drafts/Procept_TF-IDF_LaBSE_7_12.ipynb) - добавлено тестирование модели LaBSE-en-ru (clean_texts, max_length=61) - сократилась скорость работы, итоговые результаты метрик: Accuracy@5 = 0.9403, 	Accuracy@1 = 0.7546, 	MRR = 	0.8324
6. Для файла procept_labse_small.py тетрадь [Procept_LaBSE2.ipynb](https://github.com/SDI84/Prosept/blob/main/Drafts/Procept_LaBSE2.ipynb) - тестирование модели LaBSE-en-ru (clean_texts, max_length=61) - сократилась скорость работы, итоговые результаты метрик: Accuracy@5 = 0.947, 	Accuracy@1 = 0.7508, 	MRR = 	0.839
