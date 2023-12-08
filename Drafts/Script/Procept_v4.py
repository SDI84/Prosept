# Импорт необходимых библиотек
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import pandas as pd
from sentence_transformers import util
from nltk.corpus import stopwords
import re

# Загрузка стоп-слов для английского и русского языков
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
stop_words = stop_words_en.union(stop_words_ru)

# Загрузка токенизатора и модели LaBSE_ru_en (516 Мб)
tokenizer = AutoTokenizer.from_pretrained("cointegrated/LaBSE-en-ru")
model = AutoModel.from_pretrained("cointegrated/LaBSE-en-ru")

# Функция для очистки текста
def clean_texts(name):
    if not pd.isna(name):
        name = ' '.join(re.split(r"([A-Za-z][A-Za-z]*)", name))
        name = ' '.join(re.split(r"([0-9][0-9]*)", name))
        name = name.lower()
        name = re.sub(r"[^а-яa-z\d\s]+", ' ', name)
        name = re.sub(r"prosept", ' ', name)
        name = ' '.join([word for word in name.split() if word not in stop_words])
    else:
        name = ''
    return name

# Функция для обучения сокращенной модели LaBSE_ru_en
def t_fit_transformers(df, tokenizer, model, func=clean_texts, df_columns=['name']):
    # Подготовка текстов для обучения
    df_tmp = df[df_columns[0]].apply(func)
    if len(df_columns) > 1:
        for i in range(1, len(df_columns)):
            df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func)
    
    # Получение векторов из текстов
    inputs = tokenizer(df_tmp.tolist(), padding=True, truncation=True, max_length=96, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs)
    
    return model_output.last_hidden_state.mean(dim=1)

# Функция для предсказания схожести между дилерскими названиями и названиями производителя
def t_predict_transformers(dealer_names, product_embedding, tokenizer, model):
    # Получение векторов из названий дилеров
    dealer_embedding = model(**tokenizer(dealer_names.apply(clean_texts).tolist(), padding=True, truncation=True, max_length=96, return_tensors='pt')).last_hidden_state.mean(dim=1)
    
    # Оценка косинусной схожести
    return util.pytorch_cos_sim(dealer_embedding, product_embedding)

# Основная функция для предсказания
def prosept_predict(product, dealerprice) -> list:
    # Преобразование данных в датафреймы
    df_product = pd.read_json(product)
    df_dealerprice = pd.read_json(dealerprice)


    # Создание основного датафрейма для результатов
    df_res = df_dealerprice[['id', 'product_key', 'product_name']]
    df_dealerprice_unique = (df_dealerprice[['product_name']]
                             .drop_duplicates(subset='product_name')
                             .reset_index(drop=True))

    # Обучение модели
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']
    product_embedding_transformers = t_fit_transformers(df_product, tokenizer, model, clean_texts, columns)

    # Предсказание схожести
    df_predict_transformers = t_predict_transformers(df_dealerprice_unique['product_name'], product_embedding_transformers, tokenizer, model)

    # Получение топ-10 наилучших предсказаний
    N_BEST = 10
    quality, indices = df_predict_transformers.topk(N_BEST)

    # Добавление результатов в датафрейм
    df_dealerprice_unique.loc[:, 'predict'] = indices.tolist()
    df_dealerprice_unique.loc[:, 'quality'] = quality.tolist()
    df_res = df_res.merge(df_dealerprice_unique, how='left', on=['product_name'])
    df_res['queue'] = [[x for x in range(1, N_BEST+1)] for j in range(len(df_res))]
    df_res = df_res.explode(['predict', 'queue', 'quality'])
    df_res = df_res.reset_index(drop=True)

    # Добавление информации о продуктах
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict', axis=1)

    # Добавление временной метки
    df_res['create_date'] = datetime.now()

    # Преобразование результатов в JSON
    return df_res.to_json()