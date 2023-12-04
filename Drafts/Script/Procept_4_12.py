# Импорт библиотек
import pandas as pd
import re

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def clean_texts(name):
    '''
    Функции очистки текста.
    Принимает на вход строку - название товара,
    возвращает его в отредактированном виде
    '''
    # стоп-слова для английского и русского языков
    stop_words_en = set(stopwords.words('english'))   # изменение тут
    stop_words_ru = set(stopwords.words('russian'))
    # объединим стоп-слова
    stop_words = stop_words_en.union(stop_words_ru)
    
    if not pd.isna(name):
        # разделение слов
        name = ' '.join(re.split(r"([A-Za-z][A-Za-z]*)", name))
        name = ' '.join(re.split(r"([0-9][0-9]*)", name))
        # нижний регистр
        name = name.lower()
        # удаление пунктуации
        name = re.sub(r"[^а-яa-z\d\s]+", ' ', name)        
        # удаление слова "prosept"
        name = re.sub(r"prosept", ' ', name)
        # удаление стоп-слов
        name = ' '.join([word for word in name.split() if word not in stop_words])
    
    else:
        name = ''
    
    return name 

def prosept_predict(product: dict, dealer: dict, dealerprice: dict) -> list:
    
    """
    Функция для предсказания n ближайших названий производителя 
    для каждого товара дилера.
    product - список товаров, которые производит и распространяет заказчик; 
    dealer - список дилеров;
    dealerprice - результат работы парсера площадок дилеров;
    """
    
    # Преобразование словарей в DataFrame
    df_product = pd.DataFrame.from_dict(product)
    df_dealerprice = pd.DataFrame.from_dict(dealerprice)
    df_dealer = pd.DataFrame.from_dict(dealer)  # изменение тут
    
    # Датафрейм df_res будет содержать рекомендации
    df_res = df_dealerprice[['id', 'product_key']]
    df_res.columns = ['id', 'key']
    
    # Загружаем предобученную модель LaBSE, при первом запуске потребуется загрузить 1.8 Гб данных
    model_LaBSE = SentenceTransformer('LaBSE')

    # В качестве исходного вектора продуктов используем все столбцы с наименованием продукции
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']

    # Функция подготовки данных для модели LaBSE
    def t_fit_LaBSE(df,func=clean_texts,df_columns=['name']):
        df_tmp = df[df_columns[0]].apply(func) # изменение тут
        if len(df_columns) > 1:
            for i in range(1, len(df_columns)):
                df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func) # изменение тут
        model = model_LaBSE.encode(df_tmp)
        return model, df[['id', 'name']]

    # Получаем вектора из датафрейма df_product
    product_embedding_LaBSE, df_product_LaBSE = t_fit_LaBSE(df_product,clean_texts,columns)

    # Функция предсказания на основе модели LaBSE
    def t_predict_LaBSE(dealer_names):
        dealer_embedding_LaBSE = model_LaBSE.encode(df_dealerprice['product_name'].apply(clean_texts))
        return util.pytorch_cos_sim(dealer_embedding_LaBSE, product_embedding_LaBSE) 

    # Получаем матрицу расстояний
    df_predict_LaBSE = t_predict_LaBSE(df_dealerprice['product_name'])

    # 10 индексов лучших совпадений для строк
    N_BEST = 10
    top_k_matches = []
    top_k_quality = []
    for i in range(df_predict_LaBSE.shape[0]):
        quality, indices = df_predict_LaBSE[i].topk(N_BEST)
        top_k_matches.append(indices.tolist())
        top_k_quality.append(quality.tolist())
    
    # Сохраним предсказания в df_res
    
    #def tens(res):    
        #return [res[s].item() for s in range(10)]

    df_res.loc[:, 'predict'] = top_k_matches
    #df_res['predict'] = df_res['predict'].apply(tens)
    df_res.loc[:, 'quality'] = top_k_quality
    #df_res['quality'] = df_res['quality'].apply(tens)
    df_res['queue'] = [[x for x in range(1, N_BEST + 1)] for j in range(len(df_res))] # изменение тут
    df_res = df_res.explode(['predict', 'queue', 'quality'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict', axis=1)
    df_res['create_date'] = datetime.now()

    #return df_res.to_json() 
    return df_res.to_dict()
