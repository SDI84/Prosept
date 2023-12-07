import re
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_texts(name):
    '''
    Функции очистки текста.
    Принимает на вход строку - название товара,
    возвращает его в отредактированном виде
    '''     
    if not pd.isna(name):
        # отделение цифр от слов
        name = ' '.join(re.split(r"([0-9][0-9]*)", name))
        # нижний регистр
        name = name.lower()
        # удаление пунктуации
        name = re.sub(r"[^а-яa-z\d\s]+", ' ', name)        
        # удаление слов "prosept, просепт и professional"
        name = re.sub(r"prosept|просепт|professional", ' ', name)
    else:
        name = ''    
    return name

def prosept_predict(product:list, dealerprice:list) -> list:
    
    """
    Функция для предсказания n ближайших названий производителя 
    для каждого товара дилера.
    product - список товаров, которые производит и распространяет заказчик; 
    dealerprice - результат работы парсера площадок дилеров;
    """
    
    # Преобразование словарей в DataFrame
    df_product = pd.DataFrame.from_dict(product)
    df_dealerprice = pd.DataFrame.from_dict(dealerprice)

    #Датафрейм df_res будет содержать рекомендации
    df_res = df_dealerprice[['id', 'product_key']]
    
    # В качестве исходного вектора продуктов используем все столбцы с наименованием продукции
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']
    
    #  TF-IDF
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2), max_df=0.6)

    def t_fit_tfidf(df,func=clean_texts,df_columns=['name']):

        df_tmp = df[df_columns[0]].apply(func)
        if len(df_columns)>1:
            for i in range(1,len(df_columns)):
                df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func)

        model = vectorizer.fit_transform(df_tmp)

        return model, df[['id','name']]

    # Получаем вектора из датафрейма df_product
    product_vec, df_product_tfidf =  t_fit_tfidf(df_product,clean_texts,columns)

    # Функция предсказания
    def t_predict_tfidf(dealer_names, product_vec=product_vec):
        dealer_vec = vectorizer.transform(dealer_names.apply(clean_texts))
        return cosine_similarity(dealer_vec, product_vec) 

    #Получаем матрицу расстояний
    df_predict_tfidf = t_predict_tfidf(df_dealerprice['product_name'], product_vec)
    
    # 10 индексов лучших совпадений для строк
    N_BEST = 10
    indices =  df_predict_tfidf.argsort()[:, -N_BEST:][:, ::-1]
    quality = np.take_along_axis(df_predict_tfidf, indices, axis=1)
    
    # Сохраним предсказания в df_res
    df_res.loc[:,'predict'] = indices.tolist()
    df_res.loc[:,'quality'] = quality.tolist()
    df_res['queue'] = [list(range(1, N_BEST+1)) for _ in range(len(df_res))]
    df_res=df_res.explode(['predict', 'quality', 'queue'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict',axis=1)
    df_res['create_date'] = datetime.now()

     # Результат в JSON
    result_json = df_res.to_json(orient='records')

    return json.loads(result_json)