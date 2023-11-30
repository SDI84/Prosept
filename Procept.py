#Импорт библиотек
import pandas as pd
import re

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


def prosept_predict(product: dict,dealer: dict,dealerprice: dict,productdealerkey: dict)->dict:
    
    """
    product - список товаров, которые производит и распространяет заказчик; 
    dealer - список дилеров;
    dealerprice - результат работы парсера площадок дилеров;
    productdealerkey - таблица матчинга товаров заказчика и товаров дилеров;
    """
    
    #Преобразуем входные данные в DataFrame
    df_dealer = pd.read_json(dealer)
    df_dealerprice = pd.read_json(dealerprice)
    df_product = pd.read_json(product)
    df_productdealerkey = pd.read_json(productdealerkey)
    
    #Датафрейм df_res будет содержать рекомендации
    df_res = df_dealerprice[['product_key']]
    df_res.columns = ['key']
    
    # Функции очистки текста
    
    # стоп-слова для английского и русского языков
    stop_words_en = set(stopwords.words('english'))
    stop_words_ru = set(stopwords.words('russian'))

    # объединим стоп-слова
    stop_words = stop_words_en.union(stop_words_ru)
    
    def clean_texts(texts):

        if not pd.isna(texts):
            # нижний регистр
            texts = texts.lower()

            # замена всех вхождений \n,\\n,\t,\\ на пробел
            texts = (texts.replace('\\n', ' ')
                     .replace('\n', ' ')
                     .replace('\n\n', ' ')
                     .replace('\t',' ')
                     .replace('\\', ' '))

            # удаление всех вхождений ссылок, начинающихся с "https"
            texts = texts.replace('http\S+', '')

            # удаление текста, заканчивающегося на ".com"
            texts = texts.replace('\ [a-z]*\.com', ' ')

            # удаление знаков препинания (кроме ')
            texts = texts.replace("[^а-яa-z\d\s']+", ' ')

            texts = texts.replace(r'\bprosept\b', '')

            # удаление стоп-слов
            texts = ' '.join([word for word in texts.split() if word not in stop_words])
            #texts = texts.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        else:
            texts=''

        return texts    
    
    #Функция разделения слов
    def str_edit(res):
        if not pd.isna(res):
            res = ' '.join(re.split(r"([A-Za-z][A-Za-z]*)",res))
            #res = ' '.join(re.split(r"\b[А-ЯЁ]([А-ЯЁ][А-ЯЁа-яё]*)",res))
            res = ' '.join(re.split(r"([0-9][0-9]*)",res))
            res = ' '.join(res.split()).lower()
        else:
            res = ''

        return res 
    
    #Объединение двух функций очистки текста.
    def str_edit_clean(res):
        return clean_texts(str_edit(res))
    
    # В дальнейшем будем использовать функцию обработки текста str_edit_clean
    func = str_edit_clean
    
    #Загружаем предъобученную модель LaBSE, при первом запуске потреуется загрузить 1.8 Гб данных
    model_LaBSE = SentenceTransformer('LaBSE')
    
    #В качестве исходного вектора продуктов используем все столбцы с наименованием продукции
    columns = ['name','ozon_name','name_1c','wb_name']
    
    def t_fit_LaBSE(df,func,df_columns=['name']):

        df_tmp = df[df_columns[0]].apply(func)

        if len(df_columns)>1:
            for i in range(1,len(df_columns)):
                df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func)

        model = model_LaBSE.encode(df_tmp)

        return model,df[['id','name']]

    #Получим вектора из датафрейма df_product.
    product_embedding_LaBSE, df_product_LaBSE =  t_fit_LaBSE(df_product,func,columns)

    def t_predict_LaBSE(txt):
        txt_embedding = model_LaBSE.encode(func(txt))
        return util.pytorch_cos_sim(txt_embedding,product_embedding_LaBSE)
    
    #Получаем вектор для спарсенного датафрейма.
    df_predict_LaBSE = df_dealerprice['product_name'].apply(t_predict_LaBSE)

    # 10 индексов лучших совпадений для строк
    n=10
    top_k_matches = []
    top_k_quality = []
    for i in range(df_predict_LaBSE.shape[0]): 
        quality,indices = df_predict_LaBSE.iloc[i].topk(n)
        #print(rank)
        top_k_matches.append(indices)
        top_k_quality.append(quality)


    #Сохраним предсказания в.
    def tens(res):    
        return [res[0][s].item() for s in range(10)]

    df_res.loc[:,'predict'] = top_k_matches
    df_res['predict'] = df_res['predict'].apply(tens)
    df_res.loc[:, 'quality'] = top_k_quality
    df_res['quality'] = df_res['quality'].apply(tens)
    df_res['queue'] = [[x for x in range(1,n+1)] for j in range(len(df_res))]
    df_res=df_res.explode(['predict','queue','quality'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict',axis=1)
    df_res['create_date'] = datetime.now()

    return df_res.to_json()
