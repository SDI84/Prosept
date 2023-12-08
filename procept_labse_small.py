# Импорт необходимых библиотек
import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import pandas as pd
from sentence_transformers import util
from nltk.corpus import stopwords
import re
import json

# Загрузка стоп-слов для английского и русского языков
stop_words_en = set(stopwords.words('english'))
stop_words_ru = set(stopwords.words('russian'))
stop_words = stop_words_en.union(stop_words_ru)

# Загрузка токенизатора и модели LaBSE_ru_en (516 Мб)и
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



# Основная функция для предсказания
def prosept_predict(product, dealerprice) -> list:
    #задаем функцию обработки текста
    func = clean_texts
    #выдача 10 правильных ответов
    N_BEST = 10
    
    # Преобразование словарей в DataFrame
    df_product = pd.DataFrame.from_dict(product)
    df_dealerprice = pd.DataFrame.from_dict(dealerprice)


    # Создание основного датафрейма для результатов
    df_res = df_dealerprice[['id', 'product_key', 'product_name']]
    df_dealerprice_unique = (df_dealerprice[['product_name']]
                             .drop_duplicates(subset='product_name')
                             .reset_index(drop=True))

    # Обучение модели
    columns = ['name', 'ozon_name', 'name_1c', 'wb_name']

    def t_fit_LaBSE(df,func,df_columns=['name']):


        df_tmp = df[df_columns[0]].apply(func)

        if len(df_columns)>1:
            for i in range(1,len(df_columns)):
                df_tmp = df_tmp + ' ' + df[df_columns[i]].apply(func)


        list_tmp = df_tmp.tolist()
        #print(list_tmp)
        encoded_input = tokenizer(list_tmp
                                  , padding=True
                                  , truncation=True
                                  , max_length=61
                                  , return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)            




        #model = model_LaBSE.encode(df_tmp)

        return embeddings,df[['id','name']]

    #Получим вектора из датафрейма df_product.
    product_embedding_LaBSE, df_product_LaBSE =  t_fit_LaBSE(df_product,func,columns)
    


    def t_predict_LaBSE(txt):          
        encoded_input = tokenizer(func(txt)
                                  , padding=True
                                  , truncation=True
                                  , max_length=61
                                  , return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.pooler_output
        txt_embedding = torch.nn.functional.normalize(embeddings)             


        return util.pytorch_cos_sim(txt_embedding,product_embedding_LaBSE)





        
    # Получаем матрицу расстояний
    df_predict_LaBSE = df_dealerprice_unique['product_name'].apply(t_predict_LaBSE)
    #df_predict_LaBSE = util.pytorch_cos_sim(df_dealerprice_unique['product_name'].apply(t_predict_LaBSE)
                                            #,product_embedding_LaBSE)
    # 10 индексов лучших совпадений для строк
    top_k_matches = []
    top_k_quality = []
    for i in range(df_predict_LaBSE.shape[0]): 
        quality,indices = df_predict_LaBSE.iloc[i].topk(N_BEST)
        #print(rank)
        top_k_matches.append(indices)
        top_k_quality.append(quality)


    #Сохраним предсказания в.
    def tens(res):    
        return [res[0][s].item() for s in range(N_BEST)]


    df_dealerprice_unique.loc[:,'predict'] = top_k_matches    
    df_dealerprice_unique.loc[:,'quality'] = top_k_quality
    

    df_dealerprice_unique['predict'] = df_dealerprice_unique['predict'].apply(tens)
    df_dealerprice_unique['quality'] = df_dealerprice_unique['quality'].apply(tens)



    
    df_res=df_res.merge(df_dealerprice_unique, how='left', on=['product_name'])

    
    df_res['queue'] = [[x for x in range(1,N_BEST+1)] for j in range(len(df_res))]
    df_res=df_res.explode(['predict','queue','quality'])
    df_res = df_res.reset_index(drop=True)
    tmp_df = df_product['id'].loc[df_res['predict']].reset_index(drop=True)
    df_res['product_id'] = tmp_df
    df_res = df_res.drop('predict',axis=1)
    df_res['create_date'] = datetime.now()
    df_res = df_res.drop(columns = ['product_name'],axis=1)

     # Результат в JSON
    result_json = df_res.to_json(orient='records')

    return json.loads(result_json) # список словарей


