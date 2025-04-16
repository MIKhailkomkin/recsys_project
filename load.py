# load.py
import psycopg2
import pandas as pd
import json
import numpy as np

# Подключение к PostgreSQL
conn = psycopg2.connect(database="postgres", user="postgres", password="misha", host="localhost", port="5432")
cursor = conn.cursor()

# Преобразуем эмбеддинги в строки (или массивы, если хотите использовать FLOAT8[])
embedding_strings = [json.dumps(embedding.tolist()) for embedding in final_embeddings]

# Перебираем строки DataFrame и вставляем их в таблицу
for idx, row in result_df.iterrows():
    embedding = row["embedding"]
    embedding_str = "{" + ",".join(map(str, embedding)) + "}"  # преобразуем в строку для PostgreSQL
    cursor.execute("""
        INSERT INTO movies (id, title, popularity, embedding)
        VALUES (%s, %s, %s, %s)
    """, (row["id"], row["title"], row["popularity"], embedding_str))

# Подтверждаем изменения в базе данных
conn.commit()

# Закрываем соединение
cursor.close()
conn.close()

print("Данные успешно загружены в PostgreSQL!")
