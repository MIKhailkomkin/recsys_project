# db_utils.py
import psycopg2
import numpy as np
import pandas as pd

def load_movies_from_db():
    try:
        # Подключение к базе данных PostgreSQL
        print("Подключаемся к базе данных...")
        conn = psycopg2.connect(
            dbname="postgres",  # Замените на название вашей базы данных
            user="postgres",
            password="misha",  # Замените на ваш пароль
            host="localhost"
        )
        cursor = conn.cursor()
        
        print("Подключение успешно!")

        # Запрос данных: ID, название и эмбеддинги
        cursor.execute("SELECT id, title, embedding FROM movies")  # Обновите имя таблицы и столбца если нужно
        rows = cursor.fetchall()

        print(f"Получено {len(rows)} строк из базы данных.")

        ids, titles, embeddings = [], [], []
        for row in rows:
            ids.append(row[0])  # ID фильма
            titles.append(row[1])  # Название фильма
            embeddings.append(row[2])  # Эмбеддинг

        cursor.close()
        conn.close()

        # Создание DataFrame
        df = pd.DataFrame({"id": ids, "title": titles})

        # Преобразуем эмбеддинги в numpy массив
        embedding_matrix = np.array(embeddings)

        print("Данные успешно загружены.")
        return df, embedding_matrix

    except Exception as e:
        print(f"Ошибка при подключении или выполнении запроса: {e}")
        return None, None
