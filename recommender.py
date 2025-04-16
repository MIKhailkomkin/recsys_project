# recommender.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
from tqdm import tqdm  # для визуализации прогресса

class MovieRecommender:
    def __init__(self, expected_dim=387):
        self.expected_dim = expected_dim  # Ожидаемая размерность эмбеддингов
    
    def safe_parse(self, embedding):
        """ преобразование эмбеддинга с обработкой всех исключений"""
       
        try:
            if isinstance(embedding, str):
                parsed = ast.literal_eval(embedding)
            else:
                parsed = embedding
                
            arr = np.array(parsed, dtype=np.float32)
            if len(arr) != self.expected_dim:
                return None
            return arr
        except (ValueError, SyntaxError, TypeError):
            try:
                if isinstance(embedding, str):
                    clean_str = embedding.strip("[] \n\r\t")
                    if not clean_str:
                        return None
                    arr = np.fromstring(clean_str, sep=',', dtype=np.float32)
                    if len(arr) == self.expected_dim:
                        return arr
            except:
                pass
        return None

    def prepare_data(self, df):
        """Подготовка данных и фильтрация невалидных записей"""
        print("Обработка эмбеддингов...")
        valid_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            emb = self.safe_parse(row['emb'])
            if emb is not None:
                valid_data.append({
                    'id': row['id'],
                    'title': row['title'],
                    'embedding': emb
                })
        
        if not valid_data:
            raise ValueError("Не найдено ни одного валидного эмбеддинга")
            
        self.df = pd.DataFrame(valid_data)
        self.embeddings = np.stack(self.df['embedding'].values)
        print(f"Готово. Загружено {len(self.df)} валидных эмбеддингов.")
        
    def find_similar(self, query, top_k=5):
        """Поиск похожих фильмов"""
        query_emb = self.safe_parse(query)
        if query_emb is None:
            raise ValueError("Неверный формат query эмбеддинга")
        
        similarities = cosine_similarity([query_emb], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
        
        return [
            (self.df.iloc[i]['title'], float(similarities[i]))
            for i in top_indices
        ]
