# main.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import ttkbootstrap as ttk
from db_utils import load_movies_from_db
from recommender import MovieRecommender
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Загружаем фильмы и эмбеддинги
movie_df, embedding_matrix = load_movies_from_db()

# Инициализируем рекомендательную систему
recommender = MovieRecommender()
movie_df['emb'] = list(embedding_matrix)
recommender.prepare_data(movie_df)

class MovieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Recommendation Engine")
        self.root.geometry("1000x700")
        self.style = ttk.Style(theme="darkly")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=20)
        
        ttk.Label(
            header, 
            text="🎬 Movie Recommendation Engine", 
            font=('Helvetica', 20, 'bold')
        ).pack(side=tk.LEFT, padx=20)
        
        # Movie Selection
        selection_frame = ttk.Frame(main_frame)
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Listbox with Scrollbar
        self.listbox = tk.Listbox(
            selection_frame,
            selectmode=tk.MULTIPLE,
            bg='#303030',
            fg='white',
            font=('Helvetica', 12),
            height=15
        )
        scrollbar = ttk.Scrollbar(
            selection_frame, 
            orient=tk.VERTICAL, 
            command=self.listbox.yview
        )
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Populate listbox with all movie titles initially
        self.update_listbox(movie_df["title"])
        
        # Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        ttk.Button(
            button_frame,
            text="Get Recommendations",
            command=self.recommend,
            bootstyle=ttk.INFO,
            width=20,
            padding=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Results Frame
        self.results_frame = ttk.Frame(main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
    def update_listbox(self, titles):
        """Обновить содержимое Listbox с новыми фильмами"""
        self.listbox.delete(0, tk.END)  # Очистить список
        for title in titles:
            self.listbox.insert(tk.END, title)
    
    def clear_selection(self):
        """Очистить выбранные фильмы в Listbox"""
        self.listbox.selection_clear(0, tk.END)
        
    def recommend(self):
        """Генерация рекомендаций на основе выбранных фильмов"""
        try:
            selected_indices = [int(i) for i in self.listbox.curselection()]
            if not selected_indices:
                raise ValueError("Please select at least one movie")
            
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
                
            # Loading indicator
            loading = ttk.Label(
                self.results_frame, 
                text="Generating recommendations...", 
                font=('Helvetica', 12)
            )
            loading.pack(pady=50)
            self.root.update()
            
            def worker(index):
                title = movie_df.iloc[index]["title"]
                query_emb = embedding_matrix[index]
                similar = recommender.find_similar(query_emb)
                return (title, similar)
            
            with ThreadPoolExecutor() as executor:
                recommendations = list(executor.map(worker, selected_indices))
            
            loading.destroy()
            
            # Create notebook for tabs
            notebook = ttk.Notebook(self.results_frame)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            for title, recs in recommendations:
                # Create tab for each selected movie
                tab = ttk.Frame(notebook)
                notebook.add(tab, text=title[:20] + "...")
                
                # Header
                ttk.Label(
                    tab,
                    text=f"Movies similar to: {title}",
                    font=('Helvetica', 12, 'bold')
                ).pack(pady=10)
                
                # Create visualization
                self.create_visualization(tab, recs)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_visualization(self, parent, recommendations):
        """Создание единого большого барчарта с подписями внутри баров"""
        # Prepare data
        titles = [rec[0] for rec in recommendations]
        scores = [rec[1] for rec in recommendations]
    
        # Create figure
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
    
        # Horizontal bar chart
        y_pos = np.arange(len(titles))
        bars = ax.barh(y_pos, scores, color='#3498db')
    
        # Add value labels inside bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                    f'{score:.2f}', 
                    va='center', ha='right', color='white', fontweight='bold')
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                titles[i], va='center', ha='left', color='white', fontweight='bold')
    
    # Customize appearance
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])  # Remove default y-tick labels, as we are placing titles next to the bars
        ax.set_xlim(0, 1)
        ax.set_title('Similarity Scores')
        ax.set_facecolor('#303030')
        fig.patch.set_facecolor('#303030')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.title.set_color('white')
    
    # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

if __name__ == "__main__":
    root = ttk.Window(themename="darkly")
    app = MovieApp(root)
    root.mainloop()
