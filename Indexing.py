import webbrowser
from tkinter import ttk
import requests
import tkinter as tk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Fonction pour récupérer le contenu d'une page Wikipedia
def get_page_content(page_title):
    url = f"https://en.wikipedia.org/w/api.php?action=parse&page={page_title}&format=json"
    response = requests.get(url)
    json_data = response.json()
    html_content = json_data["parse"]["text"]["*"]
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()


# Indexer les pages en :utilisant la mesure tf-idf
def tf_idf(page_contents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(page_contents)
    return(tfidf_vectorizer,tfidf_matrix)

#calculer le similarité entre la requete et les documents de la selection 
def similarities(tfidf_vectorizer,tfidf_matrix,query):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_page_indices = cosine_similarities.argsort()[::-1]
    return cosine_similarities,related_page_indices



# Fonction pour effectuer une recherche sur Wikipedia
def search(query):
    page_titles = []
    page_contents = []
    
    # Récupérer les pages Wikipedia qui contiennent la requête
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
    response = requests.get(url)
    json_data = response.json()
    for result in json_data["query"]["search"]:
        page_titles.append(result["title"])
        page_content = get_page_content(result["title"])
        page_contents.append(page_content)
        
    # Indexer les pages en utilisant la mesure tf-idf
    tfidf_vectorizer,tfidf_matrix = tf_idf(page_contents)
    
    # Calculer les similarités cosinus entre la requête et les pages Wikipedia
    cosine_similarities,related_page_indices=similarities(tfidf_vectorizer,tfidf_matrix,query)
    return [(page_titles[i], cosine_similarities[i]) for i in related_page_indices]


# Fonction pour afficher les résultats de la recherche
def display_results(results):
    for child in results_frame.winfo_children():
        child.destroy()
    for i, result in enumerate(results):
        page_title, score = result
        result_button = ttk.Button(
            results_frame,
            text=f"{i+1}. {page_title} ({score:.2f})",
            command=lambda title=page_title: webbrowser.open(f"https://en.wikipedia.org/wiki/{title}")
        )
        result_button.pack()


# Fonction pour exécuter une recherche lorsque le bouton est cliqué
def search_button_click():
    query = query_entry.get()
    results = search(query)
    display_results(results)


# Création de l'interface graphique
root = tk.Tk()
root.title("Moteur de recherche Wikipedia")

query_frame = ttk.Frame(root, padding="10")
query_frame.pack(fill="x")

ttk.Label(query_frame, text="Recherche :").pack(side="left")
query_entry = ttk.Entry(query_frame)
query_entry.pack(side="left", fill="x", expand=True)
query_entry.bind("<Return>", lambda event: search_button_click())

search_button = ttk.Button(query_frame, text="Rechercher", command=search_button_click)
search_button.pack(side="left", padx="5")

results_frame = ttk.Frame(root, padding="10")
results_frame.pack(fill="both", expand=True)

root.mainloop()
