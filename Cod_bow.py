from pathlib import Path
from docx import Document
from nltk.corpus import stopwords
import re
import nltk
import torch
import torch.nn.functional as F

carpeta = Path.cwd()
archivos = sorted(carpeta.glob("doc*.docx"))

esperados = [f"doc{i:02d}.docx" for i in range(1, 13)]
nombres_encontrados = [ruta.name for ruta in archivos]
if nombres_encontrados != esperados:
    raise FileNotFoundError(
        f"Se esperaban los archivos {esperados} en {carpeta}. "
        f"Se encontraron: {nombres_encontrados}."
    )

documentos = []
nombres = []
for ruta in archivos:
    documento_word = Document(ruta)
    texto = " ".join(
        parrafo.text for parrafo in documento_word.paragraphs
        if parrafo.text.strip()
    )
    if not texto.strip():
        raise ValueError(f"El documento {ruta.name} no contiene texto en párrafos.")
    documentos.append(texto)
    nombres.append(ruta.name)

print(f"Documentos cargados ({len(nombres)}): {nombres}")

nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("spanish"))

def preprocesar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    palabras = texto.split()
    palabras = [p for p in palabras if p not in STOPWORDS]
    return palabras

docs_tokenizados = [preprocesar(doc) for doc in documentos]

for nombre, tokens in zip(nombres, docs_tokenizados):
    print(f"{nombre}: {len(tokens)} palabras tras preprocesado")

vocabulario = sorted(set(palabra for tokens in docs_tokenizados for palabra in tokens))
palabra_a_indice = {palabra: i for i, palabra in enumerate(vocabulario)}

print(f"Tamaño del vocabulario: {len(vocabulario)} palabras")

def vectorizar_bow(tokens, vocab_index):
    vector = torch.zeros(len(vocab_index))
    for palabra in tokens:
        if palabra in vocab_index:
            vector[vocab_index[palabra]] += 1
    return vector

matriz_bow = torch.stack([vectorizar_bow(tokens, palabra_a_indice) for tokens in docs_tokenizados])
print(f"Forma de la matriz BoW: {matriz_bow.shape}")

n_docs = matriz_bow.shape[0]
similitud_bow = torch.zeros((n_docs, n_docs))

for i in range(n_docs):
    for j in range(n_docs):
        similitud_bow[i, j] = F.cosine_similarity(
            matriz_bow[i].unsqueeze(0), matriz_bow[j].unsqueeze(0)
        )

print("Matriz de similitud del coseno (BoW):")
for i in range(n_docs):
    fila = " ".join(f"{similitud_bow[i, j]:.2f}" for j in range(n_docs))
    print(f"{nombres[i]}: {fila}")

pares = []
for i in range(n_docs):
    for j in range(i + 1, n_docs):
        pares.append((similitud_bow[i, j].item(), nombres[i], nombres[j]))

pares_ordenados = sorted(pares, key=lambda x: x[0], reverse=True)

print("Pares más similares (BoW):")
for sim, a, b in pares_ordenados[:5]:
    print(f"  {a} - {b}: {sim:.3f}")

print("Pares menos similares (BoW):")
for sim, a, b in pares_ordenados[-5:]:
    print(f"  {a} - {b}: {sim:.3f}")

for nombre, texto in zip(nombres, documentos):
    extracto = " ".join(texto.split())[:300]
    print(f"{nombre}: {extracto}{'…' if len(texto) > 300 else ''}\n")

print("Términos compartidos en los cinco pares más similares:")
for sim, a, b in pares_ordenados[:5]:
    ia, ib = nombres.index(a), nombres.index(b)
    comunes = set(docs_tokenizados[ia]) & set(docs_tokenizados[ib])
    ordenados = sorted(
        comunes,
        key=lambda p: (-(docs_tokenizados[ia].count(p) * docs_tokenizados[ib].count(p)), p)
    )
    print(f"{a} – {b} ({sim:.3f}): {', '.join(ordenados[:15])}")

N = matriz_bow.shape[0]

total_palabras_doc = matriz_bow.sum(dim=1, keepdim=True)
tf = matriz_bow / total_palabras_doc

df = (matriz_bow > 0).sum(dim=0)

idf = torch.log(torch.tensor(N, dtype=torch.float32) / df.float())

matriz_tfidf = tf * idf

print(f"Forma de la matriz TF-IDF: {matriz_tfidf.shape}")

similitud_tfidf = torch.zeros((N, N))

for i in range(N):
    for j in range(N):
        similitud_tfidf[i, j] = F.cosine_similarity(
            matriz_tfidf[i].unsqueeze(0),
            matriz_tfidf[j].unsqueeze(0)
        )

print("Matriz de similitud del coseno (TF-IDF):")

for i in range(N):
    fila = " ".join(f"{similitud_tfidf[i, j]:.2f}" for j in range(N))
    print(f"{nombres[i]}: {fila}")

pares_tfidf = []

for i in range(N):
    for j in range(i + 1, N):
        pares_tfidf.append(
            (
                similitud_tfidf[i, j].item(),
                nombres[i],
                nombres[j]
            )
        )

pares_tfidf_ordenados = sorted(
    pares_tfidf,
    key=lambda x: x[0],
    reverse=True
)

print("Pares más similares (TF-IDF):")
for sim, a, b in pares_tfidf_ordenados[:5]:
    print(f"  {a} - {b}: {sim:.3f}")

print("\nPares menos similares (TF-IDF):")
for sim, a, b in pares_tfidf_ordenados[-5:]:
    print(f"  {a} - {b}: {sim:.3f}")

print("Comparación de las 5 parejas más similares")
print("\nBoW:")
for sim, a, b in pares_ordenados[:5]:
    print(f"  {a} - {b}: {sim:.3f}")

print("\nTF-IDF:")
for sim, a, b in pares_tfidf_ordenados[:5]:
    print(f"  {a} - {b}: {sim:.3f}")