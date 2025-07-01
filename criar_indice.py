import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# --- NOVAS IMPORTAÇÕES ---
import psutil

# --- NOVA FUNÇÃO DE MEDIÇÃO ---
def print_ram_usage():
    process = psutil.Process(os.getpid())
    # rss: Resident Set Size - é a porção da memória que o processo ocupa na RAM.
    # Dividimos por 1024*1024 para converter bytes para Megabytes (MB).
    ram_mb = process.memory_info().rss / (1024 * 1024)
    print(f"--- Uso de RAM atual: {ram_mb:.2f} MB ---")


# --- Início do Script ---
print("--- Iniciando a Ferramenta de Criação de Índice ---")
print_ram_usage() # Medição inicial

# --- Configurações ---
NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'


# 1. Carregar o documento de texto limpo
print(f"\nLendo o arquivo de conhecimento: '{NOME_MANUAL_LIMPO}'...")
if not os.path.exists(NOME_MANUAL_LIMPO):
    print(f"ERRO: Arquivo '{NOME_MANUAL_LIMPO}' não encontrado.")
    exit()

with open(NOME_MANUAL_LIMPO, "r", encoding="utf-8") as f:
    texto_completo = f.read()
print_ram_usage() # Medição após ler o arquivo

# 2. Dividir o texto em parágrafos (chunks)
chunks = [p.strip() for p in texto_completo.split('\n\n') if len(p.strip()) > 50]
print(f"Texto dividido em {len(chunks)} chunks (parágrafos).")
print_ram_usage() # Medição após criar os chunks

# 3. Carregar o modelo de embedding
print(f"\nCarregando o modelo de embedding '{MODELO_EMBEDDING}'... (Este é o passo pesado)")
modelo_embedding_instance = SentenceTransformer(MODELO_EMBEDDING)
print_ram_usage() # <<< ESTA MEDIÇÃO É A MAIS IMPORTANTE

# 4. Gerar os embeddings para cada chunk
print("\nGerando embeddings para cada chunk...")
embeddings = modelo_embedding_instance.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
print("Embeddings gerados com sucesso.")
print_ram_usage() # Medição após gerar os embeddings

# 5. Criar o índice FAISS e adicionar os embeddings
dimensao = embeddings.shape[1]
indice_faiss = faiss.IndexFlatL2(dimensao)
indice_faiss.add(embeddings.astype('float32'))
print("Índice FAISS criado e populado.")
print_ram_usage() # Medição após criar o índice

# 6. Salvar os arquivos de índice e chunks no disco
faiss.write_index(indice_faiss, NOME_ARQUIVO_INDICE)
with open(NOME_ARQUIVO_CHUNKS, 'wb') as f:
    pickle.dump(chunks, f)

print("\n--- Processo Concluído! ---")
print(f"✅ Arquivo de índice '{NOME_ARQUIVO_INDICE}' salvo.")
print(f"✅ Arquivo de chunks '{NOME_ARQUIVO_CHUNKS}' salvo.")
print_ram_usage() # Medição Final