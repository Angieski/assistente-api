import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configurações ---
NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'

print("--- Iniciando a Ferramenta de Criação de Índice ---")

# 1. Carregar o documento de texto limpo
print(f"Lendo o arquivo de conhecimento: '{NOME_MANUAL_LIMPO}'...")
if not os.path.exists(NOME_MANUAL_LIMPO):
    print(f"ERRO: Arquivo '{NOME_MANUAL_LIMPO}' não encontrado. Crie este arquivo primeiro.")
    exit()

with open(NOME_MANUAL_LIMPO, "r", encoding="utf-8") as f:
    texto_completo = f.read()

# 2. Dividir o texto em parágrafos (chunks)
chunks = [p.strip() for p in texto_completo.split('\n\n') if len(p.strip()) > 50]
print(f"Texto dividido em {len(chunks)} chunks (parágrafos).")

# 3. Carregar o modelo de embedding
print(f"Carregando o modelo de embedding '{MODELO_EMBEDDING}'... (Isso pode demorar um pouco)")
modelo_embedding_instance = SentenceTransformer(MODELO_EMBEDDING)

# 4. Gerar os embeddings para cada chunk
print("Gerando embeddings para cada chunk. Este é o passo mais demorado...")
embeddings = modelo_embedding_instance.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
print("Embeddings gerados com sucesso.")

# 5. Criar o índice FAISS e adicionar os embeddings
dimensao = embeddings.shape[1]
indice_faiss = faiss.IndexFlatL2(dimensao)
indice_faiss.add(embeddings.astype('float32'))
print("Índice FAISS criado e populado.")

# 6. Salvar os arquivos de índice e chunks no disco
faiss.write_index(indice_faiss, NOME_ARQUIVO_INDICE)
with open(NOME_ARQUIVO_CHUNKS, 'wb') as f:
    pickle.dump(chunks, f)

print("\n--- Processo Concluído! ---")
print(f"✅ Arquivo de índice '{NOME_ARQUIVO_INDICE}' salvo.")
print(f"✅ Arquivo de chunks '{NOME_ARQUIVO_CHUNKS}' salvo.")
print("Seu assistente agora tem uma memória nova e atualizada!")