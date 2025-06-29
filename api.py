# api.py (Versão Estável para Publicação)
import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
from groq import Groq

print("Iniciando a configuração do servidor...")
NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"ERRO: Chave da API da Groq não encontrada. Configure a variável de ambiente. Erro: {e}")
    client = None

print("Carregando modelo de embedding...")
modelo_embedding_instance = SentenceTransformer(MODELO_EMBEDDING)
indice_faiss = None
chunks = []
if os.path.exists(NOME_ARQUIVO_INDICE) and os.path.exists(NOME_ARQUIVO_CHUNKS):
    print("Carregando índice FAISS e chunks...")
    indice_faiss = faiss.read_index(NOME_ARQUIVO_INDICE)
    with open(NOME_ARQUIVO_CHUNKS, 'rb') as f:
        chunks = pickle.load(f)
    print("Recursos carregados com sucesso.")
else:
    print("AVISO: Arquivos de índice não encontrados. A busca local estará desativada.")

def buscar_contexto_local(pergunta, top_k=3):
    if indice_faiss is None: return ""
    pergunta_embedding = modelo_embedding_instance.encode([pergunta], normalize_embeddings=True)
    distancias, indices = indice_faiss.search(pergunta_embedding.astype('float32'), top_k)
    if distancias[0][0] > 1.2: return ""
    return "\n\n---\n\n".join([chunks[i] for i in indices[0]])

def buscar_na_web(pergunta, num_artigos=1):
    query = pergunta
    try:
        with DDGS() as ddgs:
            resultados_links = list(ddgs.text(query, max_results=3, region='br-pt'))
            if not resultados_links: return "Nenhum resultado encontrado na web."
            url = resultados_links[0]['href']
            downloaded = fetch_url(url)
            if downloaded:
                texto_artigo = extract(downloaded, include_comments=False, include_tables=False)
                return texto_artigo
            return "Não foi possível extrair conteúdo da página."
    except Exception:
        return "Ocorreu um erro na busca web."

def contexto_e_relevante(pergunta, contexto):
    if not contexto or not client: return False
    prompt = f"O CONTEXTO a seguir é relevante para responder a PERGUNTA? Responda APENAS 'SIM' ou 'NÃO'.\n\nCONTEXTO: \"{contexto}\"\n\nPERGUNTA: \"{pergunta}\""
    try:
        chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192", temperature=0)
        return "SIM" in chat_completion.choices[0].message.content.upper()
    except Exception:
        return False

def obter_resposta_generativa(pergunta, contexto, historico=None): # historico é ignorado nesta versão
    if not client: return "O serviço de IA não está configurado corretamente."
    prompt = f"Você é um assistente técnico especialista. Baseado exclusivamente no CONTEXTO, responda a PERGUNTA de forma direta.\n\nREGRAS:\n- Se a resposta não estiver no CONTEXTO, responda APENAS com: 'Não encontrei informações sobre isso na fonte consultada.'\n- NUNCA mencione o contexto ou a fonte.\n\nCONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}\n\nRESPOSTA:"
    chat_completion = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama3-8b-8192")
    return chat_completion.choices[0].message.content

app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400
    pergunta_atual = data['question']
    contexto_manual = buscar_contexto_local(pergunta_atual)
    if contexto_e_relevante(pergunta_atual, contexto_manual):
        resposta_final = obter_resposta_generativa(pergunta_atual, contexto_manual)
    else:
        contexto_web = buscar_na_web(pergunta_atual)
        resposta_final = obter_resposta_generativa(pergunta_atual, contexto_web)
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)