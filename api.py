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

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'all-MiniLM-L6-v2'

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

# --- FUNÇÕES DE LÓGICA DA IA (O CÉREBRO) ---
def buscar_contexto_local(pergunta, top_k=3):
    if indice_faiss is None: return ""
    pergunta_embedding = modelo_embedding_instance.encode([pergunta], normalize_embeddings=True)
    distancias, indices = indice_faiss.search(pergunta_embedding.astype('float32'), top_k)
    # Verificação de relevância baseada na distância. Se for muito distante, não retorna nada.
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

# --- FUNÇÃO GENERATIVA ÚNICA E DEFINITIVA ---
def obter_resposta_generativa(pergunta_atual, historico, contexto_manual, contexto_web):
    if not client: return "O serviço de IA não está configurado corretamente (sem chave de API)."
    
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Sua tarefa é responder a PERGUNTA ATUAL do usuário.
    Para isso, você tem um HISTÓRICO de conversa e duas fontes de conhecimento: um MANUAL TÉCNICO e CONTEÚDO DA WEB.

    **SUA LÓGICA DE DECISÃO E REGRAS ESTRITAS:**
    1.  **PRIORIDADE MÁXIMA AO MANUAL:** Verifique PRIMEIRO se o CONTEXTO DO MANUAL TÉCNICO contém a resposta para a PERGUNTA ATUAL (use o HISTÓRICO para entender perguntas como "e sobre ele?").
    2.  **SE A RESPOSTA ESTIVER NO MANUAL:** Baseie sua resposta **100%** no manual. IGNORE completamente o CONTEÚDO DA WEB.
    3.  **SE A RESPOSTA NÃO ESTIVER NO MANUAL:** Então, e somente então, use o CONTEÚDO DA WEB para responder.
    4.  **SEJA DIRETO:** Responda diretamente à pergunta. Não mencione as fontes (ex: "No manual...").
    5.  **REGRA DE FALHA:** Se a informação não estiver em nenhuma das fontes, responda APENAS com: "Não encontrei informações sobre isso em minhas fontes."
    6.  **TODA RESPOSTA DEVE SER EM PORTUGUÊS BRASILEIRO.

    ---
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    ---
    FONTE 1 (PRIORITÁRIA): CONTEXTO DO MANUAL TÉCNICO
    {contexto_manual or "Nenhum contexto do manual foi encontrado para esta pergunta."}
    ---
    FONTE 2 (SECUNDÁRIA): CONTEÚDO DA WEB
    {contexto_web or "Nenhum contexto da web foi encontrado para esta pergunta."}
    ---
    PERGUNTA ATUAL DO USUÁRIO:
    {pergunta_atual}
    ---
    RESPOSTA DIRETA:
    """
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_completo}],
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content

# --- CRIAÇÃO DA API COM FLASK (LÓGICA SIMPLIFICADA) ---
app = Flask(__name__)
CORS(app)

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    # Busca em ambas as fontes
    contexto_manual = buscar_contexto_local(pergunta_atual)
    contexto_web = buscar_na_web(pergunta_atual)
    
    # Envia tudo para a IA e confia na sua lógica de decisão interna
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_manual, contexto_web)
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)