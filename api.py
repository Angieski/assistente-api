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
print("Iniciando a configuração do servidor (VERSÃO FINAL)...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'all-MiniLM-L6-v2' # Usando o modelo leve e otimizado

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Cliente da API da Groq configurado.")
except Exception as e:
    print(f"ERRO: Chave da API da Groq não encontrada. Configure a variável de ambiente. Erro: {e}")
    client = None

# Esta é a versão leve, então não carregamos os modelos de busca local na inicialização.
# Eles seriam carregados se você estivesse em um plano com mais RAM.
print("API configurada para usar a busca na web.")


# --- FUNÇÕES DE LÓGICA DA IA (O CÉREBRO) ---
def buscar_na_web(pergunta, num_artigos=1):
    print(f"Iniciando busca na web para: '{pergunta}'")
    query = pergunta
    try:
        with DDGS() as ddgs:
            resultados_links = list(ddgs.text(query, max_results=3, region='br-pt'))
            if not resultados_links: return "Nenhum resultado encontrado na web."
            url = resultados_links[0]['href']
            print(f"Extraindo conteúdo de: {url}")
            downloaded = fetch_url(url)
            if downloaded:
                texto_artigo = extract(downloaded, include_comments=False, include_tables=False)
                return texto_artigo
            return "Não foi possível extrair conteúdo da página."
    except Exception as e:
        print(f"Erro na busca web: {e}")
        return "Ocorreu um erro na busca web."

def obter_resposta_generativa(pergunta_atual, historico, contexto):
    if not client: return "O serviço de IA não está configurado corretamente (sem chave de API)."
    
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente de pesquisa que responde perguntas baseadas em um CONTEXTO obtido da web.

    REGRAS ESTRITAS:
    1. Baseie sua resposta APENAS no CONTEXTO fornecido.
    2. Seja direto e objetivo.
    3. Nunca mencione o CONTEXTO. Apenas dê a resposta.
    4. REGRA DE FALHA: Se a resposta não estiver no CONTEXTO, responda APENAS com: "Não encontrei informações sobre isso na minha pesquisa."

    ---
    HISTÓRICO DA CONVERSA: {historico_formatado}
    ---
    CONTEXTO DA WEB: {contexto}
    ---
    PERGUNTA ATUAL DO USUÁRIO: {pergunta_atual}
    ---
    RESPOSTA DIRETA:
    """
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_completo}],
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content

# --- CRIAÇÃO DA API COM FLASK ---
app = Flask(__name__)

# --- CONFIGURAÇÃO DE CORS ATUALIZADA E EXPLÍCITA ---
# Substitua 'https://consolemix.com.br' pelo domínio exato do seu site, se for diferente.
origins_permitidas = [
    "https://consolemix.com.br",
    "http://consolemix.com.br",
    "http://localhost",
    "http://127.0.0.1"
]
CORS(app, resources={r"/ask": {"origins": origins_permitidas}})
# --- FIM DA ATUALIZAÇÃO ---


@app.route('/')
def health_check():
    return "API do assistente (versão leve) está no ar e funcionando!"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    # Versão leve sempre usa a busca na web
    contexto_web = buscar_na_web(pergunta_atual)
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_web)
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)