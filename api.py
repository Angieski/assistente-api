import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO FINAL COM DOSSIÊ)...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
CONTEUDO_MANUAL = ""

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Cliente da API da Groq configurado.")
except Exception as e:
    print(f"ERRO: Chave da API da Groq não encontrada. Configure a variável de ambiente. Erro: {e}")
    client = None

if os.path.exists(NOME_MANUAL_LIMPO):
    with open(NOME_MANUAL_LIMPO, "r", encoding="utf-8") as f:
        CONTEUDO_MANUAL = f.read()
    print(f"Manual '{NOME_MANUAL_LIMPO}' carregado na memória.")
else:
    print(f"AVISO: Arquivo de manual '{NOME_MANUAL_LIMPO}' não encontrado.")


# --- FUNÇÕES DE LÓGICA DA IA ---
def buscar_na_web(pergunta):
    print(f"Iniciando busca na web para: '{pergunta}'")
    query = pergunta
    try:
        with DDGS() as ddgs:
            resultados_links = list(ddgs.text(query, max_results=1, region='br-pt'))
            if not resultados_links: return "Nenhum resultado encontrado na web."
            url = resultados_links[0]['href']
            print(f"Extraindo conteúdo de: {url}")
            downloaded = fetch_url(url)
            if downloaded:
                return extract(downloaded, include_comments=False, include_tables=False)
            return "Não foi possível extrair conteúdo da página."
    except Exception as e:
        print(f"Erro na busca web: {e}")
        return "Ocorreu um erro na busca web."

def obter_resposta_generativa(pergunta_atual, historico, contexto_manual, contexto_web):
    if not client: return "O serviço de IA não está configurado."
    
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Sua tarefa é responder a PERGUNTA ATUAL do usuário. Para isso, você tem um HISTÓRICO de conversa e duas fontes de conhecimento: um MANUAL TÉCNICO e CONTEÚDO DA WEB.

    **SUA LÓGICA DE DECISÃO E REGRAS ESTRITAS:**
    1.  **PRIORIDADE MÁXIMA AO MANUAL:** Verifique PRIMEIRO se o CONTEXTO DO MANUAL TÉCNICO contém a resposta para a PERGUNTA ATUAL (use o HISTÓRICO para entender perguntas como "e sobre ele?").
    2.  **SE A RESPOSTA ESTIVER NO MANUAL:** Baseie sua resposta **100%** no manual. IGNORE completamente o CONTEÚDO DA WEB.
    3.  **SE A RESPOSTA NÃO ESTIVER NO MANUAL:** Então, e somente então, use o CONTEÚDO DA WEB para responder.
    4.  **SEJA DIRETO:** Responda diretamente à pergunta. Não mencione as fontes (ex: "No manual...").
    5.  **REGRA DE FALHA:** Se a informação não estiver em nenhuma das fontes, responda APENAS com: "Não encontrei informações sobre isso em minhas fontes."

    ---
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    ---
    FONTE 1 (PRIORITÁRIA): CONTEXTO DO MANUAL TÉCNICO
    {contexto_manual or "Nenhuma informação do manual disponível para esta pergunta."}
    ---
    FONTE 2 (SECUNDÁRIA): CONTEÚDO DA WEB
    {contexto_web or "Nenhuma informação da web disponível para esta pergunta."}
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

# --- CRIAÇÃO DA API COM FLASK ---
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": ["https://consolemix.com.br", "http://consolemix.com.br", "http://localhost", "http://127.0.0.1"]}})

@app.route('/')
def health_check():
    return "API do assistente especialista (versão dossiê) está no ar!"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    # Busca na web é feita de forma leve e rápida
    contexto_web = buscar_na_web(pergunta_atual)
    
    # Envia o dossiê completo para a IA e confia na sua lógica de decisão
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, CONTEUDO_MANUAL, contexto_web)
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)