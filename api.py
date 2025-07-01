import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO FINAL LEVE)...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
CONTEUDO_MANUAL = ""

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Cliente da API da Groq configurado.")
except Exception as e:
    print(f"ERRO: Chave da API da Groq não encontrada. Configure a variável de ambiente. Erro: {e}")
    client = None

# Carrega o conteúdo do manual na memória uma única vez na inicialização
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

def obter_resposta_generativa(pergunta_atual, historico, contexto, fonte):
    if not client: return "O serviço de IA não está configurado corretamente."
    
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Sua tarefa é responder a PERGUNTA ATUAL do usuário baseando-se no CONTEXTO fornecido pela fonte '{fonte}'.

    REGRAS ESTRITAS:
    1.  Responda APENAS com base no CONTEXTO.
    2.  Seja direto e objetivo. Não mencione o contexto.
    3.  REGRA DE FALHA: Se a resposta não estiver no CONTEXTO, responda APENAS com: "Não encontrei informações sobre isso na fonte consultada."

    ---
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    ---
    CONTEXTO DE CONSULTA (Fonte: {fonte}):
    {contexto}
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
CORS(app)

@app.route('/')
def health_check():
    return "API do assistente especialista (versão leve) está no ar!"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    # Lógica final: usa o manual inteiro como contexto e passa para a IA decidir.
    # Se o manual não existir, o contexto será vazio.
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, CONTEUDO_MANUAL, "Manual Técnico")

    # Verificamos se a resposta da IA indica que a informação não foi encontrada
    if "não encontrei informações sobre isso" in resposta_final.lower():
        print("Resposta não encontrada no manual. Partindo para a busca na web.")
        contexto_web = buscar_na_web(pergunta_atual)
        resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_web, "Web")

    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)