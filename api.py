import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO FINAL COMPLETA)...")

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


# --- FUNÇÕES DE LÓGICA DA IA (O CÉREBRO) ---

def buscar_na_web(pergunta):
    """Função para buscar na web e extrair o conteúdo principal de um artigo."""
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
                texto_artigo = extract(downloaded, include_comments=False, include_tables=False)
                return texto_artigo
            return "Não foi possível extrair conteúdo da página."
    except Exception as e:
        print(f"Erro na busca web: {e}")
        return "Ocorreu um erro na busca web."

def obter_resposta_generativa(pergunta_atual, historico, contexto, fonte_do_contexto):
    """Gera uma resposta da IA baseada no contexto e histórico fornecidos."""
    if not client: 
        return "O serviço de IA não está configurado."
    if not contexto:
        return f"Não encontrei informações sobre isso na fonte consultada ({fonte_do_contexto})."

    historico_recente = historico[-6:]
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico_recente])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Responda a PERGUNTA ATUAL do usuário baseando-se exclusivamente no CONTEXTO DE CONSULTA.

    REGRAS ESTRITAS:
    1.  Use o HISTÓRICO DA CONVERSA para entender perguntas de acompanhamento.
    2.  Sua resposta deve vir APENAS do CONTEXTO DE CONSULTA. Não use conhecimento prévio.
    3.  Seja direto e não mencione o contexto ou a fonte. NÃO diga coisas do tipo "a fonte que consultei".
    4.  REGRA DE FALHA: Se a resposta não estiver no CONTEXTO DE CONSULTA, responda APENAS com a frase: "Não encontrei informações sobre isso na fonte consultada."
    5.  NÃO CORRIJA a ortografia do usuário NEM RESPONDA QUESTÕES DE GRAMATICA. VOCE É UM ASSISTENTE TÉCNICO, não um PROFESSOR DE PORTUGUES.
    6.  TODA RESPOSTA DEVE SER EM PORTUGUÊS. NÃO RESPONDA EM OUTRAS LINGUAGENS.

    ---

    --- 

    ---
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    ---
    CONTEXTO DE CONSULTA (Fonte: {fonte_do_contexto}):
    {contexto}
    ---
    PERGUNTA ATUAL DO USUÁRIO:
    {pergunta_atual}
    ---
    RESPOSTA DIRETA:
    """
    
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt_completo}],
        model="llama-3.1-8b-instant"
    )
    return chat_completion.choices[0].message.content

# --- CRIAÇÃO DA API COM FLASK ---
app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": ["https://consolemix.com.br", "http://consolemix.com.br", "http://localhost", "http://127.0.0.1"]}})

@app.route('/')
def health_check():
    return "API do assistente especialista (versão final com cascata) está no ar!"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    # --- LÓGICA DE CASCATA IMPLEMENTADA ---
    
    # 1. Tenta responder usando o manual primeiro.
    print(f"Tentando responder '{pergunta_atual}' com o manual...")
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, CONTEUDO_MANUAL, "Manual Técnico")
    
    # 2. Verifica se a resposta do manual foi a mensagem de falha.
    if "não encontrei informações sobre isso" in resposta_final.lower():
        print("Resposta não encontrada no manual. Partindo para a busca na web.")
        # Se foi, busca na web e gera uma nova resposta.
        contexto_web = buscar_na_web(pergunta_atual)
        resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_web, "Web")
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)