import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO FINAL COM CORS EXPLÍCITO)...")

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


# --- FUNÇÃO GENERATIVA ---
def obter_resposta_generativa(pergunta_atual, historico, contexto):
    if not client: 
        return "O serviço de IA não está configurado corretamente."
    if not CONTEUDO_MANUAL:
        return "Desculpe, a base de conhecimento (manual) não foi carregada no servidor."

    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Sua única função é responder a PERGUNTA ATUAL do usuário baseando-se exclusivamente no MANUAL TÉCNICO COMPLETO fornecido.

    REGRAS ESTRITAS E ABSOLUTAS:
    1.  Leia todo o MANUAL TÉCNICO COMPLETO para encontrar a informação relevante. Use o HISTÓRICO DA CONVERSA para entender perguntas de acompanhamento.
    2.  Responda de forma direta e objetiva, como se você fosse o especialista que sabe a informação, sem mencionar que consultou um manual.
    3.  REGRA DE FALHA: Se, após ler todo o manual, a resposta não estiver lá, responda APENAS com a frase: "Não encontrei informações sobre isso no manual."

    ---
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    ---
    MANUAL TÉCNICO COMPLETO:
    {CONTEUDO_MANUAL}
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

# --- CORREÇÃO DEFINITIVA DE CORS ---
# Estamos dizendo explicitamente quais domínios podem acessar nossa API.
origins_permitidas = [
    "https://consolemix.com.br",
    "http://consolemix.com.br",
    "http://localhost",
    "http://127.0.0.1"
]
CORS(app, resources={r"/ask": {"origins": origins_permitidas}})
# --- FIM DA CORREÇÃO ---


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
    
    # A lógica foi simplificada: a IA sempre tentará usar o manual.
    # O prompt instrui a IA sobre o que fazer se a resposta não estiver lá.
    resposta_final = obter_resposta_generativa(pergunta_atual, historico, CONTEUDO_MANUAL)
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)