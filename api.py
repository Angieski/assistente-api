import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO ULTRA-LEVE)...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
CONTEUDO_MANUAL = ""

try:
    # Carrega a chave de API das variáveis de ambiente do servidor (Render)
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


# --- FUNÇÃO GENERATIVA ÚNICA ---
def obter_resposta_generativa(pergunta_atual, historico):
    if not client: 
        return "O serviço de IA não está configurado corretamente (sem chave de API)."
    if not CONTEUDO_MANUAL:
        # Se não houver manual, ele não pode responder.
        return "Desculpe, a base de conhecimento (manual) não foi carregada no servidor."

    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico])
    
    prompt_completo = f"""
    Você é um assistente técnico especialista. Sua única função é responder a PERGUNTA ATUAL do usuário baseando-se exclusivamente no MANUAL TÉCNICO COMPLETO fornecido.

    REGRAS ESTRITAS E ABSOLUTAS:
    1.  Leia todo o MANUAL TÉCNICO COMPLETO para encontrar a informação relevante. Use o HISTÓRICO DA CONVERSA para entender perguntas de acompanhamento (ex: "e sobre ele?").
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
CORS(app)

@app.route('/')
def health_check():
    return "API do assistente especialista (versão ultra-leve) está no ar!"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "A pergunta (question) é obrigatória."}), 400

    pergunta_atual = data['question']
    historico = data.get('history', [])
    
    resposta_final = obter_resposta_generativa(pergunta_atual, historico)
        
    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)