import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
import re

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
print("Iniciando a configuração do servidor (VERSÃO FINAL COMPLETA)...")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
CONTEUDO_MANUAL = ""
CHUNKS_MANUAL = []

try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Cliente da API da Groq configurado.")
except Exception as e:
    print(f"ERRO: Chave da API da Groq não encontrada. Configure a variável de ambiente. Erro: {e}")
    client = None

def dividir_em_chunks(texto, tamanho_chunk=500):
    """Divide o texto em chunks menores baseados em parágrafos."""
    paragrafos = texto.split('\n\n')
    chunks = []
    chunk_atual = ""

    for paragrafo in paragrafos:
        if len(chunk_atual) + len(paragrafo) < tamanho_chunk:
            chunk_atual += paragrafo + "\n\n"
        else:
            if chunk_atual:
                chunks.append(chunk_atual.strip())
            chunk_atual = paragrafo + "\n\n"

    if chunk_atual:
        chunks.append(chunk_atual.strip())

    return chunks

def detectar_idioma(texto):
    """Detecta o idioma do texto (pt, es, en) com alta precisão."""
    texto_lower = texto.lower()
    texto_com_espacos = f' {texto_lower} '

    # Padrões EXCLUSIVOS de cada idioma (peso 5)
    exclusivos_es = ['el ', ' la ', ' del ', ' al ', ' los ', ' las ', ' es ', ' son ', 'cuánto', 'cuanto', 'cómo', 'qué', 'que ', 'está', 'cuál', 'cual']
    exclusivos_pt = [' o ', ' a ', ' do ', ' da ', ' ao ', ' os ', ' as ', ' não', ' nao', ' são', ' sao', ' tem ', ' qual ', ' você', ' voce', ' é ']
    exclusivos_en = [' the ', ' does ', ' which ', ' that ', ' this ', ' these ', ' those ', ' have ', ' has ', ' is ', ' are ']

    # Verbos típicos (peso 3)
    verbos_es = ['cuesta', 'hacer', 'configurar', 'tiene', 'es']
    verbos_pt = ['custa', 'fazer', 'configurar', 'tem', 'é']
    verbos_en = ['cost', 'costs', 'make', 'configure', 'has', 'have', 'is', 'are']

    # Palavras comuns (peso 1)
    comuns_pt = ['para', 'com', 'em', 'de', 'como', 'por']
    comuns_es = ['para', 'con', 'en', 'de', 'como', 'por']
    comuns_en = ['to', 'for', 'in', 'of', 'how', 'with']

    # Inicializa scores
    score_pt = 0
    score_es = 0
    score_en = 0

    # Conta exclusivos (peso alto)
    score_es += sum(5 for palavra in exclusivos_es if palavra in texto_com_espacos)
    score_pt += sum(5 for palavra in exclusivos_pt if palavra in texto_com_espacos)
    score_en += sum(5 for palavra in exclusivos_en if palavra in texto_com_espacos)

    # Conta verbos (peso médio)
    score_es += sum(3 for verbo in verbos_es if verbo in texto_lower)
    score_pt += sum(3 for verbo in verbos_pt if verbo in texto_lower)
    score_en += sum(3 for verbo in verbos_en if verbo in texto_lower)

    # Conta palavras comuns (peso baixo)
    score_pt += sum(1 for palavra in comuns_pt if palavra in texto_com_espacos)
    score_es += sum(1 for palavra in comuns_es if palavra in texto_com_espacos)
    score_en += sum(1 for palavra in comuns_en if palavra in texto_com_espacos)

    # Regras de desempate específicas
    # Se tem artigos "el" ou "la" sozinhos, muito provável ser espanhol
    if ' el ' in texto_com_espacos or ' los ' in texto_com_espacos:
        score_es += 3
    if ' del ' in texto_com_espacos or ' al ' in texto_com_espacos:
        score_es += 3

    # Se tem "the", muito provável ser inglês
    if ' the ' in texto_com_espacos:
        score_en += 3

    # Se tem "o" ou "a" sozinhos seguidos de substantivo, provável ser português
    if ' o ' in texto_com_espacos or ' os ' in texto_com_espacos:
        score_pt += 2
    if ' do ' in texto_com_espacos or ' da ' in texto_com_espacos or ' ao ' in texto_com_espacos:
        score_pt += 3

    # Retorna o idioma com maior score
    if score_es > score_pt and score_es > score_en:
        return 'es'
    elif score_en > score_pt and score_en > score_es:
        return 'en'
    else:
        return 'pt'  # Padrão é português

def verificar_pergunta_sobre_valores(pergunta):
    """Verifica se a pergunta é sobre valores/preços/licença em qualquer idioma."""
    # Português
    palavras_pt = ['valor', 'preco', 'preço', 'quanto custa', 'custa', 'custo',
                   'licenca', 'licença', 'plano', 'planos', 'mensalidade',
                   'assinatura', 'pagar', 'pagamento', 'reais', 'r$']
    # Espanhol
    palavras_es = ['valor', 'precio', 'cuánto cuesta', 'cuesta', 'costo',
                   'licencia', 'plan', 'planes', 'mensualidad',
                   'suscripción', 'pagar', 'pago']
    # Inglês
    palavras_en = ['value', 'price', 'how much', 'cost', 'costs',
                   'license', 'plan', 'plans', 'monthly', 'subscription',
                   'pay', 'payment', 'pricing']

    pergunta_lower = pergunta.lower()
    todas_palavras = palavras_pt + palavras_es + palavras_en
    return any(palavra in pergunta_lower for palavra in todas_palavras)

def obter_resposta_valores(idioma):
    """Retorna a resposta sobre valores no idioma especificado."""
    respostas = {
        'pt': """Para informações sobre valores, planos e licenças do Console Mix, entre em contato diretamente com nossa equipe de suporte:

📞 Telefones:
• (42) 99985-3754
• (42) 99848-8284

🕒 Horário de Atendimento:
Segunda a Sexta, das 9h às 18h (horário de Brasília)

Nossa equipe terá prazer em apresentar as melhores opções de planos para você!""",

        'es': """Para información sobre precios, planes y licencias de Console Mix, póngase en contacto directamente con nuestro equipo de soporte:

📞 Teléfonos:
• (42) 99985-3754
• (42) 99848-8284

🕒 Horario de Atención:
Lunes a Viernes, de 9h a 18h (horario de Brasilia)

¡Nuestro equipo estará encantado de presentarle las mejores opciones de planes para usted!""",

        'en': """For information about pricing, plans and licenses for Console Mix, please contact our support team directly:

📞 Phone numbers:
• (42) 99985-3754
• (42) 99848-8284

🕒 Business Hours:
Monday to Friday, 9am to 6pm (Brasilia time)

Our team will be happy to present you with the best plan options!"""
    }
    return respostas.get(idioma, respostas['pt'])

def encontrar_chunks_relevantes(pergunta, chunks, top_k=3):
    """Encontra os chunks mais relevantes baseado em palavras-chave simples."""
    # Normaliza a pergunta
    palavras_pergunta = set(re.findall(r'\w+', pergunta.lower()))

    # Remove palavras muito comuns (stop words simples)
    stop_words = {'o', 'a', 'de', 'da', 'do', 'em', 'para', 'com', 'um', 'uma', 'os', 'as', 'dos', 'das', 'é', 'e', 'ou'}
    palavras_pergunta = palavras_pergunta - stop_words

    # Calcula score de relevância para cada chunk
    scores = []
    for idx, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for palavra in palavras_pergunta if palavra in chunk_lower)
        scores.append((score, idx, chunk))

    # Ordena por score e retorna os top_k
    scores.sort(reverse=True, key=lambda x: x[0])

    # Se o melhor score for 0, retorna string vazia para acionar fallback
    if not scores or scores[0][0] == 0:
        return ""

    chunks_relevantes = [chunk for score, idx, chunk in scores[:top_k] if score > 0]
    return "\n\n---\n\n".join(chunks_relevantes)

if os.path.exists(NOME_MANUAL_LIMPO):
    with open(NOME_MANUAL_LIMPO, "r", encoding="utf-8") as f:
        CONTEUDO_MANUAL = f.read()
    CHUNKS_MANUAL = dividir_em_chunks(CONTEUDO_MANUAL, tamanho_chunk=500)
    print(f"Manual '{NOME_MANUAL_LIMPO}' carregado e dividido em {len(CHUNKS_MANUAL)} chunks.")
else:
    print(f"AVISO: Arquivo de manual '{NOME_MANUAL_LIMPO}' não encontrado.")


# --- FUNÇÕES DE LÓGICA DA IA (O CÉREBRO) ---

def buscar_na_web(pergunta):
    """Função para buscar na web e extrair o conteúdo principal de um artigo."""
    print(f"[WEB SEARCH] Iniciando busca na web para: '{pergunta}'")
    query = pergunta
    try:
        with DDGS() as ddgs:
            # Busca até 3 resultados para ter backup
            resultados_links = list(ddgs.text(query, max_results=3, region='br-pt'))
            if not resultados_links:
                print("[WEB SEARCH] Nenhum resultado encontrado no DuckDuckGo")
                return None

            print(f"[WEB SEARCH] Encontrados {len(resultados_links)} resultados")

            # Tenta extrair conteúdo de cada resultado até conseguir
            for i, resultado in enumerate(resultados_links):
                url = resultado['href']
                print(f"[WEB SEARCH] Tentativa {i+1}/{len(resultados_links)}: Extraindo de {url}")

                try:
                    downloaded = fetch_url(url)
                    if downloaded:
                        texto_artigo = extract(downloaded, include_comments=False, include_tables=False)

                        # Verifica se extraiu conteúdo útil (mínimo 100 caracteres)
                        if texto_artigo and len(texto_artigo.strip()) > 100:
                            print(f"[WEB SEARCH] Sucesso! Extraídos {len(texto_artigo)} caracteres")
                            return texto_artigo
                        else:
                            print(f"[WEB SEARCH] Conteúdo muito curto ou vazio, tentando próximo...")
                    else:
                        print(f"[WEB SEARCH] Falha no download, tentando próximo...")
                except Exception as e:
                    print(f"[WEB SEARCH] Erro ao processar {url}: {e}")
                    continue

            print("[WEB SEARCH] Não foi possível extrair conteúdo útil de nenhum resultado")
            return None

    except Exception as e:
        print(f"[WEB SEARCH] Erro na busca: {e}")
        return None

def obter_resposta_generativa(pergunta_atual, historico, contexto, fonte_do_contexto, idioma='pt'):
    """Gera uma resposta da IA baseada no contexto e histórico fornecidos."""
    if not client:
        return "O serviço de IA não está configurado."
    if not contexto:
        mensagens_falha = {
            'pt': "Não encontrei informações sobre isso na fonte consultada.",
            'es': "No encontré información sobre esto en la fuente consultada.",
            'en': "I didn't find information about this in the consulted source."
        }
        return mensagens_falha.get(idioma, mensagens_falha['pt'])

    historico_recente = historico[-6:]
    historico_formatado = "\n".join([f"Usuário: {msg['content']}" if msg['role'] == 'user' else f"Assistente: {msg['content']}" for msg in historico_recente])

    # Instruções de idioma
    instrucoes_idioma = {
        'pt': "TODA RESPOSTA DEVE SER EM PORTUGUÊS.",
        'es': "TODA RESPUESTA DEBE SER EN ESPAÑOL.",
        'en': "ALL RESPONSES MUST BE IN ENGLISH."
    }

    instrucao_idioma = instrucoes_idioma.get(idioma, instrucoes_idioma['pt'])

    prompt_completo = f"""
    Você é um assistente técnico especialista. Responda a PERGUNTA ATUAL do usuário baseando-se exclusivamente no CONTEXTO DE CONSULTA.

    REGRAS ESTRITAS:
    1.  Use o HISTÓRICO DA CONVERSA para entender perguntas de acompanhamento.
    2.  Sua resposta deve vir APENAS do CONTEXTO DE CONSULTA. Não use conhecimento prévio.
    3.  Seja direto e não mencione o contexto ou a fonte. NÃO diga coisas do tipo "a fonte que consultei".
    4.  REGRA DE FALHA: Se a resposta não estiver no CONTEXTO DE CONSULTA, responda APENAS com a frase: "Não encontrei informações sobre isso na fonte consultada."
    5.  NÃO CORRIJA a ortografia do usuário NEM RESPONDA QUESTÕES DE GRAMATICA. VOCE É UM ASSISTENTE TÉCNICO, não um PROFESSOR.
    6.  {instrucao_idioma}

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

    # --- DETECÇÃO DE IDIOMA ---
    idioma_detectado = detectar_idioma(pergunta_atual)
    print(f"Idioma detectado: {idioma_detectado}")

    # --- VERIFICAÇÃO DE PERGUNTAS SOBRE VALORES ---
    if verificar_pergunta_sobre_valores(pergunta_atual):
        print(f"Pergunta sobre valores detectada: '{pergunta_atual}'")
        resposta_final = obter_resposta_valores(idioma_detectado)
        return jsonify({"answer": resposta_final})

    # --- LÓGICA DE CASCATA IMPLEMENTADA COM CHUNKING ---

    # 1. Encontra os chunks mais relevantes do manual baseado na pergunta
    print(f"Tentando responder '{pergunta_atual}' com o manual...")
    contexto_manual = encontrar_chunks_relevantes(pergunta_atual, CHUNKS_MANUAL, top_k=3)

    # 2. Se encontrou chunks relevantes, tenta responder com eles
    if contexto_manual:
        resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_manual, "Manual Técnico", idioma_detectado)
    else:
        # Se não encontrou nenhum chunk relevante, marca para buscar na web
        mensagens_falha = {
            'pt': "Não encontrei informações sobre isso na fonte consultada.",
            'es': "No encontré información sobre esto en la fuente consultada.",
            'en': "I didn't find information about this in the consulted source."
        }
        resposta_final = mensagens_falha.get(idioma_detectado, mensagens_falha['pt'])

    # 3. Verifica se a resposta do manual foi a mensagem de falha.
    if any(msg in resposta_final.lower() for msg in ["não encontrei", "no encontré", "didn't find"]):
        print("[FALLBACK] Resposta não encontrada no manual. Partindo para a busca na web.")
        # Se foi, busca na web e gera uma nova resposta.
        contexto_web = buscar_na_web(pergunta_atual)

        if contexto_web:
            print(f"[FALLBACK] Contexto da web obtido com sucesso ({len(contexto_web)} caracteres)")
            resposta_final = obter_resposta_generativa(pergunta_atual, historico, contexto_web, "Web", idioma_detectado)
        else:
            print(f"[FALLBACK] Busca na web falhou. Idioma detectado: {idioma_detectado}")
            print(f"[FALLBACK] Pergunta original: '{pergunta_atual}'")

            # Re-detecta idioma para garantir precisão
            idioma_final = detectar_idioma(pergunta_atual)
            print(f"[FALLBACK] Idioma re-detectado: {idioma_final}")

            # Mensagens quando a busca web falha
            mensagens_falha_web = {
                'pt': "Desculpe, não encontrei informações sobre isso no manual e também não consegui buscar na internet no momento. Por favor, tente reformular sua pergunta ou entre em contato com o suporte.",
                'es': "Lo siento, no encontré información sobre esto en el manual y tampoco pude buscar en Internet en este momento. Por favor, intente reformular su pregunta o póngase en contacto con el soporte.",
                'en': "Sorry, I couldn't find information about this in the manual and I was unable to search the internet at this time. Please try rephrasing your question or contact support."
            }
            resposta_final = mensagens_falha_web.get(idioma_final, mensagens_falha_web['pt'])
            print(f"[FALLBACK] Mensagem selecionada para idioma '{idioma_final}'")


    return jsonify({"answer": resposta_final})

if __name__ == '__main__':
    app.run(debug=True, port=5000)