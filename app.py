import streamlit as st
import faiss
import numpy as np
import os
import pickle
from groq import Groq
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract
import time
import re

# --- Configurações Iniciais ---
st.set_page_config(page_title="Assistente Especialista IA", page_icon="🧠")

NOME_MANUAL_LIMPO = "manual_limpo.txt"
NOME_ARQUIVO_INDICE = "indice_faiss.bin"
NOME_ARQUIVO_CHUNKS = "chunks.pkl"
MODELO_EMBEDDING = 'paraphrase-multilingual-MiniLM-L12-v2'

# Configura o cliente da API da Groq
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error("Chave de API da Groq não encontrada. Por favor, configure o arquivo .streamlit/secrets.toml")
    st.stop()


# --- Funções de Cache e Busca ---

def detectar_idioma(texto):
    """Detecta o idioma do texto (pt, es, en) com alta precisão."""
    texto_lower = texto.lower()
    texto_com_espacos = f' {texto_lower} '

    # Padrões EXCLUSIVOS de cada idioma (peso 5)
    exclusivos_es = ['el ', ' la ', ' del ', ' al ', ' los ', ' las ', 'cuánto', 'cuanto', 'cómo', 'qué', 'está', 'cuál', 'cual']
    exclusivos_pt = [' o ', ' a ', ' do ', ' da ', ' ao ', ' os ', ' as ', ' não', ' nao', ' são', ' sao', ' tem ', ' qual ', ' você', ' voce']
    exclusivos_en = [' the ', ' does ', ' which ', ' that ', ' this ', ' these ', ' those ', ' have ', ' has ']

    # Verbos típicos (peso 3)
    verbos_es = ['cuesta', 'hacer', 'configurar', 'tiene']
    verbos_pt = ['custa', 'fazer', 'configurar', 'tem']
    verbos_en = ['cost', 'costs', 'make', 'configure', 'has', 'have']

    # Palavras comuns (peso 1)
    comuns_pt = ['para', 'com', 'em', 'de', 'como', 'que', 'por']
    comuns_es = ['para', 'con', 'en', 'de', 'como', 'que', 'por']
    comuns_en = ['to', 'for', 'in', 'of', 'how', 'what', 'with']

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

🌐 Site:
consolemix.com.br/console

Nossa equipe terá prazer em apresentar as melhores opções de planos para você!""",

        'es': """Para información sobre precios, planes y licencias de Console Mix, póngase en contacto directamente con nuestro equipo de soporte:

📞 Teléfonos:
• (42) 99985-3754
• (42) 99848-8284

🕒 Horario de Atención:
Lunes a Viernes, de 9h a 18h (horario de Brasilia)

🌐 Sitio web:
consolemix.com.br/console

¡Nuestro equipo estará encantado de presentarle las mejores opciones de planes para usted!""",

        'en': """For information about pricing, plans and licenses for Console Mix, please contact our support team directly:

📞 Phone numbers:
• (42) 99985-3754
• (42) 99848-8284

🕒 Business Hours:
Monday to Friday, 9am to 6pm (Brasilia time)

🌐 Website:
consolemix.com.br/console

Our team will be happy to present you with the best plan options!"""
    }
    return respostas.get(idioma, respostas['pt'])

@st.cache_resource
def carregar_modelo_embedding():
    return SentenceTransformer(MODELO_EMBEDDING)

@st.cache_data
def carregar_recursos_busca(_timestamp):
    if os.path.exists(NOME_ARQUIVO_INDICE) and os.path.exists(NOME_ARQUIVO_CHUNKS):
        indice_faiss = faiss.read_index(NOME_ARQUIVO_INDICE)
        with open(NOME_ARQUIVO_CHUNKS, 'rb') as f:
            chunks = pickle.load(f)
        return indice_faiss, chunks
    return None, None

def buscar_contexto_local(pergunta, modelo_emb, indice, chunks, top_k=3):
    if indice is None: return ""
    pergunta_embedding = modelo_emb.encode([pergunta], normalize_embeddings=True)
    distancias, indices = indice.search(pergunta_embedding.astype('float32'), top_k)
    
    if distancias[0][0] > 1.0: 
        return ""

    contexto = "\n\n---\n\n".join([chunks[i] for i in indices[0]])
    return contexto

@st.cache_data
def buscar_na_web(pergunta, num_artigos=3):
    query = f"{pergunta}"
    contextos_web, urls_usadas = [], []
    try:
        with DDGS() as ddgs:
            resultados_links = list(ddgs.text(query, max_results=5, region='br-pt'))
            if not resultados_links:
                st.warning("Nenhum resultado encontrado no DuckDuckGo")
                return None, []

            for i, link in enumerate(resultados_links[:num_artigos]):
                url = link['href']
                st.write(f"Lendo artigo {i+1}: {url}")
                try:
                    downloaded = fetch_url(url)
                    if downloaded:
                        texto_artigo = extract(downloaded, include_comments=False, include_tables=False)
                        if texto_artigo and len(texto_artigo) > 200:
                            contexto_formatado = f"Fonte {i+1}: {link['title']}\nURL: {url}\nCONTEÚDO: {texto_artigo[:2500]}..."
                            contextos_web.append(contexto_formatado)
                            urls_usadas.append(link['href'])
                except Exception as e:
                    st.warning(f"Erro ao processar {url}: {str(e)}")
                    continue
                time.sleep(0.2)

            if not contextos_web:
                st.warning("Não foi possível extrair conteúdo útil de nenhum resultado")
                return None, []

            return "\n\n===\n\n".join(contextos_web), urls_usadas
    except Exception as e:
        st.error(f"Erro na busca web: {str(e)}")
        return None, []


# --- Funções do "Especialista" Generativo ---

# --- NOVA FUNÇÃO: O "INSPETOR DE QUALIDADE" ---
@st.cache_data
def contexto_e_relevante(pergunta, contexto):
    if not contexto:
        return False
    
    prompt = f"""
    Analise a PERGUNTA e o CONTEXTO a seguir. O contexto contém informação suficiente para responder a pergunta de forma satisfatória?
    Responda APENAS com a palavra 'SIM' ou 'NÃO'.

    PERGUNTA: "{pergunta}"

    CONTEXTO: "{contexto}"
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0, # Baixa temperatura para respostas mais diretas
        )
        resposta = chat_completion.choices[0].message.content.strip().upper()
        return "SIM" in resposta
    except Exception:
        return False


def obter_resposta_generativa(pergunta, contexto, fonte, idioma='pt'):
    # Instruções de idioma
    instrucoes_idioma = {
        'pt': "Leia o contexto e sintetize uma resposta clara e útil em português.",
        'es': "Lea el contexto y sintetice una respuesta clara y útil en español.",
        'en': "Read the context and synthesize a clear and helpful response in English."
    }

    mensagens_falha = {
        'pt': "Não encontrei uma resposta para isso.",
        'es': "No encontré una respuesta para esto.",
        'en': "I didn't find an answer for this."
    }

    instrucao = instrucoes_idioma.get(idioma, instrucoes_idioma['pt'])
    msg_falha = mensagens_falha.get(idioma, mensagens_falha['pt'])

    prompt = f"""
    Aja como um assistente técnico especialista. Sua tarefa é responder a PERGUNTA do usuário.
    A resposta DEVE ser baseada exclusivamente no CONTEXTO fornecido, que veio da fonte: '{fonte}'.
    - {instrucao}
    - Não invente informações. Se a resposta não estiver no contexto, diga "{msg_falha}".

    ---
    CONTEXTO:
    {contexto}
    ---
    PERGUNTA DO USUÁRIO:
    {pergunta}
    ---
    RESPOSTA DETALHADA:
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content


# --- APLICAÇÃO STREAMLIT ---
st.title("🧠 Assistente de Pesquisa Especialista")
st.caption("Com tomada de decisão inteligente sobre as fontes de informação")

# Carrega os recursos
modelo_embedding_instance = carregar_modelo_embedding()
indice_faiss, chunks = (None, None)
if os.path.exists(NOME_MANUAL_LIMPO):
    file_timestamp = os.path.getmtime(NOME_MANUAL_LIMPO)
    indice_faiss, chunks = carregar_recursos_busca(file_timestamp)

# Lógica do Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou seu assistente especialista. Eu verifico seu manual e, se necessário, pesquiso a fundo na web. Como posso ajudar?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Qual é a sua dúvida sobre áudio?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resposta_final = ""
        urls_usadas_na_resposta = []

        # --- DETECÇÃO DE IDIOMA ---
        idioma_detectado = detectar_idioma(prompt)

        # --- VERIFICAÇÃO DE PERGUNTAS SOBRE VALORES ---
        if verificar_pergunta_sobre_valores(prompt):
            resposta_final = obter_resposta_valores(idioma_detectado)
            st.markdown(resposta_final)
        else:
            with st.spinner("Consultando o manual..."):
                contexto_manual = buscar_contexto_local(prompt, modelo_embedding_instance, indice_faiss, chunks)

            # --- NOVA LÓGICA DE DECISÃO INTELIGENTE ---
            if contexto_e_relevante(prompt, contexto_manual):
                fonte_usada = "Manual Técnico"
                contexto_final = contexto_manual
                with st.expander(f"🔬 Fonte Utilizada ({fonte_usada})"):
                    st.text(contexto_final)

                with st.spinner("Encontrei uma resposta relevante no manual! Gerando..."):
                    resposta_final = obter_resposta_generativa(prompt, contexto_final, fonte_usada, idioma_detectado)
            else:
                with st.spinner("O manual não foi suficiente. Iniciando pesquisa aprofundada na web..."):
                    contexto_web, urls_usadas_na_resposta = buscar_na_web(prompt)

                if contexto_web:
                    fonte_usada = "Web (Visão Geral de Múltiplos Artigos)"

                    with st.expander(f"🔬 Fonte Utilizada ({fonte_usada})"):
                        st.text(contexto_web)

                    with st.spinner("O especialista está analisando os artigos e pensando na resposta..."):
                        resposta_final = obter_resposta_generativa(prompt, contexto_web, fonte_usada, idioma_detectado)
                else:
                    # Busca web falhou
                    mensagens_falha_web = {
                        'pt': "Desculpe, não encontrei informações sobre isso no manual e também não consegui buscar na internet no momento. Por favor, tente reformular sua pergunta ou entre em contato com o suporte.",
                        'es': "Lo siento, no encontré información sobre esto en el manual y tampoco pude buscar en Internet en este momento. Por favor, intente reformular su pregunta o póngase en contacto con el soporte.",
                        'en': "Sorry, I couldn't find information about this in the manual and I was unable to search the internet at this time. Please try rephrasing your question or contact support."
                    }
                    resposta_final = mensagens_falha_web.get(idioma_detectado, mensagens_falha_web['pt'])

            if urls_usadas_na_resposta:
                fontes_formatadas = "\n\n---\n*Fontes da web consultadas:*\n"
                for url in urls_usadas_na_resposta:
                    fontes_formatadas += f"- {url}\n"
                resposta_final += fontes_formatadas

            st.markdown(resposta_final)

    st.session_state.messages.append({"role": "assistant", "content": resposta_final})