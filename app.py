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
            if not resultados_links: return "Nenhum resultado encontrado na web.", []
            
            for i, link in enumerate(resultados_links[:num_artigos]):
                url = link['href']
                st.write(f"Lendo artigo {i+1}: {url}")
                downloaded = fetch_url(url)
                if downloaded:
                    texto_artigo = extract(downloaded, include_comments=False, include_tables=False)
                    if texto_artigo and len(texto_artigo) > 200:
                        contexto_formatado = f"Fonte {i+1}: {link['title']}\nURL: {url}\nCONTEÚDO: {texto_artigo[:2500]}..."
                        contextos_web.append(contexto_formatado)
                        urls_usadas.append(link['href'])
                time.sleep(0.2)
            if not contextos_web: return "Não foi possível extrair conteúdo útil.", []
            return "\n\n===\n\n".join(contextos_web), urls_usadas
    except Exception: return "Não foi possível buscar na web.", []


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


def obter_resposta_generativa(pergunta, contexto, fonte):
    prompt = f"""
    Aja como um assistente técnico especialista. Sua tarefa é responder a PERGUNTA do usuário.
    A resposta DEVE ser baseada exclusivamente no CONTEXTO fornecido, que veio da fonte: '{fonte}'.
    - Leia o contexto e sintetize uma resposta clara e útil em português.
    - Não invente informações. Se a resposta não estiver no contexto, diga "Não encontrei uma resposta para isso.".

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

        with st.spinner("Consultando o manual..."):
            contexto_manual = buscar_contexto_local(prompt, modelo_embedding_instance, indice_faiss, chunks)

        # --- NOVA LÓGICA DE DECISÃO INTELIGENTE ---
        if contexto_e_relevante(prompt, contexto_manual):
            fonte_usada = "Manual Técnico"
            contexto_final = contexto_manual
            with st.expander(f"🔬 Fonte Utilizada ({fonte_usada})"):
                st.text(contexto_final)
            
            with st.spinner("Encontrei uma resposta relevante no manual! Gerando..."):
                resposta_final = obter_resposta_generativa(prompt, contexto_final, fonte_usada)
        else:
            with st.spinner("O manual não foi suficiente. Iniciando pesquisa aprofundada na web..."):
                contexto_web, urls_usadas_na_resposta = buscar_na_web(prompt)
                fonte_usada = "Web (Visão Geral de Múltiplos Artigos)"
            
            with st.expander(f"🔬 Fonte Utilizada ({fonte_usada})"):
                st.text(contexto_web)

            with st.spinner("O especialista está analisando os artigos e pensando na resposta..."):
                resposta_final = obter_resposta_generativa(prompt, contexto_web, fonte_usada)

        if urls_usadas_na_resposta:
            fontes_formatadas = "\n\n---\n*Fontes da web consultadas:*\n"
            for url in urls_usadas_na_resposta:
                fontes_formatadas += f"- {url}\n"
            resposta_final += fontes_formatadas
        
        st.markdown(resposta_final)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta_final})