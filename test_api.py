import requests
import json

# URL onde nossa API local está rodando
API_URL = "http://127.0.0.1:5000/ask"

# A pergunta que queremos fazer para a nossa IA
pergunta_teste = "O que é um compressor de áudio?" 
# (ou qualquer outra pergunta que você queira testar)

# Monta o cabeçalho e o corpo da requisição
headers = {
    "Content-Type": "application/json"
}
data = {
    "question": pergunta_teste
}

print(f"Enviando pergunta para a API: '{pergunta_teste}'")

try:
    # Faz a requisição POST para a nossa API
    response = requests.post(API_URL, headers=headers, json=data)

    # Verifica se a requisição foi bem-sucedida (código 200)
    if response.status_code == 200:
        print("\n--- Resposta da IA Recebida com Sucesso! ---")
        # Imprime a resposta JSON formatada
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"\n--- Erro ao Chamar a API ---")
        print(f"Status Code: {response.status_code}")
        print(f"Resposta: {response.text}")

except requests.exceptions.ConnectionError as e:
    print("\n--- ERRO DE CONEXÃO ---")
    print("Não foi possível conectar à API. Você se lembrou de deixar o servidor 'api.py' rodando no outro terminal?")