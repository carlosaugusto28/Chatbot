import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

# Carregue o modelo Spacy
nlp = spacy.load("pt_core_news_sm")

# Função para processar o texto
def preprocess(text):
    return ' '.join([token.lemma_.lower() for token in nlp(text) if not token.is_stop and not token.is_punct])

# Carregar o JSON com perguntas e respostas
def load_faq(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys()), list(data.values())

# Caminho para o arquivo JSON
FAQ_FILE = "faq.json"

# Carregar perguntas e respostas do JSON
perguntas, respostas = load_faq(FAQ_FILE)

# Pré-processar as perguntas
perguntas_processadas = [preprocess(pergunta) for pergunta in perguntas]

# Criar e treinar o modelo
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, MultinomialNB())
model.fit(perguntas_processadas, respostas)

# Função para encontrar a resposta mais similar
def encontrar_resposta_mais_similar(pergunta_usuario, limiar_similaridade=0.5, resposta_padrao="Desculpe, não entendi a sua pergunta. Pode reformular?"):
    pergunta_usuario_processada = preprocess(pergunta_usuario)
    tfidf_vector = model.named_steps['tfidfvectorizer'].transform([pergunta_usuario_processada])
    similaridades = cosine_similarity(tfidf_vector, model.named_steps['tfidfvectorizer'].transform(perguntas_processadas))
    similaridade_maxima = similaridades.max()
    
    if similaridade_maxima < limiar_similaridade:
        return resposta_padrao
    
    indice_pergunta_similar = similaridades.argmax()
    return respostas[indice_pergunta_similar]

# Exemplo de teste
if __name__ == "__main__":
    while True:
        user_input = input("Faça sua pergunta: ")
        if user_input.lower() in ["sair", "exit"]:
            print("Até mais!")
            break
        resposta = encontrar_resposta_mais_similar(user_input)
        print(f"Resposta: {resposta}")
