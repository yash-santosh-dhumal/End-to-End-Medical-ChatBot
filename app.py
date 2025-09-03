from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()



PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

print("Gemini key loaded:", os.getenv("GEMINI_API_KEY"))

embeddings = download_hugging_face_embeddings()

index_name = "medibot"

# FIRST_RUN = False   # 👈 change this to True only once when adding docs

# if FIRST_RUN:
#     docsearch = PineconeVectorStore.from_documents(
#         documents=text_chunks,
#         embedding=embeddings,
#         index_name=index_name
#     )
# else:
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":8})


llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",   
    temperature=0.4,
    max_output_tokens=500,
    api_key=os.getenv("GEMINI_API_KEY")
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form.get("msg")
        print("Incoming msg:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Response:", response)
        return str(response["answer"])
    except Exception as e:
        print("Error in /get:", e)
        return "Error: " + str(e), 500




if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True) 

