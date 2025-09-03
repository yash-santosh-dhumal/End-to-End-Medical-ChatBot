from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medibot"

#Create index only if it does NOT exist
# if index_name not in pc.list_indexes().names():
pc.create_index(
    name=index_name,
    dimension=384,   # must match your embedding model output size
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1",
    )
)


# Embed each chunk and upsert the embeddings into your Pinecone index.
#FIRST_RUN = False   # 👈 change this to True only once when adding docs

# if FIRST_RUN:
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
# else:
#     docsearch = PineconeVectorStore.from_existing_index(
#         index_name=index_name,
#         embedding=embeddings
#     )

