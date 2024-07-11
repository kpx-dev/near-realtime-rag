from dotenv import load_dotenv
load_dotenv()
import redis
import os 
# import numpy as np
# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import BedrockEmbeddings

def main(): 
    memdb_client = redis.Redis(
        host=os.environ.get('MEMDB_HOST'), # ex: 'mycluster.memorydb.us-east-1.amazonaws.com',
        port=6379,
        decode_responses=True,
        ssl=True,
        ssl_cert_reqs="none"
    )
    try:
        memdb_client.ping()
        print("Connection to MemoryDB successful")
    except Exception as e:
        print("An error occurred while connecting to MemDB:", e)

    print('hello')

if __name__ == '__main__':
    main()