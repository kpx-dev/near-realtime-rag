from dotenv import load_dotenv
load_dotenv()
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
from requests_aws4auth import AWS4Auth
from datetime import datetime, timedelta
import time
import boto3
import json
import sys
import traceback
import redis
import numpy as np
import sys
import os

import botocore

client = boto3.client('opensearchserverless')
service = 'aoss'
region = 'us-east-1'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
collection_name = "demo-semantic-cache-xyz"
index_name = "cache"
vector_size = 1024
model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
# model_id = 'anthropic.claude-3-haiku-20240307-v1:0'

# Load the local embedding model: SentenceTransformer model: 768 dim
# model_name = 'sentence-transformers/msmarco-distilbert-base-tas-b'
# model = SentenceTransformer(model_name)

bedrock_client = boto3.client('bedrock-runtime',region_name=region)

def get_embedding(text_content, local = False):
    # if local:
    #     return model.encode(text_content).tolist()

    try:
        body_content = json.dumps({"inputText": text_content})
        response = bedrock_client.invoke_model(
            body=body_content,
            contentType="application/json",
            accept="*/*",
            modelId="amazon.titan-embed-text-v2:0"  # 1024 
        )
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding')
        
        return embedding
    except Exception as e:
        print("Error generating embedding: ", e)
        raise e

def delete_index(client): 
    try:
        client.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting index: {e}")

def create_index(client):
    if client.indices.exists(index=index_name): 
        print("Index already exists: ", index_name)
        return

    index_body = {
        "settings": { "index.knn": True },
        'mappings': {
            'properties': {
                "query": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                "v_query": { "type": "knn_vector", "dimension": vector_size },
                "cache_value": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
                "reference_url": {"type":"text","fields":{"keyword":{"type":"keyword","ignore_above":256}}},
            }
        }
    }

    # Create index
    try:
        response = client.indices.create(
            index=index_name,
            body=index_body
        )
        print('\nCreating index:')
        print(response)
    except Exception as e:
        print("Problem creating index ", e)

def add_document(client, doc):
    try: 
        # Add a document to the index.
        response = client.index(
            index=index_name,
            body=doc,
        )
        print('\nDocument added:')
        print(response)
    except Exception as e:
        print("Problem indexing document ", e)

def chat_with_llm(user_query):
    system_prompt = """
    Do not return preamble. 
    Always return data in JSON format using the following fields:
    "content": "content of the answer in single line",
    "reference_url": "The URL used to reference this response"
    """

    messages = [{"role": "user", "content": [{"text": user_query}]}]

    converse_api_params = {
        "modelId": model_id,
        "system": [{"text": system_prompt}],
        "messages": messages,
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 4096},
    }

    response = bedrock_client.converse(**converse_api_params)

    return response["output"]

def get_cache_doc(aoss_client, v_query):
    query = {
        "size": 10,
        "fields": ["cache_value", "reference_url"],
        "query": {
            "knn": {
                "v_query": {
                    "vector": v_query,
                    "k": vector_size
                }
            }
        }
    }

    response = aoss_client.search(
        body=query,
        index=index_name
    )
    # print(json.dumps(response))
    # exit()

    if response["hits"]["hits"]:
        if response["hits"]["max_score"] >= 0.5:
            doc = response["hits"]["hits"][0]["_source"]
            return {"content": doc["cache_value"], "reference_url": doc["reference_url"]} 
    else:
        return None

def connect_opensearch():
    OS_HOST = os.getenv("OS_HOST")

    kwargs = {
        "connection_class": RequestsHttpConnection,
        "http_compress": True,        
        "http_auth": awsauth,
        "use_ssl": True,
        "verify_certs": True,
        "timeout": 15
    }
    
    return OpenSearch(OS_HOST, **kwargs)

def main(): 
    aoss_client = connect_opensearch()
    # delete_index(aoss_client)
    # print(aoss_client.count())
    create_index(aoss_client)

    # query = "How to protect from scam?"
    query = "How to watch Olympics?"               
    v_query = get_embedding(query, local=False) # 1s 

    # check for cache:
    cache = get_cache_doc(aoss_client, v_query)
    if cache:
        return cache 

    answer = chat_with_llm(query)
    answer_dict = json.loads(answer['message']['content'][0]['text'])
    answer_txt = answer_dict['content']
    answer_ref_url = answer_dict['reference_url']

    doc = {
        "query": query,
        "v_query": v_query,
        "cache_value": answer_txt,
        "reference_url": answer_ref_url,
    }
    add_document(aoss_client, doc)
    # print(aoss_client.count())

    return answer_txt

if __name__ == "__main__":
    start_time = time.time()

    res = main()
    print(json.dumps(res, indent=4))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\nThe function took {execution_time:.6f} seconds to execute.\n\n")