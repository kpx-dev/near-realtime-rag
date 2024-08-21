# near-realtime-rag

Near Realtime RAG solution leveraging Bedrock KB + (MemoryDB or OpenSearch Serverless) for semantic caching.

This repo will evaluate 2 AWS DB: 1/ MemoryDB and 2/ OpenSearch Serverless.

## Best Practice

1. [Prompt Caching](https://www.anthropic.com/news/prompt-caching)
2. [Semantic Cache](https://aws.amazon.com/blogs/database/improve-speed-and-reduce-cost-for-generative-ai-workloads-with-a-persistent-semantic-cache-in-amazon-memorydb/)

## Getting Started

1. Create OpenSearch Serverless with Vector: https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless-vector-search.html
2. Create MemoryDB: https://docs.aws.amazon.com/memorydb/latest/devguide/getting-started.createcluster.html

## Connecting to MemoryDB from your laptop

There are several ways, but SSH Tunnel seem to be the easiest to test. You want to launch an EC2 in a Public Subnet and use it as a jumpbox to SSH tunneling into MemoryDB. Other methods: https://docs.aws.amazon.com/memorydb/latest/devguide/accessing-memorydb.html

```bash 
ssh -L 6379:clustercfg.your-cluster-endpoint.xyz.memorydb.us-east-1.amazonaws.com:6379 ec2-user@IP-of-your-public-EC2-Jumpbox -i ~/.ssh/ec2-keypair.pem
```

## Install Required Packages

```bash
# activate virtualenv
virtualenv env
source env/bin/activate

# install required Python packages
pip install -r requirements.txt
```

## Run the demo

```bash
# copy the required .env file and update its key / value: 
cp 1-memdb/env-sample 1-memdb/.env
cp 2-opensearch/env-sample 2-opensearch/.env


# start the MemDB tunnel (replace cluster DNS and EC2 IP with your own):
ssh -L 6379:clustercfg.your-cluster-endpoint.xyz.memorydb.us-east-1.amazonaws.com:6379 ec2-user@IP-of-your-public-EC2-Jumpbox -i ~/.ssh/ec2-keypair.pem

# start the MemDB demo:
python 1-memdb/memdb-cache-demo.py 

# start the OpenSearch demo:
python 2-opensearch/opensearch-cache-demo.py
```