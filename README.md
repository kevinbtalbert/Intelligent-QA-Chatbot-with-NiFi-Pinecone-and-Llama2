# Intelligent QA Chatbot with NiFi, Pinecone, and Llama2
In this Applied ML Prototype (AMP) we leverage Cloudera DataFlow (CDF) to scrape a website and load the vectors generated with Cloudera Machine Learning (CML). The prototype deploys an Application in CML using a Llama2 model from Hugging Face to answer questions augmented with knowledge extracted from the website. This prototype introduces Pinecone as a database for storing vectors for semantic search.

![](/assets/catalog-entry.png)

## AMP Architecture
![](/assets/architecture.png)


## Building your custom knowledge base
To build your own custom knowledge base, you will want to follow the instructions [here](USER_START_HERE/Build_Your_Own_Knowledge_Base_Tools/README.md) in the folder `USER_START_HERE`. There are guides for a CDF and Pythonic implementation of how to do this. Then, you will want to rerun the `Populate Vector DB` Job to ensure your vector DB has the latest embeddings. If you already have documents to fill the knowledge base with, simply copy them directly to the `/data` directory, and rerun the `Populate Vector DB` Job.

## Understanding the User Inputs

### Inputs
**Select Model** - Here the user can select the Llama2 13B parameter chat model (`llama-2-13b-chat`)

**Select Temperature (Randomness of Response)** - Here the user can scale the randomness of the model's response. Lower numbers ensure a more approximate, objective answer while higher numbers encourage model creativity.

**Select Number of Tokens (Length of Response)** - Here several options have been provided. The number of tokens the user uses directly correlate with the length of the response the model returns.

**Topic Weight** - The concept of a topic weight allows the user to further curate a model's response by ensuring it closely aligns to the known context.

**Question** - Just as it sounds; this is where the user can provide a question to the model

### Outputs
**Llama2 Model Response** - This is the response generated by the model given the context in your vector database. Note that if the question cannot correlate to content in your knowledge base, you may get hallucinated responses.

**Context Data Source(s)** - By default, only one data source is provided to the model (this can be adjusted in the project source code). This is provided to help the user understand what data was fed as context to the model and if its response can be trusted given the data source. This supposes good naming conventions for the datasets in `/data/*`

**Pinecone Match Score** *(Pinecone only)*- This can be thought of as a relevancy score. A lower number may indicate there was not much relevant content in the vector database to sustain answering the user question. 

## Two Flavors: UI (Front end) and API
This project allows you to access the context-driven LLM using two flavors: a UI and an API. The Gradio UI is initialized as the running application by default.

### Gradio - UI (Front end)
This is the default application choice for the AMP. You should be able to access the view through your applications nav. When it starts, you will be able to select the default model (`llama-2-13b-chat`), temperature (a good default is 1), number of tokens (a good default maybe 100), topic weight (a domain for the corpus of knowledge to prioritize), and question for the model to process. Defaults will be selected if you choose not to answer these; however, a question is required. 

![](/assets/interface.png)

As a note, Gradio comes with a Python-based API option too, which is accessible from the link shown at the bottom of the Gradio interface.

![](/assets/gradio-api-access.png)

![](/assets/gradio-api-full.png)

### FastAPI - REST API
To enable the FastAPI REST API, a user can either create a new application pointing to `3_app/api_app.py`, with the same resource requirements as the Gradio application (2 CPUs, 16 GB Memory, and 1 GPU), or modify the existing application to point to `3_app/api_app.py` instead. 

#### Forming a request to the API
Requests can be formed intra-domain or cross-domain. For cross-domain requests, you'll need to ensure unauthenticated app access is allowed for the POST endpoint to be reachable. Be cognizant of the amount of tokens and temperature you feed into the payload parameters. Most requests for a couple of sentences should use around 200 tokens, a paragraph could use upwards of 600-800. 

![](/assets/unauthenticated-access1.png)

Note that if this option does not appear, you may need to update your Administrative security settings to mirror the below, where "Allow applications to be configured with unauthenticated access" is checked:

![](/assets/unauthenticated-access2.png)

For request syntax: 
GET and POST to the ROOT of the CML application endpoint (e.g. `https://subdomain.domain.go01-dem.ylcu-atmi.cloudera.site/`)
Successful GET request should indicate the API is up and running:

![](/assets/GET-endpoint.png)

Forming the POST request can be done through Postman or natively in CML:

1. Postman

Form the payload/url and body to match the below, and add the header `Content-Type | application/json`

```
{
    "inputs": "What is Cloudera Machine Learning?",
    "parameters": {
        "temperature": 1,
        "max_tokens": 100
    }
}
```

![](/assets/postman-setup.png)

Note that in future development, the `engine` may also be customized to include more than the Llama2 one which comes with the AMP deployment.

2. Pythonic (Available in the 3_app folder as a Jupyter Notebook)

## Requirements
#### CML Instance Types
- A GPU instance is required to perform inference on the LLM
  - [CML Documentation: GPUs](https://docs.cloudera.com/machine-learning/cloud/gpu/topics/ml-gpu.html)
- A CUDA 5.0+ capable GPU instance type is recommended *(AMP will fail on Step 2 if this requirement is not met)*
  - The torch libraries in this AMP require a GPU with CUDA compute capability 5.0 or higher. (i.e. NVIDIA V100, A100, T4 GPUs)

#### Resource Requirements
This AMP creates the following workloads with resource requirements:
- CML Session: `2 CPU, 16GB MEM`
- CML Jobs: `2 CPU, 8GB MEM`
- CML Application: `2 CPU, 1 GPU, 16GB MEM`

#### External Resources
This AMP requires pip packages and models from huggingface. Depending on your CML networking setup, you may need to whitelist some domains:
- pypi.python.org
- pypi.org
- pythonhosted.org
- huggingface.co
- pinecone.io (if using Pinecone)

## Technologies Used
#### Open-Source Models and Utilities
- [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/sentence-transformers/all-mpnet-base-v2/resolve/main/all-mpnet-base-v2.tar.gz)
     - Vector Embeddings Generation Model
- [llama-2-13b-chat.Q5_0.gguf](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/blob/main/llama-2-13b-chat.Q5_0.gguf)
   - Instruction-following Large Language Model
- [Hugging Face transformers library](https://pypi.org/project/transformers/)
#### Vector Database
- [Milvus](https://github.com/milvus-io/milvus)
- [Pinecone](https://www.pinecone.io/)
#### Chat Frontend
- [Gradio](https://github.com/gradio-app/gradio)
#### Chat API
- [FastAPI](https://fastapi.tiangolo.com/)

## Deploying on CML
Navigate to the Prototype Catalog on a CML workspace, select its tile, click "Launch as Project", and click "Configure Project". Then you can provide the necessary variables to select between Pinecone or Milvus as your vector DB and accept other project level settings.


## The Fine Print

All the components of the application (knowledge base, context retrieval, prompt enhancement LLM) are running within CDF and CML. This application does not call any external model APIs nor require any additional training of an LLM. The knowledge base is generated using the user-passed sitemaps in NiFi (CDF) or Python, depending on the user preference.

IMPORTANT: Please read the following before proceeding.  This AMP includes or otherwise depends on certain third party software packages.  Information about such third party software packages are made available in the notice file associated with this AMP.  By configuring and launching this AMP, you will cause such third party software packages to be downloaded and installed into your environment, in some instances, from third parties' websites.  For each third party software package, please see the notice file and the applicable websites for more information, including the applicable license terms.

If you do not wish to download and install the third party software packages, do not configure, launch or otherwise use this AMP.  By configuring, launching or otherwise using the AMP, you acknowledge the foregoing statement and agree that Cloudera is not responsible or liable in any way for the third party software packages.

By configuring and launching this AMP, you will cause **TheBloke/Llama-2-13B-chat-GGUF**, which is a third party large language model (LLM), to be downloaded and installed into your environment from the third party’s website. Additionally, you will be downloading **sentence-transformers/all-mpnet-base-v2**, which is the embedding model used in this project. Please see [TheBloke/Llama-2-13B-chat-GGUF](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF) and [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) for more information about the LLM and embedding model, including the applicable license terms.  

If you do not wish to download and install **TheBloke/Llama-2-13B-chat-GGUF** and **sentence-transformers/all-mpnet-base-v2**, do not deploy this repository.  By deploying this repository, you acknowledge the foregoing statement and agree that Cloudera is not responsible or liable in any way for **TheBloke/Llama-2-13B-chat-GGUF** and **sentence-transformers/all-mpnet-base-v2**. 

If you choose to use Pinecone instead of Milvus as your Vector DB, you acknowledge data will be transmitted to your tenant Pinecone account. You acknowledge outbound connections will be made to https://www.pinecone.io/ and Cloudera is not responsible for the data which leaves its platform. Author: Cloudera Inc.