# ner_conll2003

Simple containerized named entity recognition api which use the pretrained on
[conll2003](https://huggingface.co/datasets/conll2003) dataset [model](https://huggingface.co/dslim/bert-base-NER).

User can perform classification by sending the text as a string in JSON body and evaluate the performance of the model by uploading *csv* file with `tokens` and `ner_tags` columns. File with suitable structure and content can be downloaded with use of the script in *data* folder.

# How to use?

You can use the docker image to interact with api.

First of all, pull the image (alternatively you can build it on a local machine):

```docker pull ghcr.io/shkalikovoleh/ner_conll2003:latest```

Run image and expose 8000 port:

```docker run -p 8000:8000 ghcr.io/shkalikovoleh/ner_conll2003:latest```

On `127.0.0.1:8000/docs` you can find a Swagger documentation of the api and
even test some queries (or at least generate the curl command, sometimes Swagger don't send any request for some reason). I would recommend to use Postman or similar tool.

# Limitations
Current implementation has several limitations:

- Current docker image is optimized (in terms of size) for CPU and doesn't support GPU computation (even if NVIDIA runtime for docker is installed) since I don't have a suitable GPU to test it. In order to enable GPU support one can change the base image of docker container to `huggingface/transformers-pytorch-gpu` and delete first 3 line of the `requirements.txt` file.
