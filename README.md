# PDF Chat

A self-project to implement RAG pipeline, where we can chat with any uploaded PDFs

## Setup
Install `ollama`

Pull `llama3.1` model
```sh
# This pulls the 8B parameter model (approx 4.7GB).
ollama pull llama3.1
```

Install python requirements:
```sh
pip install -r requirements.txt
```
For this, I used python `3.12.3` inside WSL2 Ubuntu.

Test ollama using:
```sh
python test_ollama.py
```

## Usage
```sh
python app.py
```
