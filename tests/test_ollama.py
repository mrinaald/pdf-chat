# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mrinaald)

import asyncio
from ollama import AsyncClient


OLLAMA_MODEL_NAME = "llama3.1"
OLLAMA_MODEL_TAG = "latest"
# OLLAMA_MODEL = f"{OLLAMA_MODEL_NAME}:{OLLAMA_MODEL_TAG}"

OLLAMA_MODEL = "llama3.1"

async def get_joke():
    message = {'role': 'user', 'content': 'Tell me a joke!'}
    response = await AsyncClient().chat(model=OLLAMA_MODEL, messages=[message])

    print("="*50)
    print(response)
    print("="*50)
    print(response.message.content)
    print("="*50)


if __name__ == "__main__":
    asyncio.run(get_joke())
