# Gonzalo

Gonzalo es un agente conversacional basado en FastChat (https://github.com/lm-sys/FastChat) y Vicuna-Langchain (https://github.com/HaxyMoly/Vicuna-LangChain), capaz de leer los documentos de texto en directorio `fastchat\serve\documents`, permitiendo al usuario el consultar las fuentes utilizando el lenguaje natural.

Para ello se han integrado embeddings (`hiiamsid/sentence_similarity_spanish_es` · Hugging Face) y un cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · Hugging Face) en español. 

Para ejecutarlo es necesario instalar el modelo correspondiente.
