from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.docstore.document import Document
from sentence_transformers.cross_encoder import CrossEncoder

import json
import numpy as np
import re

embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')

class PromptGenerator:
    def __init__(self, knowledge_dir = 'fastchat/serve/documents', k = 3, chunk_length = 600):
        self.knowledge_dir = knowledge_dir  # Path to the knowledge base
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self.chunk_length = chunk_length  # Length of each chunk of text
        self._read_paragraphs()
        print('Loading CrossEncoder...')
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Prompt generation
    def _generate_prmopt(self, docs):
        return str.join('\n', docs)

    def _read_paragraphs(self):
        print('Loading embeddings...')
        self.embeddings=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es')
        print(f'Loading txt documents from {self.knowledge_dir}')
        loader =DirectoryLoader(self.knowledge_dir, show_progress=True, use_multithreading=True, glob='*.txt', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'},)
        data = loader.load()
        print(f'Splitting {len(data)} documents...')
        documents = []
        index = 0
        for d in data:
            print(f'Processing {d.metadata["source"]}')
            paragraphs = re.split('\n\n', d.page_content)
            for paragraph in paragraphs:
                documents += [Document(page_content=paragraph, metadata={'index': str(index)})]
                index += 1
        self.n_documents = index
        print(f'Found {index} documents')
        print('Creating vector store...')
        self.docsearch = Chroma.from_documents(documents, self.embeddings, ids=[str(index) for index in range(self.n_documents)])
        print('Saving documents...')
        with open('documents.txt', 'w', encoding='utf-8') as f:
            for document in documents:
                index = document.metadata['index']
                f.write(json.dumps({
                    'index': index,
                    'page_content': document.page_content,
                    'embedding': self.docsearch.get(index, include=['embeddings'])['embeddings'][0],
                }) + '\n')

    def get_prompt(self, question):
        question_vector = self.embeddings.embed_query(question)
        docs=[doc for doc in self.docsearch.similarity_search_by_vector_with_relevance_scores(question_vector, k=int(np.min([10, self.n_documents])))]
        scores = self.cross_encoder.predict([[question, doc[0].page_content] for doc in docs])
        filter=scores>0.3
        scores = np.array(scores)[filter]
        docs = np.array(docs)[filter]
        top_k_indices = scores.argsort()[::-1][:self.k]
        log = {
            'text': question,
            'text_vector': question_vector,
            'top_k_indices': [doc[0].metadata['index'] for doc in docs[top_k_indices]],
            'top_k_scores': scores[top_k_indices].tolist(),
            }
        context_docs = [docs[i][0].page_content for i in top_k_indices]
        if len(context_docs) == 0:
            prompt = ''
        else:
            prompt = self._generate_prmopt(context_docs)
        return prompt, log


if __name__ == '__main__':
    promptGenerator = PromptGenerator()
    prompt = promptGenerator.get_prompt("¿Qué clase tiene la especialidad MARMT de 1º EMIES el jueves a las 8?")
    print(prompt)
