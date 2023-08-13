from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers.cross_encoder import CrossEncoder
from chromadb.config import Settings

import numpy as np

knowledge_persistence = './.knowledge'

class PromptGenerator:
    def __init__(self, k = 3, chunk_length = 600):
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self.chunk_length = chunk_length  # Length of each chunk of text
        print('Loading vector store...')
        self.docsearch = Chroma(
            persist_directory=knowledge_persistence, 
            embedding_function=HuggingFaceEmbeddings(model_name='hiiamsid/sentence_similarity_spanish_es'), 
            client_settings=Settings(anonymized_telemetry=False))
        print('Loading CrossEncoder...')
        self.cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    # Prompt generation
    def _generate_prmopt(self, docs):
        return str.join('\n', docs)

    def get_prompt(self, question):
        question_vector = self.docsearch.embeddings.embed_query(question)
        docs = self.docsearch.similarity_search_by_vector(question_vector, k=10)
        if len(docs) > 0:
            scores = self.cross_encoder.predict([[question, doc.page_content] for doc in docs])
            sorted_indeces = np.argsort(scores)[::-1]
            context_docs = []
            for index in sorted_indeces[:self.k]:
                if scores[index] < 0.3:
                    break
                context_docs.append(docs[index].page_content)
            prompt = self._generate_prmopt(context_docs)
        else:
            scores = []
            sorted_indeces = []
            prompt = ''
        log = {
            'text': question,
            'text_vector': question_vector,
            'top_k_indices': [docs[index].metadata['index'] for index in sorted_indeces],
            'top_k_scores': [float(scores[index]) for index in sorted_indeces],
            }
        return prompt, log


if __name__ == '__main__':
    promptGenerator = PromptGenerator()
    prompt = promptGenerator.get_prompt("¿Qué clase tiene la especialidad MARMT de 1º EMIES el jueves a las 8?")
    print(prompt)
