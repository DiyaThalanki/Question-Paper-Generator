import cohere
import uuid
import hnswlib
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
import nltk
from nltk.tokenize import sent_tokenize

co = cohere.Client("5gRsyvOhjKGENuwyaTIIPzjw40OB4l9BNhtDr9Lb")

class Vectorstore:
    def __init__(self, raw_documents: Dict[str, str]):
        self.raw_documents = raw_documents
        self.docs = [] #chunked versions of documents
        self.docs_embs = [] #embeddings of chunked documents
        self.retrieve_top_k = 10 #retrieval
        self.rerank_top_k = 3 #ranking
        self.load_and_chunk() 
        self.embed()
        self.index()

        #documents->chunks->embeddings
    def load_and_chunk(self)->None:
        #loads raw data from url and makes it into chunks 
        #Each chunk is turned into a dictionary with three fields:
        #title: The web page’s title
        #text: The textual content of the chunk
        #url: The web page’s URL
        print("Loading documents...")


        #load and chunk for html documents
        # for raw_document in self.raw_documents:
        #     elements = partition_html(url=raw_document["url"])
        #     chunks = chunk_by_title(elements)
        #     for chunk in chunks:
        #         self.docs.append(
        #             {
        #                 "title": raw_document["title"],
        #                 "text": str(chunk),
        #                 "url": raw_document["url"],
        #             }
        #         )
        if "title" not in self.raw_documents or "content" not in self.raw_documents:
            raise KeyError(
                f"Missing 'title' or 'content' in raw_documents: {self.raw_documents}"
            )
        
        file_name = self.raw_documents["title"]
        file_content = self.raw_documents["content"]  # Text extracted from the file

        # Chunk the file content by sentences (or paragraphs/word count)
        sentences = sent_tokenize(file_content)
        chunk_size = 5  # Number of sentences per chunk
        chunks = [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]

        # Add each chunk to the docs list
        for chunk in chunks:
            self.docs.append(
                {
                    "title": file_name,
                    "text": chunk,
                    "url": None,  # No URL for uploaded files
                }
            )
        
    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        print("Embedding document chunks...")

        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        #dense retrival- embbed the query and do a similarity search 
        #reranking- rearrganges based on what is the most similar

        # Dense retrieval
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        # Reranking
        rank_fields = ["title", "text"] # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )
            
        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    #"url": self.docs[doc_id]["url"],
                }
            )
        return docs_retrieved

def run_chatbot(raw_documents,message, chat_history=[]):
    vectorstore = Vectorstore(raw_documents)
    
    # Generate search queries, if any        
    response = co.chat(message=message,
                        model="command-r-plus",
                        search_queries_only=True,
                        chat_history=chat_history)
    
    search_queries = []
    for query in response.search_queries:
        search_queries.append(query.text)

    # If there are search queries, retrieve the documents
    if search_queries:
        print("Retrieving information...", end="")

        # Retrieve document chunks for each query
        documents = []
        for query in search_queries:
            documents.extend(vectorstore.retrieve(query))

        # Use document chunks to respond
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            documents=documents,
            chat_history=chat_history,
        )

    else:
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            chat_history=chat_history,
        )
        
    # Print the chatbot response and citations
    chatbot_response = ""
    print("\nChatbot:")

    for event in response:
        if event.event_type == "text-generation":
            print(event.text, end="")
            chatbot_response += event.text
        if event.event_type == "stream-end":
            if event.response.citations:
                print("\n\nCITATIONS:")
                for citation in event.response.citations:
                    print(citation)
            if event.response.documents:
                print("\nCITED DOCUMENTS:")
                for document in event.response.documents:
                    print(document)
            # Update the chat history for the next turn
            chat_history = event.response.chat_history

    return chat_history
        

