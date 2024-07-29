from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langfuse.callback import CallbackHandler
from vector_store import vectorstore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == '__main__':
    print("Retrieving...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    llm = ChatGroq(
        temperature=0,
        # model="mixtral-8x7b-32768",
        # model="llama-3.1-70b-versatile",
        model="llama-3.1-8b-instant",
        # model="llama3-70b-8192",
        # model="gemma2-9b-it",
    )

    vectorstore = vectorstore  # importado

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    query = "what is pinecone in machine learning?"
    # result = retrieval_chain.invoke(input={"input": query})

    # print(result)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that yout don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks for asking" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
    )

    # Initialize Langfuse handler


    langfuse_handler = CallbackHandler(
        secret_key="sk-lf-0c04d735-a3fa-44a1-90d0-6ca44daa6625",
        public_key="pk-lf-d7e6b046-82c7-42b3-a877-d50c6b6d56b8",
        host="http://localhost:3000"
    )

    res = rag_chain.invoke(query, config={"callbacks": [langfuse_handler]})
    print(res)

    #
    #
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    #
    # print(result)
