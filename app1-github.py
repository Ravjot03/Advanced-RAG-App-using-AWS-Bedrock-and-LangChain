import streamlit as st
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# AWS Bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

# Prompt template
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 250 words 
to summarize with detailed explanations. If you don't know the answer, 
just say that you don't know.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Functions
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_documents(documents)

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def load_vectorstore():
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

def get_claude_llm():
    return Bedrock(model_id="anthropic.claude-v2", model_kwargs={"max_tokens_to_sample": 200})

def get_llama2_llm():
    return Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={"max_gen_len": 512})

def get_response_llm(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})

# UI
def main():
    st.set_page_config(page_title="📄 Chat with PDFs | AWS Bedrock", layout="wide")

    st.title("📄 Chat with PDF using AWS Bedrock")
    st.caption("Ask detailed questions and get context-aware answers from your uploaded PDF files.")

    with st.sidebar:
        st.header("📂 Vector Store Management")
        if st.button("🔄 Create/Update Vector Store"):
            with st.spinner("Ingesting and indexing documents..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully!")

        st.markdown("---")
        llm_choice = st.selectbox("🤖 Choose LLM Model", ["Claude (Anthropic)", "LLaMA 3 (Meta)"])

    col1, col2 = st.columns([1, 2])

    with col1:
        user_question = st.text_area("💬 Enter your question:", height=150, placeholder="What is the main idea of the PDF?")

    if st.button("Generate Answer") and user_question.strip():
        with st.spinner("Generating answer..."):
            vectorstore = load_vectorstore()
            llm = get_claude_llm() if llm_choice == "Claude (Anthropic)" else get_llama2_llm()
            result = get_response_llm(llm, vectorstore, user_question)

            with col2:
                with st.expander("📝 Answer", expanded=True):
                    st.markdown(result['result'])

                with st.expander("📚 Source Documents"):
                    for doc in result['source_documents']:
                        st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                        st.code(doc.page_content[:1000], language="markdown")  # Limit for readability
    else:
        with col2:
            st.info("Enter a question and click 'Generate Answer' to get started.")

if __name__ == "__main__":
    main()
