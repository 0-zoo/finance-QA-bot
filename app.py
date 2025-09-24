import os
from dotenv import load_dotenv
import streamlit as st

# LangChain & OpenAI 모듈
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool
# ----------------------------
# 0. 환경 변수 불러오기
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.title("📊 재무제표 QA + 계산기 챗봇")

# ----------------------------
# 1. 문서 준비 (업로드 or 기본 파일)
# ----------------------------
uploaded_file = st.file_uploader("재무제표 PDF 업로드", type="pdf")

# 업로드된 파일 사용
if uploaded_file is not None and "vectorstore" not in st.session_state:
    file_path = "data/uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ 문서 업로드 완료")

# 기본 문서 사용
elif uploaded_file is None and "vectorstore" not in st.session_state:
    file_path = "data/sample_report.pdf"
    if os.path.exists(file_path):
        st.info("📂 업로드된 파일이 없어 기본 문서(sample_report.pdf)를 로드합니다.")
    else:
        st.warning("⚠️ 기본 문서가 없습니다. PDF를 업로드해주세요.")
        file_path = None
else:
    file_path = None

# ----------------------------
# 2. 최초 벡터스토어 생성 (1회만)
# ----------------------------
if file_path and "vectorstore" not in st.session_state:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=".chroma_db"
    )

    st.session_state.vectorstore = vectorstore

# ----------------------------
# 3. Agent (질문 시 실행)
# ----------------------------
if "vectorstore" in st.session_state:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    # RAG QA 체인
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    # Python 계산기 툴
    tools = [
        Tool(
            name="Financial QA Retriever",
            func=qa_chain.run,
            description="재무제표 관련 질문에 답변합니다."
        ),
        PythonREPLTool()  # 계산기 툴
    ]

    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    query = st.text_input("질문을 입력하세요", "2022년 부채비율을 계산해줘")

    if st.button("질문하기"):
        with st.spinner("분석 중..."):
            answer = agent.run(query)
            st.write("📌 답변:", answer)
