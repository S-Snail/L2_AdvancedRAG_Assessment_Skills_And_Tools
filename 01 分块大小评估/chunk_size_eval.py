"""
    # 评估选择合适的知识块大小

"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory

from models import get_ali_clients, get_openai_client, ALI_TONGYI_MAX_MODEL, ALI_TONGYI_EMBEDDING

# 创建模型客户端
llm, embeddings = get_ali_clients()
vectordb = None

openai_client = get_openai_client()
vllm = llm_factory(client=openai_client, model=ALI_TONGYI_MAX_MODEL)
vllm_e = OpenAIEmbeddings(client=openai_client, model=ALI_TONGYI_EMBEDDING)

#  加载文档
file_path = Path(__file__).resolve().parent.parent / "data/初赛训练数据集.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print(f"文档个数：", len(docs))
