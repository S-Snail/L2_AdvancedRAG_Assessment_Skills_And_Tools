import os

from dashscope.common.constants import DASHSCOPE_API_KEY_ENV
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import OpenAI

ALI_TONGYI_API_KEY_SYSVAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALI_TONGYI_MAX_MODEL = "qwen-max"
ALI_TONGYI_DEEPSEEK_R1 = "deepseek-r1"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING = "text-embedding-v3"
ALI_TONGYI_RERANK = "gte-rerank-v2"
DEEPSEEK_API_KEY_OS_VAR_NAME = "Deepseek_Key"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"


def get_lc_model_client(
        api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME),
        base_url=ALI_TONGYI_URL,
        model=ALI_TONGYI_MAX_MODEL,
        temperature=0.7
):
    """
    通过Langchian获取大模型客户端你，默认获取qwen大模型客户端
    :param api_key:
    :param base_url:
    :param model:
    :param temperature:
    :return:
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature)


def get_ali_model_client(
        api_key=os.getenv(DASHSCOPE_API_KEY_ENV),
        base_url=ALI_TONGYI_URL,
        model=ALI_TONGYI_MAX_MODEL):
    """
    获取阿里大模型客户端
    :param api_key:
    :param base_url:
    :param model:
    :return:
    """
    return get_lc_model_client(api_key=api_key, base_url=base_url, model=model)


def get_ds_model_client(
        api_key=os.getenv(DEEPSEEK_API_KEY_OS_VAR_NAME),
        base_url=DEEPSEEK_URL,
        model=DEEPSEEK_CHAT_MODEL
):
    """
    获取DeepSeek大模型客户端
    :param api_key:
    :param base_url:
    :param model:
    :return:
    """
    return get_lc_model_client(
        api_key=api_key,
        base_url=base_url,
        model=model
    )


def get_ali_embeddings():
    """
    通过Langchain获得一个阿里通义千问嵌入模型实例
    :return:
    """
    return DashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING,
        dashscope_api_key=os.getenv(ALI_TONGYI_API_KEY_SYSVAR_NAME)
    )


def get_ali_clients():
    """
    获取阿里大模型客户端和嵌入模型客户端
    :return:
    """
    return get_ali_model_client(), get_ali_embeddings()


def get_ali_rerank(top_n=3):
    """
    通过Langchain获取一个阿里重排序模型实例
    :param top_n:
    :return:
    """
    return DashScopeRerank(
        model=ALI_TONGYI_RERANK,
        dashscope_api_key=os.getenv(ALI_TONGYI_RERANK)
    )
