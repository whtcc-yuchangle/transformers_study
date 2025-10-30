
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入gradio
import gradio as gr
# 导入transformers相关包
from transformers import pipeline
# 通过Interface加载pipeline并启动阅读理解服务
# 如果无法通过这种方式加载，可以采用离线加载的方式

# gr.Interface.from_pipeline(pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")).launch()
gr.Interface.from_pipeline(pipeline("fill-mask", model="google-bert/bert-base-chinese")).launch()
# pipe = pipeline("fill-mask", model="google-bert/bert-base-chinese")
# result = pipe("明天的[MASK]是晴天。")
# for res in result:
#     print(res['sequence'], res['score'])

