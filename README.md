# transformers_study

基于transformers的NLP学习。


如果本平台对您的科研工作提供了帮助，可在论文致谢中加入：
英文版：Thanks for the support provided by OpenI Community (https://openi.pcl.ac.cn).
中文版：感谢启智社区提供的技术支持(https://openi.pcl.ac.cn)。
  
  
如果您的成果中引用了本平台，也欢迎在下述开源项目中提交您的成果信息：
https://openi.pcl.ac.cn/OpenIOSSG/references


## 下载说明

**huggingface官网直接访问需要搭梯子，这里通过访问国内镜像网站来下载模型和数据集。**

```shell
export HF_ENDPOINT=https://hf-mirror.com 
或
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 模型下载
huggingface-cli download --resume-download gpt2 --local-dir gpt2
# 数据集下载
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```
