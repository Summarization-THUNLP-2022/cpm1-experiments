# 实验记录

## 简介

- 基于pytorch+model-center的第一次实验，目标是跑通、理解代码
- 使用CPM1模型，使用LCQMC数据集

## 对代码的修改

- 添加了cache-path选项，把从网上下载的模型存储到自定义路径下
- 添加了data-path选项，使数据和代码可以放在不同的路径下

## 结果

- jieba分词缓存文件位置无权限，修改缓存位置为"/data2/private/zhaoxinhao/tmp/"，不再报错。参考http://runxinzhi.com/yh-blog-p-13689767.html
- 跑了100个iter未报其他错误
