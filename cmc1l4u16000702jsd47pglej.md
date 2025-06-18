---
title: "在谷歌中屏蔽csdn搜索结果的方法"
datePublished: Wed Jun 18 2025 06:42:41 GMT+0000 (Coordinated Universal Time)
cuid: cmc1l4u16000702jsd47pglej
slug: csdn
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/HFmiFFxGFdU/upload/870900276a7f1ac9b362172aa695a914.jpeg
tags: 6k6h566x5py65oqa5pyv

---

最近几年CSDN越来越垃圾了，具体表现为文章浏览到一半就会跳广告要求登陆、文章需要开通ip付费才能观看，而且可看的文章内容十分低质，那么不如一次性不让浏览器返回来自CSDN的搜索结果。

1. 在Chrome的扩展中安装uBlacklist。
2. 添加如下规则：
`
*://*.csdn.net/*
`