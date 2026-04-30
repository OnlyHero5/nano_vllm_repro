# repair-plans-docs Findings

## Current Findings

1. `plans/00` 缺少独立的 `PagedAttention / BlockManager / Scheduler` 主线篇章。
2. `04-07` 默认假设 `01-03` 已完成，导致和当前 HEAD 错位。
3. 部分篇章仍在建议整文件替换，和总览承诺不一致。
4. 上游参考总体正确，但很多章节没有把“上游方向”和“当前仓库落地状态”分开。
5. 当前仓库真实测试入口仍以 `test_Day1.py` 到 `test_Day4.py` 为主。
6. 已补写新的 `plans/00` 和 `plans/02A`，后续 `04-07` 可以直接引用新的文档边界和术语定义。
