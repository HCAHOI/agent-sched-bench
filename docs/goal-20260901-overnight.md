完成以下任务：
- 完成当前的FCFS
- 完成原始Cachewise，并和FCFS做比较
while(原始CacheWise没有达到比较好的加速比) {
- *加强数据集的负载强度，考虑调高到达率，或增加task数目
- *完成FCFS，然后Cachewise，然后比较加速比，然后...（即在新负载强度下重新进入循环)
}
每轮FCFS / CacheWise必须报告：
- Mean JCT / min
- P95 JCT / min
- Tasks / h

比较好的加速比最低要求：在同一workload下，原始CacheWise的Tasks / h至少为FCFS的1.20倍。
- 完成Cachewise + Oracle Length 调度
- 完成Continuum Public
- 完成ThunderAgent
- 完成Native Priority
 if(cache hit rate依然反常，即性能最好的反而不是hit rate最好的那一批){
- *分析背后的原因
} else {
- *考虑之前unique-128的结果
- *考虑为什么unique-128上会产生这样的cache hit情况
- *将其与退出循环时的数据集比较，分析数据集有什么区别导致了这样的情况
}
Note: 为每个小任务写一个小markdown，并commit，在commit msg中标明节点
---
当前配置：
L40S + Qwen/Qwen3-4B-Instruct-2507-FP8
