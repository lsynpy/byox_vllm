# Tensor Parallelism (TP)

## Refer

[Megatron-LM paper](https://arxiv.org/abs/1909.08053)

## Demonstration

`tp_demo.py`

```py
# Normal way
[1 2] @ [1 2 1 2  @ [2 2  = [3 6 5 6] @ [2 2 = [6+12+10+6 6+12+5+6] = [34 29]
         1 2 2 2]    2 2                 2 2
                     2 1                 2 1
                     1 1]                1 1]

# Normal split way

Step 1 split N on second matrix:

[1 2] @ [1 2 1 2  = [1 2] @ [1 2  concatenate [1 2] @ [1 2
         1 2 2 2]            1 2]                      2 2]
                  = [3 6] concatenate [5 6] = [3 6 5 6]

Step 2 split K on step 1 result and third matrix:
[3 6 5 6] @ [2 2 = [3 6] @ [2 2  element-wise add [5 6] @ [2 1
             2 2            2 2]                           1 1]
             2 1
             1 1]
                 = [6+12 6+12] element-wise add [10+6 5+6]
                 = [18 18] element-wise add [16 11]
                 = [34 29]

# Tensor Parallelism way

Step 1 split N on second Matrix

[1 2] @ [1 2 1 2  = [[1 2] @ [1 2   [1 2] @ [1 2
         1 2 2 2]            1 2]           2 2]]
                  = [[3 6] [5 6]]

Step 2 split K on third matrix

[3 6] @ [2 2  element-wise add [5 6] @ [2 1
         2 2]                           1 1]
= [34 29]
```

- TP save one concatenate(gather) compare to the normal split way

## 2M,2K,2N Split cases

2 matrix multiply, (2M,2K) @ (2K,2N)

| 编号 | 拆分维度  | 子任务                             | 合并方式          | 所需通信                   | 对应并行策略                            |
| ---- | --------- | ---------------------------------- | ----------------- | -------------------------- | --------------------------------------- |
| 1    | M         | $ C_i = A_i B $                    | `concat` 行       | `gather`                   | Sequence/Data Parallel                  |
| 2    | K         | $ C_i = A_i B_i $                  | `all-reduce sum`  | `all-reduce`               | Column Parallel (TP)                    |
| 3    | N         | $ C_i = A B_i $                    | `concat` 列       | `gather`                   | Row Parallel (TP)                       |
| 4    | M & K     | $ C_i = A_i B_i $                  | 分块乘法          | `all-reduce` + `broadcast` | 2D TP（部分）                           |
| 5    | M & N     | $ C*{ij} = \sum_k A*{ik} B\_{kj} $ | 分块乘法          | `all-reduce`               | 2D Tensor Parallel                      |
| 6    | K & N     | $ C_i = A_i B_i $                  | 分块乘法          | `all-reduce`               | 2D TP（列方向）                         |
| 7    | M & N & K | $ C*{ij} = \sum_k A*{ik} B\_{kj} $ | 分块乘法 + reduce | `all-reduce` + `broadcast` | 3D Tensor Parallel / Cannon's Algorithm |
