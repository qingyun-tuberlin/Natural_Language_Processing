import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('human_eval.csv')
df2 = pd.read_csv('human_eval_cqy.csv')

# 确保两个DataFrame的行数相同（如果需要的话）
if len(df1) != len(df2):
    print(f"警告：两个文件的记录数不同！human_eval.csv有{len(df1)}行，human_eval_cqy.csv有{len(df2)}行")
    # 可以选择合并或处理不匹配的情况
    # 这里我们只合并相同行的数据，如果有索引列可以按索引合并

# 提取human_2列并添加到df1中
df1['human_2'] = df2['human_2']

# 如果需要将human_2列插入到human_1列之后
# 先获取列的顺序
cols = df1.columns.tolist()
# 找到human_1的索引
human_1_index = cols.index('human_1')
# 将human_2插入到human_1之后
cols.insert(human_1_index + 1, cols.pop(cols.index('human_2')))
# 重新排列列
df1 = df1[cols]

# 保存合并后的文件
df1.to_csv('human_eval_merged.csv', index=False)

print("合并完成！结果已保存为 human_eval_merged.csv")
print(f"合并后的列名：{df1.columns.tolist()}")