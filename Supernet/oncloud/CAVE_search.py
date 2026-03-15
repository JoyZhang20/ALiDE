import random


def generate_architecture():
    fragments = [[0], [1, 99], [2, 99, 99]]
    result = []
    while len(result) < 7:
        frag = random.choice(fragments)
        if len(result) + len(frag) <= 7:
            result.extend(frag)
    return result


def evaluate(result, acc, delay):
    model_acc = 0
    model_delay = 0
    layer_num = 0
    for i in range(len(result)):
        if result[i] == 99:
            continue
        layer_num += 1
        model_delay += delay[result[i]][i]
        model_acc += acc[result[i]][i]
    model_acc = round(model_acc / layer_num, 3)
    return model_acc, model_delay


# 精度和延迟矩阵
acc = [[80, 80, 80, 80, 80, 80, 80],
       [60, 60, 60, 60, 60, 0, 0],
       [70, 70, 70, 70, 70, 70, 0]]
delay = [[10, 10, 10, 10, 10, 10, 10],
         [10, 10, 10, 10, 10, 0, 0],
         [10, 10, 10, 10, 10, 10, 0]]

# 搜索结构
population = 1000
results = []
for _ in range(population):
    arch = generate_architecture()
    acc_val, delay_val = evaluate(arch, acc, delay)
    results.append((arch, acc_val, delay_val))

# 筛选Pareto最优结构
pareto = []
for arch, acc_val, delay_val in results:
    dominated = False
    for _, acc2, delay2 in results:
        if acc2 >= acc_val and delay2 <= delay_val and (acc2 > acc_val or delay2 < delay_val):
            dominated = True
            break
    if not dominated:
        pareto.append((arch, acc_val, delay_val))

# 输出结果
for arch, acc_val, delay_val in pareto:
    print(f"Arch: {arch}, Acc: {acc_val}, Delay: {delay_val}, Obj: {acc_val - delay_val}")
