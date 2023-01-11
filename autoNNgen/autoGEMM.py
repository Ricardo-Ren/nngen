from matrix_multiply import run
import numpy as np
import re

DNA_SIZE = 24
POP_SIZE = 3
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 4
# how to select the parameter range is important, need to do some statically analyze
# and also statically analyze to kick off some number to reduce the search space
X_BOUND = [0, 2] 
Y_BOUND = [0, 2]
Z_BOUND = [0, 4]
best_config = []
best_hardware_resource = []
best_exe_cycles = 1e20
parameter_num = 3

def F(x, y, z):
    ans = [ _ for i in range(POP_SIZE)]
    for i in range(POP_SIZE):
        rslt = run(silent=False, filename='tmp.v',par_left_col=pow(2, int(x[i])), par_left_row=pow(2, int(y[i])), par_out_col=pow(2, int(z[i])))
        ans[i] = -int(re.findall(r'\d+', rslt)[0])
    return ans
    # return 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0  #


def get_fitness(pop):
    x, y, z = translateDNA(pop)
    pred = F(x, y, z)
    print("Cycle:"+ str(-max(pred)))
    return pred
    # return pred - np.min(pred)+1e-3  # 求最大值时的适应度
    # return np.max(pred) - pred + 1e-3  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
    y_pop = pop[:, DNA_SIZE:2*DNA_SIZE]  # 后DNA_SIZE位表示Y
    z_pop = pop[:, 2*DNA_SIZE:]  # 后DNA_SIZE位表示Y
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    z = z_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Z_BOUND[1] - Z_BOUND[0]) + Z_BOUND[0]
    return x, y, z


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * parameter_num)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=[(fit) / (sum(fitness)) for fit in fitness])
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y, z = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y, z):", (x[max_fitness_index], y[max_fitness_index], z[max_fitness_index]))
    # print(F(x[max_fitness_index], y[max_fitness_index], z[max_fitness_index]))


if __name__ == "__main__":
    
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * parameter_num))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  #  iterate N generations
        x, y, z = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(pop)

        pop = select(pop, fitness) 
    print_info(pop)