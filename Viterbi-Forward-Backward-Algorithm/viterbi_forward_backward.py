import pickle
import numpy as np
from numpy.core.fromnumeric import argmax, size


def viterbi(start_prob, trans_mat, emission_mat, sequence):

    # Create decision matrix 创建决策矩阵
    table = np.zeros((2,len(sequence)))

    # Initialize decision matrix 初始化决策矩阵
    table[0][0] = start_prob[0]*emission_mat[0][sequence[0]]
    table[1][0] = start_prob[1]*emission_mat[1][sequence[0]]

    # Compute decision matrix using dynamic programming 动态规划计算
    num_hidden = 2
    for j in range(1, len(sequence)):
        for i in range(num_hidden):
            table[i][j] = max([table[k][j-1]*trans_mat[k][i]*emission_mat[i][sequence[j]] \
                        for k in range(num_hidden)])
    
    # Backtracking 回溯
    path = []
    for j in range(len(sequence)):
        path.append(argmax([table[k][j] for k in range(num_hidden)]))
    
    return path

def forward(start_prob, trans_mat, emission_mat, sequence):

    # Create decision matrix 创建决策矩阵
    table = np.zeros((2,len(sequence)))

    # Initialize decision matrix 初始化决策矩阵
    table[0][0] = start_prob[0]*emission_mat[0][sequence[0]]
    table[1][0] = start_prob[1]*emission_mat[1][sequence[0]]

    # Compute decision matrix using dynamic programming 动态规划计算
    num_hidden = 2
    for j in range(1, len(sequence)):
        for i in range(num_hidden):
            table[i][j] = sum([table[k][j-1]*trans_mat[k][i]*emission_mat[i][sequence[j]] \
                        for k in range(num_hidden)])

    # Compute probability through decision matrix and return 通过决策矩阵计算序列概率并返回
    return sum(table[:,-1])

def backward(start_prob, trans_mat, emission_mat, sequence):

    # Create decision matrix 创建决策矩阵
    table = np.zeros((2,len(sequence)))

    # Initialize decision matrix 初始化决策矩阵
    table[0][-1] = 1
    table[1][-1] = 1

    # Compute decision matrix using dynamic programming 动态规划计算
    num_hidden = 2
    for j in range(2, len(sequence)+1):
        j = len(sequence) - j
        for i in range(num_hidden):
            table[i][j] = sum([table[k][j+1]*trans_mat[i][k]*emission_mat[k][sequence[j+1]] \
                        for k in range(num_hidden)])

    # Compute probability through decision matrix and return 通过决策矩阵计算序列概率并返回
    return sum([table[k][0]*start_prob[k]*emission_mat[k][sequence[0]] \
                        for k in range(num_hidden)])

if __name__ == '__main__':
        
    # Load file 加载文件 
    with open("hmm_parameters.pkl", "rb") as f:
        hmm_param = pickle.load(f)

    # 获取`状态概率`，`转移概率`，和`发射概率`矩阵 
    # Get the `state probability`, `transition probability`, and `launch probability` matrix
    start_prob = hmm_param['start_prob']
    trans_mat = hmm_param['trans_mat']
    emission_mat = hmm_param['emission_mat']     

    seq = '管仁阳是个好同学'
    seq_ord = [ord(seq[i]) for i in range(len(seq))]

    path= viterbi(start_prob, trans_mat, emission_mat, seq_ord)
    p_for = forward(start_prob, trans_mat, emission_mat, seq_ord)
    p_back = backward(start_prob, trans_mat, emission_mat, seq_ord)
    
    print('分词结果为',path,"(0:B, 1:I)")
    print('前向概率为',p_for)
    print('后向概率为',p_back)
