import random
import sys
import time

import numpy as np

from utils.vocab import Vocab
from word2vec import CBOW

# Check Python Version
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 6


def test1():
    random.seed(42)
    np.random.seed(42)

    vocab = Vocab(corpus="./data/debug.txt")
    cbow = CBOW(vocab, vector_dim=4)
    cbow.train(corpus="./data/debug.txt", window_size=3, train_epoch=10, learning_rate=1.0)

    
    target = np.array(
        [[3.11860215, 2.66179065, 4.13034624, -4.01591387],
         [0.83989595, -3.57545255, -5.19561605, 2.35163572],
         [-0.81130629, -0.70835709, 2.15572164, 5.12860637],
         [0.07900037, -0.63210777, -0.90860262, -0.78983446],
         [-3.65570413, -1.39450143, 1.13254508, -2.26985741],
         [0.30083168, -0.96855349, 0.99481605, 3.08445524],
         [2.02103143, 3.68369802, -1.38468668, 3.01632081],
         [0.26609125, -4.92390922, 2.03494949, -1.92884174],
         [0.03072148, -0.69482314, -0.73327746, -0.58584191],
         [-0.65604885, -0.34470096, -0.58167883, -1.14657785],
         [-2.36205633, 0.60984972, -1.26989771, 0.35644506],
         [1.43893422, 1.15600495, 2.35230867, 0.10508352],
         [1.65415126, 0.17707811, 0.51990739, -0.60731304],
         [0.68432046, 3.09025483, -1.21478114, -0.05606814],
         [0.83213855, 2.86356832, -1.26720439, 0.07360025],
         [-1.79509042, -0.72334707, 0.69426802, -0.55482075],
         [-0.84337417, 0.02747693, -3.5083923, 0.23072806],
         [-2.70806637, 0.70217685, 0.36393442, -0.05244267],
         [-2.72430964, 0.99018762, -0.67736688, -1.51586982],
         [-2.3461432, -0.72832866, -0.47637021, -1.98999855],
         [0.72620685, 0.24659625, -0.33820395, -0.8728833]]
    )

    print(cbow.most_similar_tokens("i", 5))
    print(cbow.most_similar_tokens("he", 5))
    print(cbow.most_similar_tokens("she", 5))

    if (target - cbow.W1 < 1e-7).all():
        print("\nTest-1 pass :)\n")
    else:
        print("Something error :(")


def test2():
    random.seed(42)
    np.random.seed(42)

    try:
        model = CBOW.load_model("ckpt")
    except FileNotFoundError:
        vocab = Vocab(corpus="./data/treebank.txt", max_vocab_size=-1)
        model = CBOW(vocab, vector_dim=12)

    start_time = time.time()
    model.train(corpus="./data/treebank.txt", window_size=4, train_epoch=10, learning_rate=1e-2, save_path="ckpt")
    end_time = time.time()

    print(f"Cost {(end_time - start_time) / 60:.1f} min")

    print(model.most_similar_tokens("i", 10))
    print("\nTest-2 pass :)\n")


def test3():
    from utils.similarity import get_test_data, evaluate_similarity
    
    model = CBOW.load_model("ckpt")
    spearman_result, pearson_result = evaluate_similarity(model, *get_test_data())
    if spearman_result.correlation > 0.35 and pearson_result[0] > 0.5:
        print("\nTest-3 pass :)\n")
    else:
        print("Something error :(")


def main():
    test1()
    test2()
    test3()


if __name__ == '__main__':
    main()
