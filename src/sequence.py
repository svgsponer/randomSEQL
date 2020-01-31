import numpy as np
import click
import ahocorasick

from multiprocessing import Pool
from functools import partial

from sklearn.linear_model import RidgeClassifierCV


def generate_extractor(sequences, max_length=10):
    sid = np.random.randint(low=0, high=len(sequences))
    seq = sequences[sid]
    pos = np.random.randint(low=0, high=len(seq))
    sl = np.random.binomial(n=10, p=0.5)
    sl = max(sl, 1)
    s = seq[pos:pos+sl]
    return s


def generate_features(sequences, n):
    fs = np.ndarray(n, dtype='object')
    for i in range(n):
        s = generate_extractor(sequences)
        while s in fs:
            print(".", end='')
            s = generate_extractor(sequences)
        fs[i] = s
    print()
    return fs


def load_data(file_name):
    y = np.genfromtxt(file_name, skip_header=1, usecols=0)
    s = np.genfromtxt(file_name, skip_header=1, usecols=1, dtype='object')
    return y, s


def match(A, hs):
    vec = np.zeros(len(A))
    for end_index, (idx, f) in A.iter(hs):
        vec[idx] = 1
    # print(vec)
    return vec


def create_fvec(sequences, A):
    p = Pool(6)
    f_vec = p.map(partial(match, A), sequences)
    return f_vec


@click.command()
@click.option('--train_file', '-t', help="Training file")
@click.option('--test_file', '-s', help="Test file")
@click.option('--num_seq', '-n', default=10, help="Number of random sequences")
def run(train_file, test_file, num_seq):
    print("Load train data")
    y, s = load_data(train_file)

    print("Generate random features")
    ss = generate_features(s, num_seq)

    print("Generate automaton")
    A = ahocorasick.Automaton()
    for idx, f in enumerate(ss):
        A.add_word(f, (idx, f))
    A.make_automaton()

    print("Extract Feautre Vectors of train data")
    fvec = create_fvec(s, A)

    print("Learn classifier")
    cls = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    cls.fit(fvec, y)

    print("Load test data")
    y_test, s_test = load_data(test_file)
    print("Extract Feature Vector of test data")
    fvec_test = create_fvec(s_test, A)

    print("Predict")
    print(cls.score(fvec_test, y_test))


if __name__ == '__main__':
    run()
