import pyquil.quil as pq
import pyquil.api as api
from pyquil.api import ForestConnection
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
from pyquil.quil import DefGate
from pyquil import get_qc

from grove.alpha.arbitrary_state import arbitrary_state, unitary_operator

import numpy as np
import random as rand
import itertools
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def make_qvm(qvm=None):
    if qvm:
        return qvm
    else:
        return api.QVMConnection()

class DistanceBasedClassifier:
    def __init__(self, qvm, num_features, num_train, num_test, classes):
        self.num_feat = num_features
        self.num_train = num_train
        self.num_test = num_test
        self.num_class = int(np.ceil(np.log2(classes)))
        self.qvm = make_qvm(qvm)
        self.len_datreg = int(np.ceil((np.log2(self.num_feat))))
        self.len_indreg = int(np.floor((np.log2(self.num_train))))
        self.total_qubits = self.len_datreg + self.len_indreg + self.num_class+1
        self.circ = pq.Program()
        self.ro = self.circ.declare("ro", "BIT", 2)
        self.qc = get_qc(str(self.total_qubits)+"q-qvm", as_qvm=True)
        #print(self.len_datreg, self. len_indreg, self.num_class, self.total_qubits)
        
    def init_registers(self):
        """ 
        Creates quantum and classical registers
        with `num_registers` qubits each.
        """
        # index qubit
        for i in range(self.len_indreg):
            self.circ += I(i)
        # ancilla qubit
        self.circ += I(self.len_indreg)
        # data qubit
        for i in range(self.len_datreg):
            self.circ += I(i+1+self.len_indreg)
        # class register
        for i in range(self.num_class):
            self.circ += I(i+1+self.len_datreg+self.len_indreg)

    def padding_helper(self, vec):
        #print(vec, , 2**self.len_datreg)
        np.pad(vec, (0, 2**self.len_datreg - vec.shape[0]), 'constant', constant_values=(0))
        return vec

    def create_gate_test(self, mat):
        mat_definition = DefGate("test_mat", mat)
        G_MAT = mat_definition.get_constructor()
        return mat_definition, G_MAT

    def create_gate_train(self, ind, mat):
        mat_definition = DefGate("train_mat_"+str(ind), mat)
        G_MAT = mat_definition.get_constructor()
        return mat_definition, G_MAT

    def create_control_gate(self, U, num_control):
        l = int(np.log2(U.shape[0]))
        num_tot = l + num_control
        CU = np.zeros((2**num_tot, 2**num_tot), dtype=complex)
        #print("CU: ", CU, "\nU:", U, "\n Num:", num_control, "\n Tot:", num_tot)
        ind = 2**num_tot - U.shape[0]
        for i in range(ind):
            CU[i][i] = 1
        for i in range(U.shape[0]):
            for j in range(U.shape[0]):
                CU[ind + i][ind + j] = U[i][j]
        #print("CU: ", CU)
        return CU

    def unitary_gate(self, vec):
        #print(vec)
        return unitary_operator.unitary_operator(self.padding_helper(vec))

    # If you are changing any of the parameters:
    # Classes, Number of training samples, features etc
    # Modify the below function accordingly.
   
    def interfere_circuit(self, test_X, train_X, traiy_y):
        """
        Creates quantum and classical registers
        with `num_registers` qubits each.
        """
        # Step 1. 
        # Ancilla & Index in Superposition
        
        for i in range(self.len_indreg):
            self.circ += H(i)
        
        anc_ind = self.len_indreg 
        self.circ += H(anc_ind)
        
        # Step 2.
        # This step needs modification with different
        # Parameters as mentioned above
        gt = self.create_control_gate(self.unitary_gate(test_X), 1)
        t1, g1 = self.create_gate_test(gt)
        self.circ += t1
        self.circ += g1(*[x+self.len_indreg for x in range(self.len_datreg+1)])
        self.circ += X(anc_ind)

        _t = []
        _g = []
        for ind, vec in enumerate(train_X):
            _gt = self.create_control_gate(
                self.unitary_gate(vec), self.len_datreg+1)
            __t, __g = self.create_gate_train(ind,_gt)
            self.circ += __t
            _g.append(__g)
        
        ### Modify here (Creating a Recursive Function)
        self.circ += _g[-1](*[x for x in range(anc_ind+1)])
        self.circ += X(0)
        self.circ += _g[-2](*[x for x in range(anc_ind+1)])
        self.circ += X(1)
        self.circ += X(0)
        self.circ += _g[-3](*[x for x in range(anc_ind+1)])
        self.circ += X(0)
        self.circ += _g[-4](*[x for x in range(anc_ind+1)])
        #self.circ += X(1)
        self.circ += X(0)
        for i in range(self.num_class):
            self.circ += CNOT(0, 1+self.len_datreg+self.len_indreg)

    def simulate(self):
        """
        Compile and run the quantum circuit
        on a simulator backend.
        """
        
        self.circ += H(self.len_indreg) 
        self.circ += MEASURE(self.len_indreg,self.ro[0])
        for i in range(self.num_class):
            self.circ += MEASURE(i+1+self.len_datreg+self.len_indreg, self.ro[i+1])

    def interpret_results(self, result_counts):
        """
        Post-selecting only the results where
        the ancilla was measured in the |0> state.
        Then computing the statistics of the class
        qubit.
        """
        total_samples = sum(result_counts.values())

        # define lambda function that retrieves only results where the ancilla is in the |0> state
        #for state, occurences in counts.items():
        #    print(state)
        #print("Interpret Results:", 1 + self.num_class - 1)

        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state  [0] == '0']

        #print_select = lambda counts: [print(state, occurences) for state, occurences in counts.items() if state  [1 + self.num_class - 1] == '0']

        # perform the postselection
        postselection = dict(post_select(result_counts))
        #dict(print_select(result_counts))
        postselected_samples = sum(postselection.values())

        # MODIFY IN CASE OF MULTIPLE CLASSES
        psel = postselected_samples/total_samples
        retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[1:self.num_class+1] == str(binary_class)]

        prob_class0 = sum(retrieve_class(0))/postselected_samples
        prob_class1 = sum(retrieve_class(1))/postselected_samples

        #print('Probability for class 0 is', prob_class0)
        #print('Probability for class 1 is', prob_class1)
        print('Post Selection Probability', psel)

        if (prob_class0 > prob_class1):
            return (0, prob_class0)
        else:
            return (1, prob_class1)

    def classify(self, test_vector, train_X, train_y):
        """
        Classifies the `test_vector` with the
        distance-based classifier using the `training_vectors`
        as the training set.
        This functions combines all other functions of this class
        in order to execute the quantum classification.
        """
        self.interfere_circuit(test_vector, train_X, train_y)
        self.simulate()
        self.circ.wrap_in_numshots_loop(1000)
        #print(self.circ)
        executable = self.qc.compile(self.circ)
        measures = self.qc.run(executable)
        count = np.unique(measures, return_counts=True, axis=0)
        print(count)
        count = dict(zip(list(map(lambda l: ''.join(list(map(str, l))),\
                        count[0].tolist())), count[1]))
        print(count)
        y1, y2 = self.interpret_results(count)
        return y1

if __name__ == "__main__":

    qvm = api.QVMConnection()

    # prepare data
    dat = load_iris()
    print("Shape: ", dat.data.shape)
    X_data = dat.data[:100, :2]
    y = dat.target[:100]
    
    # preprocessing
    standardized_X = preprocessing.scale(X_data)
    normalized_X = preprocessing.normalize(standardized_X)
    data_X = normalized_X
    sux = 0
    # initiate an instance of the distance-based classifier
    for ind in range(10):
        # Modify Here
        ind_dat_1 = [[x, x+50] for x in range(50)]
        ind_dat = [item for sublist in ind_dat_1 for item in sublist]
        ind_train_1 = rand.sample(range(50),2)
        ind_train_2 = rand.sample(range(50,100),2)
        ind_train = ind_train_1 + ind_train_2
        ind_test = [x for x in ind_dat if x not in ind_train]
        X_train = [data_X[x] for x in ind_train]
        y_train = [y[x] for x in ind_train]
        X_test = [data_X[x] for x in ind_test]
        y_test = [y[x] for x in ind_test]

        #print(X_train, y_train)
        #print(ind_test , y_test)
        total = 0
        succ = 0

        for i, vec in enumerate(X_test[:10]):
            classifier = DistanceBasedClassifier(qvm, np.size(
                X_train[0]), np.size(y_train), np.size(y_test), 2)
            classifier.init_registers()
            y_pred = classifier.classify(vec, X_train, y_train)
            total += 1
            print(y_pred, y_test[i])
            if y_pred == y_test[i]:
                succ += 1
        print("Success: ", succ/total)
        sux += succ/total
    print("Overall Success:", sux/10)
