from RNNs import *
from Utils import *
import sys
import random
from tqdm import tqdm



sequences_length = [50,100,200,400,800]  
for sequence_length in sequences_length:
    test_with_parameters("Sequence Length - ", sequence_length, sequence_length = sequence_length)
    
hidden_sizes = [64,128,256,512,1024]    
for hidden_size in hidden_sizes:
    test_with_parameters("Hidden size - ", hidden_size, hidden_size = hidden_size)    
    
shift_variation_test()
    
    