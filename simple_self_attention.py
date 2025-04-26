'''
In this code we will be calculating the self-attention of the 2nd word in the input sequence
with respect to all the other words in the input sequence without the use of trainable weights.
'''
import torch

# This input sequence has already been manually embedded into 3 dimensional vectors for simplicity
inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2) -> query token
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)
print("Embedded input sequence: ", inputs)

'''
ATTENTION SCORES between the query x^2 and all the other input elements - DOT PRODUCT
'''
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("\nAttention Scores between x^2 and other words: ", attn_scores_2)

'''
Normalize the attention scores such that they sum upto 1 to obtain ATTENTION WEIGHTS.
    - Method-1: Divide each element by the sum of all elements in the tensor. 
    - Method-2: Softmax function -> better at managing extreme values and offers
    favourable gradient properties during training. Also ensures positive attention
    weights (can be interpreted as probabilities or relative importance).
    PyTorch softmax avoids overflow and underflow.
'''
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("\nAttention Weights (Naive) between x^2 and other words: ", attn_weights_2_tmp)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("\nAttention Weights (Softmax) between x^2 and other words: ", attn_weights_2)

'''
CONTEXT VECTOR for query token (self-attention for query token)
    - Weighted sum of all input vectors, obtained by multiplying each input vector by its
    corresponding attention weight.
'''
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("\nContext Vector for x^2: ", context_vec_2)
print('-----------------------------------------------------------------------------------------')
'''
GENERALIZED PROCEDURE FOR CONTEXT VECTOR CALCULATION FOR ALL INPUT VECTORS
'''
def compute_self_attention(input_seq):
    num_elements = len(inputs)
    
    # attention scores
    '''
    attn_scores = torch.empty(num_elements, num_elements)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i,j] = torch.dot(x_i, x_j)
    print("\nAttention Score for all input elements: ", attn_scores)
    '''
    
    # faster way of calculating attention scores
    attn_scores = inputs @ inputs.T # matrix multiplication
    print("\nAttention Score for all input elements:\n ", attn_scores)
    
    # attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("\nAttention weights for all input elements:\n ", attn_weights)
    
    # context vectors
    context_vectors = attn_weights @ inputs
    print("\nContext vectors for all input elements:\n ", context_vectors)
    
compute_self_attention(input_seq=inputs)