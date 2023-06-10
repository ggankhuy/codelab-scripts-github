import numpy as np
from scipy.special import softmax

# step 1: input : 3 inputs, d_model=4

print("x:")
x=np.array([\
[1.0,0.0,1.0,0.0],\
[0.0,2.0,0.0,2.0],\
[1.0,1.0,1.0,1.0]])

print(x)

# step 2: weights 3 dimensions x d_model=4

print("wq, wk, wv:")
w_query=np.array([[1,0,1], [1,0,0], [0,0,1], [0,1,1]])
print(w_query)

w_key=np.array([[0,0,1], [1,1,0], [0,1,0], [1,1,0]])
print(w_key)

w_value=np.array([[0,2,0], [0,3,0], [1,0,3], [1,1,0]])
print(w_value)

# step3 : matrix multiplication to obtain Q,K and V.
# [3,3] matmul [3,3] = [3,3]

print("Q/K/V:")
Q=np.matmul(x,w_query)
print(Q)
K=np.matmul(x,w_key)
print(K)
V=np.matmul(x,w_value)
print(V)


# Scaled attention scores

k_d=1
attention_scores=(Q@K.transpose())/k_d
print(f'attention_scores: \n{attention_scores}')

# step5 scaled softmax attention scores for each vector

attention_scores[0]=softmax(attention_scores[0])
attention_scores[1]=softmax(attention_scores[1])
attention_scores[2]=softmax(attention_scores[2])

print(f'attention_scores after softmax: \n{attention_scores}')

#step6 final attn presentations.

attention1=attention_scores[0].reshape(-1,1)
attention1=attention_scores[0][0]*V[0]
print(f'attention1: {attention1}')
attention2=attention_scores[0].reshape(-1,1)
attention2=attention_scores[0][1]*V[1]
print(f'attention2: {attention2}')
attention3=attention_scores[0].reshape(-1,1)
attention3=attention_scores[0][2]*V[2]
print(f'attention3: {attention3}')

#step7 summing the results.

attention_input1=attention1+attention2+attention3
print(f'attention_input: {attention_input1}')
