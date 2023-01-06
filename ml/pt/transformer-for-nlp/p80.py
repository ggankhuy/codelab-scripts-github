d_model=512
def positional_encoding(pos,pe):
    for i in range(0,512,2):
        pe[0][i]=math.sin(pos/10000**((2*i)/d_model)))
        pe[0][i+1]=math.cos(pos/10000**((2*i)/d_model)))
    return pe
