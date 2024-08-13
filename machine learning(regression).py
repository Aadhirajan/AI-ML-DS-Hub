import time
import math
import numpy as np
x_train=np.array([1, 2, 3, 4, 5, 6,7,8])
y_train=np.array([5, 8, 11, 14, 17, 20,23,26])
def compute_cost(x,y,w,b):
    cost=0
    m=x.shape[0]
    for i in range(m):
         f_wb=(w * x[i])+b
         cost=cost+(f_wb - y[i])**2
    total_cost=cost/(2*m)
    return total_cost

def compute_gradient(x,y,w,b):
    m=x.shape[0]
    df_b=0
    df_w=0 
    for i in range(m):
        f_wb=w*x[i]+b
        df_w_i=(f_wb-y[i])*x[i]
        df_b_i=(f_wb-y[i])
        df_w=df_w+df_w_i
        df_b=df_b+df_b_i
    return df_b,df_w

def gradient_descent(x,y,alpha,itr,w_in,b_in):
    w=w_in
    b=b_in
    for i in range(itr):
        dj_b,dj_w=compute_gradient(x,y,w,b)
        w=w-alpha*dj_w
        b=b-alpha*dj_b
        if i% math.ceil(itr/10) == 0:
            print(f"Iteration {i:4}: Cost {compute_cost(x,y,w,b):0.2e} ",
                  f"dj_w: {dj_w: 0.3e}, dj_b: {dj_b: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w,b


w_init=0
b_init=0
iterations=10000
tmp_alpha=1.0e-3
start_time = time.time()
w_final,b_final=gradient_descent(x_train,y_train,tmp_alpha,iterations,w_init,b_init)
end_time = time.time()
print("Execution time:", (end_time - start_time)*1000)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
print(f"x=15 then y={w_final*15+b_final:0.1f}")
print(f"x=20 then y={w_final*20+b_final:0.1f}")

    