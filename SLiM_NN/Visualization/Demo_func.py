import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


test=False

def plot_dense_NN(lines,dots):
    plt.clf()
    for line in lines:
        print(line)
        plt.plot(line[0],line[1],color='blue',alpha=line[2])
    for dot in dots:
        plt.scatter(dot[0],dot[1],color='red',s=dot[2]*500)
    plt.savefig('Output_plot.png')
    plt.show()

def weights_to_lines_and_dots(weights):
    lines=[]
    dots=[]

    for i in range(int(len(weights)/2)):
        weight=weights[i*2+0]
        print('weight='+str(weight))
        (nx1,nx2)=np.shape(weight)

        dot_weights=np.sum(abs(weight),axis=1)
        dot_weights=dot_weights/np.sum(dot_weights)
        print('dot_weights='+str(dot_weights))
        #x_(n)=w^T * x (n-1) + b
        # N           M 
        #      N*M
        dot_num=len(dot_weights)

        for j in range(dot_num):
            dots.append([i,dot_num/2-j,dot_weights[j]])
        
        print(nx1,nx2)
        weight=abs(weight)/np.sum(abs(weight))

        x1=i 
        x2=i+1

        for j in range(nx1):
            for k in range(nx2):
                y1=nx1/2-j
                y2=nx2/2-k
                line_weight=weight[j,k]
                lines.append([[x1,x2],\
                            [y1,y2],\
                            line_weight])
    

    for i in range(nx2):
        dots.append([int(len(weights)/2),\
                    nx2/2-i,1/nx2])
        

        #lines.append()
    return lines,dots

def first_weight(weights):
    weight=weights[0]
    dot_weights=np.sum(abs(weight),axis=1)
    dot_weights=dot_weights/np.sum(dot_weights)
    return dot_weights
    
#plt.plot(x, y, color='green', marker='o', linestyle='dashed',
#     linewidth=2, markersize=12)


if test==True:
    lines=[ #[x1,x2], [y1,y2], weight
            [[0,1],[0,0],1]\
            ]
    dots=[  #[x,y,weight]
            [0,0,0.5],\
            [1,0,0.8]\
            ]

    plot_dense_NN(lines,dots)


    model=tf.keras.models.load_model('./Save/linear_regression.h5')

    #get from: https://stackoverflow.com/questions/52702220/access-the-weight-matrix-in-tensorflow-in-order-to-make-apply-changes-to-it-in-n
    weights = model.get_weights()
    first_weight=first_weight(weights)
    lines,dots,first_weight=weights_to_lines_and_dots(weights)
    plot_dense_NN(lines,dots)