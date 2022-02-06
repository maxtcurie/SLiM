import numpy as np 


cpdef A_maker(float x_max, double del_x,\
    complex w1, float v1,float Zeff,float eta,\
    float alpha, float beta,\
    float ky,int ModIndex, float mu,float xstar):

    cdef int i #iteration index
    cdef float k #varible in the for loop
    cdef float mref=2.
    #cdef float BC=0.
    cdef float tau=+1.0
    cdef complex w_hat = w1/v1
    # making grid
    cdef float x_min = -x_max
    cdef float x_TEMP=1./del_x**2.

    x_grid = np.arange(x_min, x_max+del_x, del_x)
    cdef int num = len(x_grid)
    # print(num)
    # initializing matrix A
    A = np.zeros((2*num-4, 2*num-4), dtype=complex)
    L11_grid=np.zeros(num,dtype=complex)
    L12_grid=np.zeros(num,dtype=complex)

    k_list=(np.sqrt(2.)*x_grid*alpha*np.sqrt(mref*1836.))/v1
    for i in range(num):
        k=k_list[i]

        L11_grid[i]=(k**2*((0. - 1.07728*10**11 *1j) + (0. + 7.15854*10**12 *1j)*w1**2 + \
              w1*((-4.91543*10**12 + \
                   0. *1j) - (6.00731*10**12 + 0. *1j) *Zeff) - (0. + \
                 1.78339*10**11 *1j) *Zeff - (0. + 5.19058*10**10 *1j) *Zeff**2) + \
           w1*((-1.33521*10**11 + \
                0. *1j) - (0. + 1.79729*10**12 *1j)*w1**3 - (2.82974*10**11 + \
                 0. *1j) *Zeff - (1.66865*10**11 + \
                 0. *1j) *Zeff**2 - (2.98419*10**10 + 0. *1j) *Zeff**3 + \
              w1**2*((2.4174*10**12 + 0. *1j) + (2.07088*10**12 + 0. *1j) *Zeff) + \
              w1*((0. + \
                   1.0073*10**12 *1j) + (0. + 1.54224*10**12 *1j) *Zeff + (0. + \
                    4.68208*10**11 *1j)*Zeff**2)))/(k**4*((0. + 8.44405*10**12 *1j) + \
              2.59608*10**13*w1 + (0. + 9.78999*10**12 *1j)*Zeff) + \
           k**2*((0. + 3.05552*10**11 *1j) - 2.25891*10**13*w1**3 + \
              w1**2*((0. - \
                   1.93809*10**13 *1j) - (0. + 2.39757*10**13 *1j) *Zeff) + (0. + \
                 8.17822*10**11 *1j) *Zeff + (0. + \
                 6.63714*10**11 *1j) *Zeff**2 + (0. + 1.50326*10**11 *1j) *Zeff**3 + \
              w1*(3.7004*10**12 + 9.237*10**12 *Zeff + 6.14607*10**12 *Zeff**2)) + \
           w1*(2.55733*10**10 + 3.94983*10**12*w1**4 + \
              w1**3*((0. + 5.66084*10**12 *1j) + (0. + 5.78006*10**12 *1j) *Zeff) + \
              1.41442*10**11 *Zeff + 2.03834*10**11 *Zeff**2 + \
              9.31852*10**10 *Zeff**3 + 1.32234*10**10 *Zeff**4 + \
              w1**2*(-2.68081*10**12 - 5.42596*10**12 *Zeff - \
                 2.38259*10**12 *Zeff**2) + \
              w1*((0. - \
                   4.87309*10**11 *1j) - (0. + 1.59005*10**12 *1j) *Zeff - (0. + \
                    1.42976*10**12 *1j) *Zeff**2 - (0. + 3.22827*10**11 *1j) *Zeff**3)))
        
        L12_grid[i]=(k**2*((0. + \
                2.69319*10**11 *1j) + (0. + \
                 2.57128*10**11 *1j)*w1**2 + ((0. + \
                   4.45847*10**11 *1j) + (0. + 1.29765*10**11 *1j) *Zeff) *Zeff + \
              w1*(6.26151*10**11 + 8.2297*10**11 *Zeff)) + \
           w1*(-4.39014*10**10 - (0. + 4.44069*10**11 *1j)*w1**3 + \
              w1**2*(6.42974*10**11 + 8.36592*10**11 *Zeff) + \
              Zeff*(-1.6558*10**11 + (-1.7495*10**11 - \
                    4.47629*10**10 *Zeff) *Zeff) + \
              w1*((0. + \
                   2.96121*10**11 *1j) + ((0. + \
                      7.63062*10**11 *1j) + (0. + \
                       4.43186*10**11 *1j) *Zeff) *Zeff)))/(k**4*((0. + \
                8.44405*10**12 *1j) + \
              2.59608*10**13*w1 + (0. + 9.78999*10**12 *1j) *Zeff) + \
           k**2*((0. + 3.05552*10**11 *1j) - 2.25891*10**13*w1**3 + \
              w1**2*((0. - 1.93809*10**13 *1j) - (0. + 2.39757*10**13 *1j) *Zeff) + \
              Zeff*((0. + \
                   8.17822*10**11 *1j) + ((0. + \
                      6.63714*10**11 *1j) + (0. + 1.50326*10**11 *1j) *Zeff) *Zeff) + \
              w1*(3.7004*10**12 + Zeff*(9.237*10**12 + 6.14607*10**12 *Zeff))) + \
           w1*(2.55733*10**10 + 3.94983*10**12*w1**4 + \
              w1**3*((0. + 5.66084*10**12 *1j) + (0. + 5.78006*10**12 *1j) *Zeff) + \
              w1**2*(-2.68081*10**12 + (-5.42596*10**12 - \
                    2.38259*10**12 *Zeff) *Zeff) + \
              w1*((0. - 4.87309*10**11 *1j) + \
                 Zeff*((0. - \
                      1.59005*10**12 *1j) + ((0. - \
                         1.42976*10**12 *1j) - (0. + \
                          3.22827*10**11 *1j) *Zeff) *Zeff)) + \
              Zeff*(1.41442*10**11 + \
                 Zeff*(2.03834*10**11 + \
                    Zeff*(9.31852*10**10 + 1.32234*10**10 *Zeff)))))
            
    
    if ModIndex==0:
        ModG=1.
    elif ModIndex==1:
        ModG=np.exp(-((x_grid-mu)/xstar)**2)
    else:
        print("ModIndex must be 0 or 1")
        ModG=0
    
    sigma_grid = (w1*L11_grid-(1.0+eta)*np.multiply(L11_grid,ModG) - eta*np.multiply(L12_grid,ModG))/v1
    
    a_temp=mref*1836.*beta*sigma_grid
    a11=ky**2 +1j*a_temp
    a12=-1j*a_temp*x_grid
    a21=2j*mref*1836*(alpha**2)*sigma_grid/(w1*(w1+tau*ModG))*x_grid
    a22=ky**2-2j*mref*1836*(alpha**2)*sigma_grid/(w1*(w1+tau*ModG))*x_grid**2
    # populating the matrix with the components of the matrix
    # this loop populates the off-diagonal components coming from the finite difference
    
    for i in range(num-3):
        A[i, i+1], A[i+1, i], A[num-2+i, num-2+i+1],  A[num-2+i+1, num-2+i] \
        = -x_TEMP,-x_TEMP,-x_TEMP,-x_TEMP

    for i in range(num-2):
        A[i,i] = 2*x_TEMP + a11[i+1]
        A[num-2+i, num-2+i] = 2*x_TEMP + a22[i+1]
        A[num-2+i, i] = a21[i+1]
        A[i, num-2+i] = a12[i+1]

    return A


#integrate the w_finder and VectorFinder
cpdef VectorFinder_auto_Extensive(float nu,float Zeff,float eta,\
    float shat,float beta,float ky,int ModIndex,float mu,float xstar):
    mu=abs(mu)
    cdef int judge=0
    cdef int loopindex=0
    cdef int i
    cdef float x_max=20.
    cdef float del_x=0.02
    cdef complex del_w
    cdef int neg_streak=0
    cdef float total_odd_even
    cdef complex w0
    x_grid=np.arange(-x_max,x_max,del_x,dtype=complex)
    cdef int num=len(x_grid)
    b=np.ones(2*num-2)

    #new guessing model(02/02/2022)
    #guess_f=np.array([1.+eta,0.5+eta,1.5+eta],dtype=float)+0.2
    guess_f=np.array([2.5,0.5+eta,1.5+eta],dtype=float)+0.2
    #guess_gamma=0.05+0.012*guess_f**2.
    guess_mod=guess_f+1j*(0.1+0.012*guess_f**2.)
    #print(guess_mod)

    cdef int guess_num=len(guess_mod)
    w_list=np.zeros(guess_num,dtype=float)
    odd_list=np.zeros(guess_num,dtype=float)

    for i in range(guess_num):
        w0=guess_mod[i]

        del_w = 0.01j
        neg_streak=0

        A = A_maker(x_max,del_x,w0,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
        det_A_minus = np.linalg.slogdet(A)
        w0=w0+del_w
    
        while np.abs(del_w) > 10**(-3.):
            # call A_maker to create and populate matrix A
            A = A_maker(x_max,del_x,w0,nu,Zeff,eta,shat,beta,ky,ModIndex,mu,xstar)
            det_A0 = np.linalg.slogdet(A)

            #parameter for the next run
            del_w = -del_w/(1-(det_A_minus[0]/det_A0[0])*np.exp(det_A_minus[1]-det_A0[1]))
            w0 = w0 + del_w
            print(w0)
            det_A_minus = det_A0
    
            if w0.imag<0:
                neg_streak=neg_streak+1
            else:
                neg_streak=0
    
            if neg_streak==4:
                break

        if neg_streak==4:
            continue
        else:
            AInverse=np.linalg.inv(A)
            change=1.
            lold=2.
            while change > 10.**(-8):
                b=np.matmul(AInverse,b)
                b=b/np.linalg.norm(b)
                lnew=np.matmul(np.conj(b),np.matmul(A,b))
                change=np.abs(lnew-lold)
                lold=lnew
            Aparallel=b[0:num]
            
            len_z_half=int(num/2)
            Aparallel_inv=np.flip(Aparallel) 
            #even parity sum(f(x)-f(-x)) = 0 if even -- f(x)= f(-x)
            evenness=np.sum(abs(Aparallel[len_z_half:]-Aparallel_inv[len_z_half:]))
            #odd  parity sum(f(x)+f(-x)) = 0 if odd  -- f(x)=-f(-x)
            oddness=np.sum(abs(Aparallel[len_z_half:]+Aparallel_inv[len_z_half:]))

            total_odd_even=evenness+oddness
            oddness_norm=1.-oddness/total_odd_even #percentage of oddness
            eveness_norm=1.-evenness/total_odd_even #percentage of evenness
            
            #Apar has even parity, and positive growth
            if oddness_norm<0.3 and np.imag(w0)>0:
                return w0
                break
    return 0.
