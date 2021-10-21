
class SLiM_plot:
    def __init__(self,obj_SLiM):
        self.obj_SLiM=obj_SLiM


    def Plot_q_scale_rational_surfaces_colored(self,peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list,color_list):
        marker_size=5
        x_peak,x_stability_boundary_min,x_stability_boundary_max=self.ome_peak_range(peak_percent)

        x_min_index=np.argmin(abs(self.x-x_stability_boundary_min))
        x_max_index=np.argmin(abs(self.x-x_stability_boundary_max))
        
        q_range=self.q[x_min_index:x_max_index]

        x_list=[]
        y_list=[]
        x_list_error=[]
        y_list_error=[]
        fig, ax = plt.subplots()
        ax.fill_between(self.x, self.q*(1.-q_uncertainty), self.q*(1.+q_uncertainty), color='blue', alpha=.0)
        ax.axvline(x_stability_boundary_min,color='orange',alpha=0.6)
        ax.axvline(x_stability_boundary_max,color='orange',alpha=0.6)
        ax.plot(self.x,self.q,label=r'Safety factor $q_0$')
        
        
        for i in range(len(n_list)):
            n=n_list[i]
            x1=[]
            y1=[]
            x1_error=[]
            y1_error=[]

            qmin = np.min(self.q*(1.-q_uncertainty))
            qmax = np.max(self.q*(1.+q_uncertainty))
        
            m_min = math.ceil(qmin*n)
            m_max = math.floor(qmax*n)
            m_list=np.arange(m_min,m_max+1)

            for m in m_list:
                surface=float(m)/float(n)
                if np.min(q_range)*(1.-q_uncertainty)<surface and surface<np.max(q_range)*(1.+q_uncertainty):
                    x1.append(0.5*(x_stability_boundary_max+x_stability_boundary_min))
                    y1.append(surface)
                    x1_error.append(0.5*(x_stability_boundary_max-x_stability_boundary_min))
                    y1_error.append(0)
            if color_list==-1:
                ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',ms=marker_size,label='n='+str(n))
            else:
                ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',color=color_list[i],linestyle='none',ms=marker_size,label='n='+str(n))
            
            x_list.append(x1)
            y_list.append(y1)
            x_list_error.append(x1_error)
            y_list_error.append(y1_error)

        ax.plot(self.x,self.q*q_scale+q_shift,color='orange',label=r'Modified $q=$'+str(q_scale)+r'$*q_0$')

        ax.set_xlim(self.x0_min,self.x0_max)
        ax.set_xlabel(r'$\rho_{tor}$')
        ax.set_ylabel('Safety factor')
        plt.legend(loc='upper left')
        plt.show()


    def Plot_q_scale_rational_surfaces_red_and_green(self,peak_percent,\
            q_scale,q_shift,q_uncertainty,n_list,unstable_list):
        marker_size=5
        #calculate the radial stability boundary
        x_peak,x_stability_boundary_min,x_stability_boundary_max=self.ome_peak_range(peak_percent)

        x_min_index=np.argmin(abs(self.x-x_stability_boundary_min))
        x_max_index=np.argmin(abs(self.x-x_stability_boundary_max))

        q_range=self.q[x_min_index:x_max_index]

        x_list=[]
        y_list=[]
        x_list_error=[]
        y_list_error=[]
        fig, ax = plt.subplots()
        ax.fill_between(self.x, self.q*(1.-q_uncertainty), self.q*(1.+q_uncertainty), color='blue', alpha=.2)
        ax.axvline(x_stability_boundary_min,color='orange',alpha=0.6)
        ax.axvline(x_stability_boundary_max,color='orange',alpha=0.6)
        ax.plot(self.x,self.q,color='blue',label=r'Safety factor $q_0$')
        
        stable_counter=0
        unstable_counter=0
        for i in range(len(n_list)):
            n=n_list[i]

            x1=[]
            y1=[]
            x1_error=[]
            y1_error=[]

            qmin = np.min(self.q*(1.-q_uncertainty))
            qmax = np.max(self.q*(1.+q_uncertainty))
        
            m_min = math.ceil(qmin*n)
            m_max = math.floor(qmax*n)
            m_list=np.arange(m_min,m_max+1)

            for m in m_list:
                surface=float(m)/float(n)
                if np.min(q_range)*(1.-q_uncertainty)<surface and surface<np.max(q_range)*(1.+q_uncertainty):
                    x1.append(0.5*(x_stability_boundary_max+x_stability_boundary_min))
                    y1.append(surface)
                    x1_error.append(0.5*(x_stability_boundary_max-x_stability_boundary_min))
                    y1_error.append(0)

            if unstable_list[i]==1:#for unstable case
                if unstable_counter==0:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='red',ms=marker_size,label='Unstable')
                else:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='red',ms=marker_size)
                unstable_counter=unstable_counter+1
            elif unstable_list[i]==0:#for stable case
                if stable_counter==0:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='green',ms=marker_size,label='Stable')
                else:
                    ax.errorbar(x1,y1,xerr=x1_error,yerr=y1_error,marker='s',linestyle='none',color='green',ms=marker_size)
                stable_counter=stable_counter+1
            x_list.append(x1)
            y_list.append(y1)
            x_list_error.append(x1_error)
            y_list_error.append(y1_error)

        ax.plot(self.x,self.q*q_scale+q_shift,color='orange',label=r'Modified $q=$'+str(q_scale)+r'$*q_0$')
        ax.set_xlim(self.x0_min,self.x0_max)
        ax.set_xlabel(r'$\rho_{tor}$')
        ax.set_ylabel('Safety factor')
        plt.legend(loc='upper left')
        plt.show()
