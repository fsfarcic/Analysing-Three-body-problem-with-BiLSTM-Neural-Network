#### USPJEŠAN KOD ČISTO ZA MULTIPROCESSING !!!!  KORIGIRAJ SEED 
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp


def runge_kutta_4th_order(trenutno_vrijeme, state_vector_trenutni, vremenski_korak, funckija):
#  Butcher koeficijenti
    c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2],dtype=np.float32)
    a = np.array([[0, 0, 0, 0, 0, 0],
                  [1/4, 0, 0, 0, 0, 0],
                  [3/32, 9/32, 0, 0,   0, 0],
                  [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
                  [439/216, -8, 3680/513, -845/4104, 0, 0],
                  [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]],dtype=np.float32)
    b = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55],dtype=np.float32)

    k1 = vremenski_korak * funckija(trenutno_vrijeme + c[0]*vremenski_korak, \
                                    state_vector_trenutni)
    k2 = vremenski_korak * funckija(trenutno_vrijeme +c[1]*vremenski_korak, \
                                     state_vector_trenutni + a[1][0]*k1)
    k3 = vremenski_korak * funckija(trenutno_vrijeme + c[2]*vremenski_korak, \
                                    state_vector_trenutni + a[2][0]*k1 + a[2][1]*k2)
    k4 = vremenski_korak * funckija(trenutno_vrijeme + c[3]*vremenski_korak, \
                                    state_vector_trenutni + a[3][0]*k1 + a[3][1]*k2 + a[3][2]*k3)
    k5 = vremenski_korak * funckija(trenutno_vrijeme + c[4]*vremenski_korak, \
                                    state_vector_trenutni + a[4][0]*k1 + a[4][1]*k2 + a[4][2]*k3 + a[4][3]*k4)
    k6 = vremenski_korak * funckija(trenutno_vrijeme + c[5]*vremenski_korak, \
                                    state_vector_trenutni + a[5][0]*k1 + a[5][1]*k2 + a[5][2]*k3 + a[5][3]*k4 + a[5][4]*k5)

    # sljedeći korak
    state_vector_sljedeci= state_vector_trenutni + b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6


    return state_vector_sljedeci



def adaptive_runge_kutta_4th_order(t, state_vektor, h, h_min, h_max, target_err, s_factor, f):
    # Take one full step with rk4
    y_full = runge_kutta_4th_order(t, state_vektor, h, f)

    # Take two half steps with rk4
    h_half = h / 2.0
    y_half = runge_kutta_4th_order(t, state_vektor, h_half, f)
    y_half = runge_kutta_4th_order(t+h_half, y_half, h_half, f)

    # Calculate absolute and relative truncation error: take the min of two
    # Relative error

    # Absolute error
    absolute_error = np.linalg.norm(y_half - y_full,axis=0)
    # Minimum of the two
    relative_error = absolute_error/(np.linalg.norm(y_half)+ 1e-10)

    error = np.maximum(relative_error, absolute_error)
    error=max(error)

    # Prevent rounding to zero
    if error == 0.0:
        error = 1e-5

    # Calculate new step-length
    h_new = h * (abs(target_err / error))**0.25 * 0.9

    h_new=max(min(h_new,h*s_factor), h/s_factor)

    h_new = max(min(h_new, h_max), h_min)

    # Check change limit of new h to be within change bounds
    if error <= target_err or abs(h_new) <= h_min:
      t_next = t + h
      y_next = y_half
      h=h_new

    else:

      return adaptive_runge_kutta_4th_order(t,state_vektor,h_new,h_min,h_max,target_err,s_factor,f)

    return t_next, y_next ,h




def three_body_ODE_equations_3D(trenutno_vrijeme:float,state_vector:np.ndarray):
  #funkcija vraća derivacije sustava ODE koje opisuju kretanja triju tijela
  #state_vector je definiran :
  #
  #             | x1  | 0
  #             | y1  | 1
  #
  #             | x2  | 2
  #             | y2  | 3
  #
  #             | x3  | 4
  #             | y3  | 5
  #
  #             | v1x | 6
  #             | v1y | 7
  #
  #             | v2x | 8
  #             | v2y | 9
  #
  #             | v3x | 10
  #             | v3y | 11
  #

  #inicijaliziranje svih nula
  derivacije=np.zeros_like(state_vector, dtype=np.float64)

  #prvotne derivacije u pozicijama
  derivacije[:int(state_vector.shape[0]/2)]=state_vector[int(state_vector.shape[0]/2):]

  #udaljenosti
  dist_1_2=\
      np.sqrt((state_vector[0]-state_vector[2])**2 + (state_vector[1]-state_vector[3])**2  )
  dist_1_3=\
      np.sqrt((state_vector[0]-state_vector[4])**2 + (state_vector[1]-state_vector[5])**2  )
  dist_2_3=\
      np.sqrt((state_vector[2]-state_vector[4])**2 + (state_vector[3]-state_vector[5])**2  )

  # sile na planete
  derivacije[6:8]= \
      -((state_vector[0:2] - state_vector[2:4]) / dist_1_2**3) - ((state_vector[0:2] - state_vector[4:6]) / dist_1_3**3)
  derivacije[8:10]= \
      -((state_vector[2:4] - state_vector[0:2]) / dist_1_2**3) - ((state_vector[2:4] - state_vector[4:6]) / dist_2_3**3)
  derivacije[10:]= \
      -((state_vector[4:6] - state_vector[0:2]) / dist_1_3**3) - ((state_vector[4:6] - state_vector[2:4]) / dist_2_3**3)


  return derivacije


##naši random poč uvjeti za 2D

def generiranje_state_vektora(br_simulacija, trenutni_seed):
    #za pracenje rada workera
    np.random.seed(trenutni_seed)
    #ovdje se kreira state vektor
    random_pocetne_pozicije=(np.random.uniform(-1,1,((6*br_simulacija,))) ).reshape(6,-1)
    random_pocetne_brzine=(np.random.uniform(-0.5,0.5,((6*br_simulacija,))) ).reshape(6,-1)
    random_pocetne_brzine=np.round(random_pocetne_brzine,3)
    random_pocetne_pozicije=np.round(random_pocetne_pozicije,3)
    state_vektor=np.vstack((random_pocetne_pozicije,random_pocetne_brzine))
    state_vektor = state_vektor.astype(np.float64)
    #vraca se state vektor
    return state_vektor

#postavke
br_sim=10
h_initial = 0.1
h=h_initial
h_min =1e-6
h_max=10
target_err= 1e-6
s_faktor=2
t=0
vrijeme = 2
broj_sveukupnih_simulacija_stste_vektora = 10 #ovo je puta broj stupaca u state vektoru da dobijem ukupni broj smiulacija tj za 10 ovih i state od 5 stupaca to je 50 simuacija


#kreiranje multiprocwesing liste individualnih worker data
argumenti_za_multiprocessing=[]
for idx in range(broj_sveukupnih_simulacija_stste_vektora):
   argumenti_za_multiprocessing.append((t,generiranje_state_vektora(br_sim,idx),h))

#funkcija workera
def worker(args):
    #unpack 
    t_worker, state_vektor_worker, h_worker = args
    #lokalni spremnik za svakog wrokera
    lokalni_spremnik = [state_vektor_worker]
    #main while loop
    while t<vrijeme:
        t_next ,y_next ,h_worker = adaptive_runge_kutta_4th_order(t=t_worker,
                                                            state_vektor=state_vektor_worker,
                                                            h=h_worker,
                                                            h_min=h_min,
                                                            h_max=h_max,
                                                            target_err=target_err,
                                                            s_factor=s_faktor,
                                                            f=three_body_ODE_equations_3D)

        t_worker, state_vektor_worker = t_next, y_next

        # vektor_vremena.append(t)
        lokalni_spremnik.append(state_vektor_worker)
        if  t_worker + h_worker >vrijeme:
            h_worker = vrijeme - t_worker 
            if  h_worker<h_min:
                break
    
    #vraca lokalni spremnik koji se appenda u listu svih sremnika
    return lokalni_spremnik

if __name__ == '__main__':
    
    start=time.time()

    with mp.Pool(mp.cpu_count()) as pool:
        spremnik_spremnika_workera = pool.map(worker, argumenti_za_multiprocessing)
   

    # Print the results
    print("Results:", spremnik_spremnika_workera[0])
    print('broj simulacija',len(spremnik_spremnika_workera)*br_sim)
    spremnik = spremnik_spremnika_workera[0]
    print("vrijeme simulacije", time.time()-start)
    print(len(spremnik))
    a=0
    for stupac in range(spremnik[0].shape[1]):
        x1=[]
        y1=[]
        x2=[]
        y2=[]
        x3=[]
        y3=[]
        for i in range(len(spremnik)):
            x1.append(spremnik[i][0][stupac])
            y1.append(spremnik[i][1][stupac])
            x2.append(spremnik[i][2][stupac])
            y2.append(spremnik[i][3][stupac])
            x3.append(spremnik[i][4][stupac])
            y3.append(spremnik[i][5][stupac])

        plt.plot(x1,y1,'r')
        plt.plot(x2,y2,'g')
        plt.plot(x3,y3,'b')
        plt.show()
        if(a==10):
            break
        a+=1
