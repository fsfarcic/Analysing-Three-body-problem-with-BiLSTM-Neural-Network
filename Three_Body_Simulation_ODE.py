#### USPJEŠAN KOD ČISTO ZA MULTIPROCESSING !!!!  KORIGIRAJ SEED
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd

os.chdir(Path(__file__).parent)

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
    
    #pozivanjem funkcije izvlacimo samo dervacije dio, dok bufffer sile naknadno spremamo i vracamo
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
    #sada runge kuta metodom korigiramo brzine i pozicije
    state_vector_sljedeci= state_vector_trenutni + b[0]*k1 + b[1]*k2 + b[2]*k3 + b[3]*k4 + b[4]*k5 + b[5]*k6

    #sada vracamo state vektor
    return state_vector_sljedeci



def adaptive_runge_kutta_4th_order(t, state_vektor, h, h_min, h_max, target_err, s_factor, f):
    # Take one full step with rk4
    y_full = runge_kutta_4th_order(t, state_vektor, h, f)
    
    # Take two half steps with rk4
    h_half = h / 2.0
    y_half = runge_kutta_4th_order(t, state_vektor, h_half, f)
    y_half = runge_kutta_4th_order(t+h_half, y_half, h_half, f)

    # Calculate absolute and relative truncation error: take the min of two
    # Absolute error, neuzimamo dijelove koji sluze samo kao buffer za spremanje sile u vektoru
    absolute_error = np.linalg.norm(y_half[:12] - y_full[:12],axis=0)
    # Relative error
    relative_error = absolute_error/(np.linalg.norm(y_half[:12])+ 1e-10)
    # Minimum of the two
    error = np.maximum(relative_error, absolute_error)
    error = max(error)

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

    #prije vracanja updjetamo pozicije
    dist_1_2=\
        np.sqrt((y_next[0]-y_next[2])**2 + (y_next[1]-y_next[3])**2  )
    dist_1_3=\
        np.sqrt((y_next[0]-y_next[4])**2 + (y_next[1]-y_next[5])**2  )
    dist_2_3=\
        np.sqrt((y_next[2]-y_next[4])**2 + (y_next[3]-y_next[5])**2  )
    y_next[12] = dist_1_2
    y_next[13] = dist_1_3
    y_next[14] = dist_2_3
  
    #konacno return
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
  #             | r12 | 12
  #             | r13 | 13
  #             | r23 | 14


  #inicijaliziranje svih nula
  derivacije=np.zeros_like(state_vector, dtype=np.float64)

  #dervacije pozicija su brzine te u vektor derivacije stavljami njih na pozicije pozicija
  derivacije[:6]=state_vector[6:12]

  #udaljenosti
  dist_1_2=\
      np.sqrt((state_vector[0]-state_vector[2])**2 + (state_vector[1]-state_vector[3])**2  )
  dist_1_3=\
      np.sqrt((state_vector[0]-state_vector[4])**2 + (state_vector[1]-state_vector[5])**2  )
  dist_2_3=\
      np.sqrt((state_vector[2]-state_vector[4])**2 + (state_vector[3]-state_vector[5])**2  )

  # sile na planete spremljene u vektor derivacije na mjesta brzina
  derivacije[6:8]= \
      -((state_vector[0:2] - state_vector[2:4]) / dist_1_2**3) - ((state_vector[0:2] - state_vector[4:6]) / dist_1_3**3)
  derivacije[8:10]= \
      -((state_vector[2:4] - state_vector[0:2]) / dist_1_2**3) - ((state_vector[2:4] - state_vector[4:6]) / dist_2_3**3)
  derivacije[10:12]= \
      -((state_vector[4:6] - state_vector[0:2]) / dist_1_3**3) - ((state_vector[4:6] - state_vector[2:4]) / dist_2_3**3)


  return derivacije


##naši random poč uvjeti za 2D kao staze vektor
def generiranje_state_vektora(br_simulacija_jednog_procesa_tj_broj_stupaca, trenutni_seed):
    #za pracenje rada workera
    np.random.seed(trenutni_seed)
    br_simulacija = br_simulacija_jednog_procesa_tj_broj_stupaca
    #ovdje se kreira state vektor
    random_pocetne_pozicije=(np.random.uniform(-1.,1.,((6*br_simulacija,))) ).reshape(6,-1)
    random_pocetne_pozicije=np.round(random_pocetne_pozicije,3)

    random_pocetne_brzine=(np.random.uniform(-0.5,0.5,((6*br_simulacija,))) ).reshape(6,-1)
    random_pocetne_brzine=np.round(random_pocetne_brzine,3)
    
    #pocetne udaljensoti
    #udaljenosti
    dist_1_2=\
        np.sqrt((random_pocetne_pozicije[0]-random_pocetne_pozicije[2])**2 +
                 (random_pocetne_pozicije[1]-random_pocetne_pozicije[3])**2  )
    dist_1_3=\
        np.sqrt((random_pocetne_pozicije[0]-random_pocetne_pozicije[4])**2 + 
                (random_pocetne_pozicije[1]-random_pocetne_pozicije[5])**2  )
    dist_2_3=\
        np.sqrt((random_pocetne_pozicije[2]-random_pocetne_pozicije[4])**2 + 
                (random_pocetne_pozicije[3]-random_pocetne_pozicije[5])**2  )


    state_vektor=np.vstack((random_pocetne_pozicije,random_pocetne_brzine))
    state_vektor=np.vstack((state_vektor,dist_1_2))
    state_vektor=np.vstack((state_vektor,dist_1_3))
    state_vektor=np.vstack((state_vektor,dist_2_3))
    state_vektor = state_vektor.astype(np.float64)
    #vraca se state vektor
    return state_vektor


#funkcija workera
def worker(args):
    #unpack
    t_worker, state_vektor_worker, h_worker, h_min, h_max, target_err, s_faktor = args
    #lokalni spremnik za svakog wrokera
    lokalni_spremnik = [state_vektor_worker]
    #main while loop
    while t_worker<vrijeme:
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
    #vanjska kontrola simulacije u slucaju pada sustava je s broj_uzastupnih_simulacija varijablom
    broj_uzastupnih_simulacija = 31
    for _ in range(broj_uzastupnih_simulacija):
        #provjeravamo dali postoji lokalni direktorij u kojeg se spremaju .csv fajlovi, ako nema napravimo
        filename_counter = 0
        if(os.path.isdir("Simulacije")== False):
            os.mkdir("Simulacije")
            print("Simulacije direktorij ne postoji te je izraden.")
        else:
            print("Simulacije direktorij vec postoji.")
            #uzet koliko ih ima pa najveći broj za naming
            csv_filenames = [file for file in os.listdir(os.path.join(os.getcwd(),"Simulacije")) if file.endswith(".csv")]
            filename_counter = len(csv_filenames)

        #=============POSTAVKE=============#
        ime_racunala = "Three Body Problem"
        br_lokalnih_sim=5
        h_initial = 0.1
        h=h_initial
        h_min =1e-6
        h_max=10
        target_err= 1e-6
        s_faktor=2
        t=0
        vrijeme = 2
        ukpuni_broj_globalnih_simulacija = 200 #ovo je puta broj stupaca u state vektoru da dobijem ukupni broj smiulacija tj za 10 ovih i state od 5 stupaca to je 50 simulacija
        #===================================#

        #pratimo seed I osiguravamo da je za svaku simulaciju drugaciji
        if(os.path.isfile("seed.npy")):
            seed = int(np.load("seed.npy"))+100
            print("Loaded saved seed")
        else:
            seed = int(hash(ime_racunala))
            seed = abs(seed)%(2**32)
            print("Created new seed")

        #kreiranje multiprocesing liste individualnih worker data
        #workeri svi dobivaju razlicite pocetne uvijete
        argumenti_za_multiprocessing=[]
        for idx in range(ukpuni_broj_globalnih_simulacija):
            argumenti_za_multiprocessing.append((t,
                                                 generiranje_state_vektora(br_lokalnih_sim,
                                                                           idx*10+seed),
                                                 h,
                                                 h_min, h_max, target_err, s_faktor))
            if(idx==ukpuni_broj_globalnih_simulacija-1):
                np.save("seed.npy",seed+idx*10)
                 
        #kreiranje i pokretanje radnika, tqdm prati progress
        print()
        print("SIMULATING THREE BODY:")
        print(mp.cpu_count())
        with mp.Pool(mp.cpu_count()) as pool:
            spremnik_spremnika_workera=[]
            with tqdm(total=ukpuni_broj_globalnih_simulacija) as pbar:
                for rezultat_spremnik in pool.imap(worker,argumenti_za_multiprocessing):
                    # Update progress bar for each task completed
                    spremnik_spremnika_workera.append(rezultat_spremnik)
                    pbar.update(1)

        #spremanje formatiranju .csv datoteka zasebno svake simulacije
        print()
        print("SAVING RESULTS INSIDE SIMULACIJE DIRECTORY")
        with tqdm(total=len(spremnik_spremnika_workera)) as pbar:
            #sad spremnik po spremnik
            for spremnik in spremnik_spremnika_workera:
            #za svaki spremnik napravit buffere za koordinate te iterirat po dubini
            #svaki buffer se puni prazni za svaki stupac zasebno
                for stupac_state_vektora in range(spremnik[0].shape[1]):
                    trenutni_x1 = []
                    trenutni_y1 = []
                    trenutni_x2 = []
                    trenutni_y2 = []
                    trenutni_x3 = []
                    trenutni_y3 = []
                    
                    trenutni_vx1 = []
                    trenutni_vy1 = []
                    trenutni_vx2 = []
                    trenutni_vy2 = []
                    trenutni_vx3 = []
                    trenutni_vy3 = []

                    r_12 = []
                    r_13 = []
                    r_23 = []
                    
                    #napuni buffere
                    for dubina in range(len(spremnik)):
                        trenutni_x1.append(spremnik[dubina][0][stupac_state_vektora])
                        trenutni_y1.append(spremnik[dubina][1][stupac_state_vektora])
                        trenutni_x2.append(spremnik[dubina][2][stupac_state_vektora])
                        trenutni_y2.append(spremnik[dubina][3][stupac_state_vektora])
                        trenutni_x3.append(spremnik[dubina][4][stupac_state_vektora])
                        trenutni_y3.append(spremnik[dubina][5][stupac_state_vektora])

                        trenutni_vx1.append(spremnik[dubina][6][stupac_state_vektora])
                        trenutni_vy1.append(spremnik[dubina][7][stupac_state_vektora])
                        trenutni_vx2.append(spremnik[dubina][8][stupac_state_vektora])
                        trenutni_vy2.append(spremnik[dubina][9][stupac_state_vektora])
                        trenutni_vx3.append(spremnik[dubina][10][stupac_state_vektora])
                        trenutni_vy3.append(spremnik[dubina][11][stupac_state_vektora])

                        r_12.append(spremnik[dubina][12][stupac_state_vektora])
                        r_13.append(spremnik[dubina][13][stupac_state_vektora])
                        r_23.append(spremnik[dubina][14][stupac_state_vektora])


                    #sad se napravi panda dataframe i spremi csv file
                    dataframe = pd.DataFrame({
                                    "Tijelo 1 - X ":trenutni_x1,
                                    "Tijelo 1 - Y ":trenutni_y1,
                                    "Tijelo 2 - X ":trenutni_x2,
                                    "Tijelo 2 - Y ":trenutni_y2,
                                    "Tijelo 3 - X ":trenutni_x3,
                                    "Tijelo 3 - Y ":trenutni_y3,
                                    "Tijelo 1 - Brzina u X smjeru":trenutni_vx1,
                                    "Tijelo 1 - Brzina u Y smjeru":trenutni_vy1,
                                    "Tijelo 2 - Brzina u X smjeru":trenutni_vx2,
                                    "Tijelo 2 - Brzina u Y smjeru":trenutni_vy2,
                                    "Tijelo 3 - Brzina u X smjeru":trenutni_vx3,
                                    "Tijelo 3 - Brzina u Y smjeru":trenutni_vy3,
                                    "Udaljenost izmedu Tijela 1 i 2":r_12,
                                    "Udaljenost izmedu Tijela 1 i 3":r_13,
                                    "Udaljenost izmedu Tijela 2 i 3":r_23,
                                    })
                    dataframe.to_csv(os.path.join(os.getcwd(),f"Simulacije/{filename_counter}.csv"),index=False)
                    #povecat counter
                    filename_counter = filename_counter + 1
                #pbar update nakon sto je cijeli lokalni spremnik gotov
                pbar.update(1)

        print()
        print()
        print("ALL FILES SAVED.")