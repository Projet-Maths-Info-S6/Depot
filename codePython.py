import pandas as pd
import librosa
from librosa import display
import numpy as np
import IPython.display as ipd
from IPython.display import Audio
from glob import glob
import matplotlib.pyplot as plt
from moviepy.editor import *
from pydub import AudioSegment
import scipy as scipy
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
import os
from scipy.spatial import distance
import time
import h5py
import random
import soundfile



def get_fft(data):
   yf=rfft(data)
   return yf



#Affiche la transformée de fourier
def display_fft(fft):
   N=duration*sampling_rate
   N=int(N)
   yf=fft
   yf=yf[:len(yf)//2]
   xf=rfftfreq(N//2,1/sampling_rate)
   xf=np.delete(xf,len(xf)-1)
   plt.plot(xf, np.abs(yf))
   plt.show()
   return yf

def audiosegment_to_librosawav(audiosegment):    
    samples = audiosegment.get_array_of_samples()
    arr = np.array(samples).astype(np.float32)/32768 # 16 bit 
    arr = librosa.core.resample(arr, audiosegment.frame_rate*2, 22050, res_type='kaiser_best') 
    return arr



def get_time_codes(samples,sampling_rate,coeff,nameTC):      
   duration = librosa.get_duration(y=samples, sr=sampling_rate)
   tempo, beats = librosa.beat.beat_track(y=samples, sr=sampling_rate)
   tab=librosa.frames_to_time(beats, sr=sampling_rate)
   new_tab=[]
   i=0
   if coeff>=1:
       for t in tab:
           if i%coeff==0:
               new_tab.append(t)
           i+=1
   if coeff<=1:
       while i<len(tab)-1:
           new_tab.append(tab[i])
           dist=(tab[i+1]-tab[i])
           invert_coeff=int(1/coeff)
           for k in range (1,invert_coeff):
               new_tab.append(tab[i]+dist*coeff*k)
           i+=1
   np.save(nameTC,new_tab) 
   return new_tab


def get_all_fft(file_path,coeff,nameTC):
    samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None ) 
    time_codes=get_time_codes(samples,sampling_rate,coeff,nameTC)
    fft_tab=[]
    min_len_fft=23000
    for i in range (1,len(time_codes)):
        newAudio = AudioSegment.from_wav(file_path)
        newAudio = newAudio[time_codes[i-1]*1000:time_codes[i]*1000]
        data = audiosegment_to_librosawav(newAudio)
        fft=get_fft(data)
        if i==1:    #On cherche le fft de longueur minimale
            min_len_fft=len(fft)
        else:
            if min_len_fft>len(fft):
                min_len_fft=len(fft)
        fft_tab.append(fft)     #On retire à tous les fft leurs derniers éléments afin qu'ils aient tous la même longueure égale à celle du fft de taille minimale
    fft_tab_norm=normalize_size(fft_tab, min_len_fft)
    print("Taille du plus cours fft =",min_len_fft)
    return fft_tab_norm


def get_all_chromas(file_path,coeff):
    samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None ) 
    time_codes=get_time_codes(samples,sampling_rate,coeff)
    chromas_tab=[]
    min_len_chroma=2
    for i in range (1,len(time_codes)):
        newAudio = AudioSegment.from_wav(file_path)
        newAudio = newAudio[time_codes[i-1]*1000:time_codes[i]*1000]
        data = audiosegment_to_librosawav(newAudio)
        chroma=librosa.feature.chroma_cqt(y=samples, sr=sampling_rate)
        if i==1:    #On cherche le fft de longueur minimale
            min_len_chroma=len(chroma)
        else:
            if min_len_chroma>len(chroma):
                min_len_chroma=len(chroma)
        chromas_tab.append(chroma)     
    #chroma_tab_norm=normalize_size(chromas_tab, min_len_chroma)
    print("Taille du plus cours fft =",min_len_chroma)
    return chromas_tab

def get_all_mfcc(file_path,coeff):
    samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0,duration=None ) 
    time_codes=get_time_codes(samples,sampling_rate,coeff)
    mfcc_tab=[]
    min_len_mfcc=2
    for i in range (1,len(time_codes)):
        newAudio = AudioSegment.from_wav(file_path)
        newAudio = newAudio[time_codes[i-1]*1000:time_codes[i]*1000]
        data = audiosegment_to_librosawav(newAudio)
        mfcc=librosa.feature.mfcc(y=samples, sr=sampling_rate)
        if i==1:    #On cherche le fft de longueur minimale
            min_len_mfcc=len(mfcc)
        else:
            if min_len_mfcc>len(mfcc):
                min_len_mfcc=len(mfcc)
        mfcc_tab.append(mfcc)     
    #chroma_tab_norm=normalize_size(chromas_tab, min_len_chroma)
    print("Taille du plus cours mfcc =",min_len_mfcc)
    return mfcc_tab


        
def normalize_size(T,mini):
    normalized_tab=[]
    for tab in T:
        normalized_tab.append(tab[:mini])
    return normalized_tab

def distance_eucl(tab1,tab2):  #A optimiser
    temp=[]
    for i in range(0,len(tab1)-1):
        e=(tab1[i]-tab2[i])**2
        temp.append(e)
    s=0
    for e in temp:
        s=s+e
    s=np.abs(s)
    if s==0.0:
        return 1.0
    s=np.log(s)
    return s

def distanceKL(tab1,tab2):
    s=0
    for i in range (0,len(tab1)-1):
        if tab2[i]==0:
            s+=np.nan()
        else:     
            s+=(tab1[i]*np.log(tab1[i]/tab2[i])-tab1[i]+tab2[i])
    s=np.abs(s)
    return s


def distanceIS(tab1,tab2):
    s=0
    for i in range (0,len(tab1)-1):
        if tab2[i]==0:
            s+=np.nan()
        else:
            s+=(tab1[i]/tab2[i]-np.log(tab1[i]/tab2[i])-1)
    s=np.abs(s)
    return s


def make_sym(matrice):
    height=np.shape(matrice)[0]
    for i in range(0,height):
        for j in range(i,height):
            matrice[j][i]=matrice[i][j]              
    return matrice


def matrice_dist (tab_fft,type_distance):
    matrice=np.ones((len(tab_fft),len(tab_fft)))
    matrice=matrice*(-1)
    print("nombre de fft ",len(tab_fft))
    for i in range(0,len(tab_fft)-2):
        for j in range(i+2,len(tab_fft)-1):
            if type_distance=='eucl':
                matrice[i][j]=distance_eucl(tab_fft[i],tab_fft[j])
            if type_distance=='IS':
                matrice[i][j]=distanceKL(tab_fft[i],tab_fft[j])
            if type_distance=='KL':
                matrice[i][j]=distanceIS(tab_fft[i],tab_fft[j])
    return matrice


    


def dist_3_by_3(tab_fft,type_dist):
    matrice=np.ones((len(tab_fft),len(tab_fft)))
    matrice=matrice*(-1)
    for i in range(0,len(tab_fft)-1):
        if i==0:                #On gère les cas où les intervalles en questions sont sur les extrémités et donc ne sont pas entouré par d'autres
            vector1=np.concatenate((tab_fft[i],tab_fft[i+1],tab_fft[i+2]))      
        if i==len(tab_fft)-1:
            vector1=np.concatenate((tab_fft[i-2],tab_fft[i-1],tab_fft[i]))
        else:
            vector1=np.concatenate((tab_fft[i-1],tab_fft[i],tab_fft[i+1]))
        for j in range(i,len(tab_fft)-1):
            if j==0:               
                vector2=np.concatenate((tab_fft[j],tab_fft[j+1],tab_fft[j+2]))      
            if j==len(tab_fft)-1:
                vector2=np.concatenate((tab_fft[j-2],tab_fft[j-1],tab_fft[j]))
            else:
                vector2=np.concatenate((tab_fft[j-1],tab_fft[j],tab_fft[j+1]))
                
            if type_distance=='eucl':
                matrice[i][j]=distance_eucl(vector1,vector2)
            if type_distance=='IS':
                matrice[i][j]=distanceKL(vector1,vector2)
            if type_distance=='KL':
                matrice[i][j]=distanceIS(vector1,vector2)
    return matrice



def minimum(matrice):
    mini=[]
    for i in range(0,len(matrice)):
        liste=matrice[i]
        if i==0:
            m=(liste[1],1)
        else:   
            m=(liste[0],0)
        for k in range (0,len(liste)):
            if liste[k]<m[0] and liste[k]>0.001 and liste[k]!=-1:
                m=(liste[k],k)
        mini.append(m)
    return mini
        
def tri(seuil,matrice):
    matrice=minimum(matrice)
    trie=[]
    for j in range (0,len(matrice)):
        if matrice[j][0]<=seuil:
            trie.append((j,matrice[j][1]))
    return trie


            
def display(matrice):
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
    caxes = axes.matshow(matrice,interpolation ='nearest')
    figure.colorbar(caxes) 
    plt.show()

def inter_to_tc(inter,tc):
    newtab=[]
    for e in inter:
        a=tc[e[0]]
        b=tc[e[1]]
        ntc=(a,b)
        newtab.append(ntc)
    return newtab

def proba(prob):
    rand=random.random()
    if rand<prob:
        return True
    else:
        return False
    

def generate(file_path,newtimecodes,length,prob):
    orig, sr = librosa.load(file_path)
    if len(newtimecodes)==0:
        print('Aucune similarité détecté avec ces paramètres')
        return False
    tab_duree=[]
    tab_intervalle=[]
    duree=0
    newAudio = AudioSegment.from_wav(file_path)
    finalArray=[]
    debut=newtimecodes[0][1]
    duree+=newtimecodes[0][0]
    tab_duree.append(int(duree))
    while duree<length:
        
        fin=newtimecodes[0][0]
        j=0
        while fin<debut+0.001 or proba(prob)==False:
            if j==len(newtimecodes)-1:
                j=0
            j+=1
            fin=newtimecodes[j][0]
        intervalle = newAudio[debut*1000:fin*1000]
        array = audiosegment_to_librosawav(intervalle)  #On transforme l'audio en un tableau exploitable par librosa
        out = array.copy()                              #La copie permet de ne pas modifier le fichier originel
        apply_fadein(out, sr, 0.1)
        apply_fadeout(out, sr, 0.1)
        finalArray=np.concatenate((finalArray,out))     
        duree+=abs(fin-debut)
        tab_intervalle.append((debut,fin))
        tab_duree.append(int(duree))
        debut=newtimecodes[j][1]
        
    finalSegment=newAudio[debut*1000:newAudio.duration_seconds*1000]
    endArray = audiosegment_to_librosawav(intervalle) 
    endOut = endArray.copy()                              
    apply_fadein(endOut, sr, 0.1)
    finalArray=np.concatenate((finalArray,endOut))
    soundfile.write('InfiniteAudio.wav', finalArray, samplerate=sr)
    return tab_duree,tab_intervalle


def interface_durees(tab_intervalle,timecodes):
    intervalle=[]
    for e in tab_intervalle:
        i=0
        while e[0]!=timecodes[i]:
            i+=1
        while e[1]!=timecodes[i]:
            intervalle.append((i,timecodes[i+1]-timecodes[i]))
            i+=1
    return intervalle
        

def generate_mix_2_musiques(file_path1,file_path2,jumps1,jumps2,length,prob):
    orig1, sr1 = librosa.load(file_path1)
    orig2, sr2 = librosa.load(file_path2)
    if len(jumps1)==0 and len(jumps2)==0:
        print('Aucune similarité détecté avec ces paramètres')
        return False
    tab_duree=[]
    duree=0
    musique1 = AudioSegment.from_wav(file_path1)
    musique2 = AudioSegment.from_wav(file_path2)
    finalArray=[]
    debut=0
    while duree<length:
        fin=jumps1[0][0]
        j=0
        print(debut)
        while fin<debut+0.001 or proba(prob)==False:
            if j==len(jumps1)-1:
                inter=musique1[debut*1000:]
                array = audiosegment_to_librosawav(inter)  
                out = array.copy()
                apply_fadein(out, sr2, 0.1)
                apply_fadeout(out, sr2, 0.1)
                finalArray=np.concatenate((finalArray,out))     
                duree+=abs(fin-debut)
                debut=0
                j=0
            else:
                j+=1
                fin=jumps1[j][0]


            
        intervalle = musique1[debut*1000:fin*1000]
        array = audiosegment_to_librosawav(intervalle)  
        out = array.copy()
        apply_fadein(out, sr1, 0.1)
        apply_fadeout(out, sr1, 0.1)
        finalArray=np.concatenate((finalArray,out))     
        duree+=abs(fin-debut)
        tab_duree.append(int(duree))
        
        if duree<length:
            debut=jumps1[j][1]
            fin=jumps2[0][0]
            j=0
            while fin<debut+0.001 or proba(prob)==False:
                if j==len(jumps2)-1:
                    inter=musique2[debut*1000:]
                    array = audiosegment_to_librosawav(inter)  
                    out = array.copy()
                    apply_fadein(out, sr2, 0.1)
                    apply_fadeout(out, sr2, 0.1)
                    finalArray=np.concatenate((finalArray,out))     
                    duree+=abs(fin-debut)
                    debut=0
                    j=0
                else:
                    j+=1
                    fin=jumps2[j][0]
            intervalle2 = musique2[debut*1000:fin*1000]
            array = audiosegment_to_librosawav(intervalle2)  
            out = array.copy()
            apply_fadein(out, sr2, 0.1)
            apply_fadeout(out, sr2, 0.1)
            finalArray=np.concatenate((finalArray,out))     
            duree+=abs(fin-debut)
            tab_duree.append(int(duree))
        debut=jumps2[j][1]
    soundfile.write('Mix2musiques.wav', finalArray, samplerate=sr1)
    return tab_duree  







def apply_fadeout(audio, sr, duration):
    length = int(duration*sr)
    end = audio.shape[0]
    start = end - length
    fade_curve = x = np.linspace(-np.pi/2,0,length)
    fade_curve = -np.sin(x)
    audio[start:end] = audio[start:end] * fade_curve
    
    
def apply_fadein(audio, sr, duration):
    length = int(duration*sr)
    end = length
    start = 0
    fade_curve = np.linspace(0.0, np.pi/2, length)
    fade_curve = np.sin(fade_curve)
    audio[start:end] = audio[start:end] * fade_curve    
    


def normalize_2_musics(tab1,tab2):
    if len(tab1[0])<len(tab2[0]):
        tab2=normalize_size(tab2, len(tab1[0]))
    else:
        tab1=normalize_size(tab1, len(tab2[0]))
    return tab1,tab2
        
def matrice_dist_2musiques(vectors1,vectors2):
    matrice=np.ones((len(vectors1),len(vectors2)))
    matrice=matrice*(-1)
    print("nombre de vecteurs du morceau 1: ",len(vectors1))
    print("nombre de vecteurs du morceau 2: ",len(vectors2))
    for i in range(0,len(vectors1)-1):
        for j in range(0,len(vectors2)-1):
            matrice[i][j]=distance_eucl(vectors1[i], vectors2[j])
    return matrice


def minimums_matrice_2_musiques(matrice):
    mini1=[]
    mini2=[]
    for i in range (0,matrice.shape[0]):
        liste=matrice[i]
        if i==0:
            m=(liste[1],1)
        else:   
            m=(liste[0],0)
        for k in range (0,matrice.shape[1]):
            if liste[k]<m[0] and liste[k]>0.001 and liste[k]!=-1:
                m=(liste[k],k)
        mini1.append(m)
    for i in range (0,matrice.shape[1]):
        if i==0:
            m=(matrice[0][i],1)
        else:   
            m=(matrice[0][0],0)
        for k in range (0,matrice.shape[0]):
            if matrice[k][i]<m[0] and matrice[k][i]>0.001 and matrice[k][i]!=-1:
                m=(matrice[k][i],k)
        mini2.append(m)
    return mini1,mini2

def seuil(matrice_mini,seuil):
    trie=[]
    for i in range(0,len(matrice_mini)-1):
        if matrice_mini[i][0]<seuil:
            trie.append((i,matrice_mini[i][1]))
    return trie
            

def inter_to_tc_2(inter, tc1,tc2):
    newtab=[]
    for e in inter:
        a=tc1[e[0]]
        b=tc2[e[1]]
        ntc=(a,b)
        newtab.append(ntc)
    return newtab
        

def matrice_tkinder (matrice):
    root = tk.Tk()
    root.grid_rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    frame_main = tk.Frame(root, bg="gray")
    frame_main.grid(sticky='news')


    # Create a frame for the canvas with non-zero row&column weights
    frame_canvas = tk.Frame(frame_main)
    frame_canvas.grid(row=2, column=0, pady=(5, 0), sticky='nw')
    frame_canvas.grid_rowconfigure(0, weight=1)
    frame_canvas.grid_columnconfigure(0, weight=1)
    # Set grid_propagate to False to allow 5-by-5 buttons resizing later
    frame_canvas.grid_propagate(False)

    # Add a canvas in that frame
    canvas = tk.Canvas(frame_canvas, bg="yellow")
    canvas.grid(row=0, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame_canvas, orient="vertical", command=canvas.yview)
    hsb = tk.Scrollbar(frame_canvas, orient="horizontal", command=canvas.yview)
    vsb.grid(row=0, column=1, sticky='ns')
    hsb.grid(row=1, column=0, sticky='ew')
    canvas.configure(yscrollcommand=vsb.set,xscrollcommand=hsb.set)
    # Create a frame to contain the buttons
    frame_buttons = tk.Frame(canvas, bg="blue")
    canvas.create_window((0, 0), window=frame_buttons, anchor='nw')

    # Add 9-by-5 buttons to the frame
    rows = 20#len(matrice)
    columns = 20#len(matrice[0])
    buttons = [[tk.Button() for j in range(columns)] for i in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            buttons[i][j] = tk.Button(frame_buttons, text=(matrice[i][j]))
            buttons[i][j].grid(row=i, column=j, sticky='news')

    # Update buttons frames idle tasks to let tkinter calculate buttons sizes
    frame_buttons.update_idletasks()

    # Resize the canvas frame to show exactly 5-by-5 buttons and the scrollbar
    first5columns_width = sum([buttons[0][j].winfo_width() for j in range(0, int(columns/2) )])
    first5rows_height = sum([buttons[i][0].winfo_height() for i in range(0, int(rows/2) )])
    frame_canvas.config(width=first5columns_width + vsb.winfo_width(),
    height=first5rows_height)

    # Set the canvas scrolling region
    canvas.config(scrollregion=canvas.bbox("all"))

    # Launch the GUI
    root.mainloop()
    
""" 
    
file_path="hiver.wav"


file_path2="printemps.wav"    #Generer 2 musqiques

fft1=get_all_fft(file_path,1,'tc1')
fft2=get_all_fft(file_path2,1,'tc2')

fft1,fft2 = normalize_2_musics(fft1,fft2)

mat = matrice_dist_2musiques(fft1, fft2)
np.save("mat2mus",mat)

m=np.load("mat2mus.npy")
mini1,mini2 = minimums_matrice_2_musiques(m)

mini1=seuil(mini1, 5)
mini2=seuil(mini2, 5)

tc1=np.load("tc1.npy")
tc2=np.load("tc2.npy")

ntc1 = inter_to_tc_2(mini1,tc1,tc2)
ntc2 = inter_to_tc_2(mini2,tc2,tc1)

print(generate_mix_2_musiques(file_path, file_path2, ntc1, ntc2, 300, 0.5))
"""


nom=input('Nom du fichier audio (ne pas oubliter le .wav) = ')

file_path="./"+nom




coeff=int( input('Nombre de beats par portion = '))
tps1=time.time()
test = get_all_fft(file_path,coeff,'tc')
tps2=time.time()
print("temps pour récupérer les ffts :",tps2-tps1," secondes")
mat=matrice_dist(test,'eucl')
np.save('matrice_dist_10.txt', mat)



tps3=time.time()
mm=np.load('matrice_dist_10.txt.npy')
mm=make_sym(mm)
#print("temps pour comparer tous les ffts :",tps3-tps2," secondes")
print(np.shape(mm))
tc=np.load('tc.npy')
display(mm)
seuil=float(input("Seuil au delà duquel un saut n'est pas effectué = "))
t=tri(seuil,mm)
ntc=inter_to_tc(t,tc)
duree=int(input("Durée du son généré (en secondes) = "))
prob=float(input("Probabilité d'effectuer un saut = "))
tab_duree,tab_intervalle=generate(file_path,ntc,duree,prob)
print(tab_duree)

interface_duree=interface_duree(tab_intervalle,tc)
tps4=time.time()
print("temps pour générer le morceau de 5min :",tps4-tps3," secondes")


