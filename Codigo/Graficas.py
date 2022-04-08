import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def grafica_1(frec, r, frec_corte):
    ii = frec > frec_corte
    
    x = frec[ii]/1E6
    y = r[ii]*1E3

    lab_x = "Frecuencia (MHz)"
    lab_y = "Amplitud (mV)"

    plt.figure(figsize=(8,4.5))
    plt.plot(x, y)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia +In para n=10000")
    plt.savefig("Fig1-Datos1.png")
    return None

def grafica_2(frec, r):
    ii = r > 10E-6
    frec_corte = frec[ii][-1]
    
    ii = frec > frec_corte
    
    x = frec[ii]/1E6
    y = r[ii]*1E6
    
    frec_corte = x[0]
    
    lab_x = "Frecuencia (MHz)"
    lab_y = "Amplitud ($\mu$V)"

    plt.figure(figsize=(8,4.5))
    plt.plot(x, y)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia +In para n=10000, $f > f_{corte}$ " + "= {:.2f} MHz".format(frec_corte))
    plt.savefig("Fig2-Datos1.png")
    return frec_corte

def grafica_3(frec, r, frec_corte):
    ii = (0 < frec) & (frec < frec_corte*1E6)
    
    x = frec[ii]/1E3
    y = r[ii]*1E3

    lab_x = "Frecuencia (kHz)"
    lab_y = "Amplitud (mV)"

    plt.figure(figsize=(8,4.5))
    plt.plot(x, y)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia +In para n=10000, $f < f_{corte}$ " + "= {:.2f} MHz".format(frec_corte))
    plt.savefig("Fig3-Datos1.png")
    return None

def grafica_4(frec_1, r_1, frec_corte):
    ii = frec_1 > frec_corte

    x1 = frec_1[ii]
    y1 = r_1[ii]*1E3

    plt.figure(figsize=(14,4.5))
    
    plt.subplot(1, 2, 1)
    lab_x = "Frecuencia (MHz)"
    lab_y = "Amplitud (mV)"
    plt.plot(x1/1E6, y1)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    
    plt.subplot(1, 2, 2)
    lab_x = "Frecuencia (Hz)"
    lab_y = "Amplitud (mV)"
    plt.plot(x1, y1)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.xscale("log")

    plt.suptitle("Amplitud vs Frecuencia +Out para n=10000")
    plt.savefig("Fig4-Datos2.png")
    return None


df1 = pd.read_csv("Datos1.csv", sep=";")
df1 = np.array(df1)

df2 = pd.read_csv("Datos2.csv", sep=";")
df2 = np.array(df2)

#Datos 1

frec = df1[7,4:]
r = df1[14,4:]

frec_corte = 0.1
grafica_1(frec, r, frec_corte)

frec_corte = grafica_2(frec, r)
grafica_3(frec, r, frec_corte)

#Datos 2

frec_1 = df2[7,4:]
r_1 = df2[14,4:]

frec_corte = 0.1
grafica_4(frec_1, r_1, frec_corte)

