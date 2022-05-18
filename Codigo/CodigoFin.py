import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def Datos1(frec_1, r_1):
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
        
        print("mu 1 = {} E-6 V".format(np.mean(y)))
        print("sigma 1 = {} E-6 V".format(np.std(y)))
        
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
        
        pos = 4
        
        mu = np.mean(y[pos:])
        sigma = np.std(y[pos:])
        
        print()
        print("frec = {} E3 Hz".format(x[pos]))
        
        print("mu 2 = {} E-3 V".format(mu))
        print("sigma 2 = {} E-3 V".format(sigma))
        
        lab_x = "Frecuencia (kHz)"
        lab_y = "Amplitud (mV)"
        
        def funcion(x, a, b, c):
            return a*x**(-b) + c

        popt,pcov = curve_fit(funcion, x, y)
        
        print()
        print("a = {}".format(popt[0]))
        print("b = {}".format(popt[1]))
        print("c = {}".format(popt[2]))
        
        cant = 10000
        x_ajuste = np.linspace(x[0], x[-1], cant)
        y_ajuste = funcion(x_ajuste, popt[0], popt[1], popt[2])
        
        plt.figure(figsize=(8,4.5))
        plt.scatter(x, y, s=15, label="Datos Experimentales")
        plt.plot(x_ajuste, y_ajuste, c="r", label="Ajuste")
        plt.xlabel(lab_x)
        plt.ylabel(lab_y)
        plt.legend()
        plt.title("Amplitud vs Frecuencia +In para n=10000, $f < f_{corte}$ " + "= {:.2f} MHz".format(frec_corte))
        plt.savefig("Fig3-Datos1.png")
        
        plt.figure()
        plt.figure(figsize=(8,4.5))
        plt.scatter(x, y - funcion(x, popt[0], popt[1], popt[2]), s=15)
        plt.xlabel(lab_x)
        plt.ylabel(lab_y)
        plt.title("Residuos del Ajuste Realizado")
        plt.savefig("Fig4-Datos1.png")
        
        return None

    frec_corte = 0.1
    grafica_1(frec_1, r_1, frec_corte)

    frec_corte = grafica_2(frec_1, r_1)
    grafica_3(frec_1, r_1, frec_corte)
    return None

def Datos2(frec_2, r_2): 
    def grafica_4(frec_1, r_1, frec_corte):
        ii = frec_1 > frec_corte

        x1 = frec_1[ii]
        y1 = r_1[ii]*1E3
        
        plt.figure()
        lab_x = "Frecuencia (MHz)"
        lab_y = "Amplitud (mV)"
        plt.plot(x1/1E6, y1, c="black")
        plt.xlabel(lab_x)
        plt.ylabel(lab_y)
        plt.axvline(1E-3, c="darkred", linestyle="--", label="1 kHz")
        plt.axvline(25, c="red", linestyle="--", label="25 MHz")
        plt.axvline(50, c="lightcoral", linestyle="--", label="50 MHz")
        plt.title("Amplitud vs Frecuencia +Out para n=10000")
        plt.legend()
        plt.savefig("Fig1-Datos2.png")

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
        plt.savefig("Fig2-Datos2.png")
        return None

    frec_corte = 0.1
    grafica_4(frec_2, r_2, frec_corte)
    return None

def Datos3():
    def gen_interv():
        interv = np.ones((len(archivos),2))
        interv[:,0] = 9
        interv[:,1] = 11
        l = len(interv)
        
        for i in range(l):
            if i == l-2:
                interv[i,0] = 4.65*10**(i+1)
                interv[i,1] = 5.4*10**(i+1)
            elif i == l-1:
                interv[i,0] *= 10**(i)
                interv[i,1] *= 10**(i)
            else:   
                interv[i,0] *= 10**(i+1)
                interv[i,1] *= 10**(i+1)
        
        return interv

    def gauss(x, a, mu0, sigma):
        y = a * np.exp(-(x - mu0)**2 / (2 * sigma**2))
        return y

    def ajuste_gauss(x, y):
        mu0 = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mu0)**2) / sum(y))
        peak = max(y) 
        p0 = [peak, mu0, sigma]  
        
        popt,pcov = curve_fit(gauss, x, y, p0)
        return popt,pcov

    archivos = ["FFT_100Hz.csv", "FFT_1kHz.csv", "FFT_10kHz.csv", "FFT_100kHz.csv", "FFT_1MHz.csv", "FFT_5MHz.csv", "FFT_10MHz.csv"]
    titulos = ["100 Hz", "1 kHz", "10 kHz", "100 kHz", "1 MHz", "5 MHz", "10 MHz"]
    interv = gen_interv()

    cant = len(archivos)
    sigmas = np.zeros(cant)
    BW = np.zeros(cant)
    cortes_frec = [1E2, 1E3, 1E4, 1E5, 1E6, 5E6, 1E7]

    for i in range(cant):
        df = pd.read_csv(archivos[i])
        df = np.array(df)
        
        frec = df[:,0]
        trasnf_fourier = df[:,1]
        trasnf_fourier -= np.min(trasnf_fourier)
                
        plt.figure(figsize=(14,4.5))
        plt.suptitle("Transformada Fourier para $f = ${} ".format(titulos[i]))
        
        plt.subplot(1, 2, 1)
        plt.plot(frec, trasnf_fourier)
        
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud (dB)")
        
        ii = (interv[i,0] <= frec) & (frec <= interv[i,1])
        x = frec[ii]
        y = trasnf_fourier[ii]
        
        popt,pcov = ajuste_gauss(x, y)
        sigmas[i] = popt[2]
        
        BW[i] = 2*sigmas[i]*np.sqrt(np.log(2))/cortes_frec[i]
        
        x_gauss = np.linspace(interv[i,0], interv[i,1], 1000)
        y_gauss = gauss(x_gauss, popt[0], popt[1], popt[2])
        
        plt.subplot(1, 2, 2)
        plt.scatter(x, y, label="FFT")
        plt.plot(x_gauss, y_gauss, c="r", label="Ajuste")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud (dB)")
        plt.legend()
        plt.savefig("Fig{}_Datos3.png".format(i + 1))


    plt.figure()
    plt.scatter(cortes_frec, BW)
    plt.title("Ancho de banda relativo vs Frecuencia")
    plt.xscale("log")
    plt.xlabel("f")
    plt.ylabel("BW")
    plt.savefig("Fig8_Datos3.png")
    
    print("mean = {}".format(np.mean(BW)))
    print("std = {}".format(np.std(BW)))
    
    return None

def Datos4(frec4_in, r4_in, frec4_out, r4_out):
    frec4_in *= 1E-6
    frec4_out *= 1E-6
    r4_in *= 1E3
    r4_out *= 1E3
    
    lab_x = "Frecuencia (MHz)"
    lab_y = "Amplitud (mV)"

    plt.figure(figsize=(8,4.5))
    plt.plot(frec4_in[1:], r4_in[1:])
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia solo conexión +In-Top para n=10000")
    plt.savefig("Fig1-Datos4.png")
    
    plt.figure(figsize=(8,4.5))
    plt.plot(frec4_out[1:], r4_out[1:])
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia solo conexión +Out-Bottom para n=10000")
    plt.savefig("Fig2-Datos4.png")
    return None

def Datos5(frec4_in, r4_in, frec4_out, r4_out, frec5_in, r5_in, frec5_out, r5_out, frec_1, r_1, frec_2, r_2):
    lab_x = "Frecuencia (MHz)"
    lab_y = "Amplitud (mV)"

    plt.figure(figsize=(8,4.5))
    plt.plot(frec5_in[1:]*1E-6, r5_in[1:]*1E3)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia +In-Top Montaje para n=10000")
    plt.savefig("Fig1-Datos5.png")
    
    plt.figure(figsize=(8,4.5))
    plt.plot(frec5_out[1:]*1E-6, r5_out[1:]*1E3)
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.title("Amplitud vs Frecuencia +Out-Bottom Montaje para n=10000")
    plt.savefig("Fig2-Datos5.png")
    
    ii = frec_1 < 10E6
    frec_1 = frec_1[ii]
    r_1 = r_1[ii]
        
    plt.figure(figsize=(8,4.5))
    plt.plot(frec_1[1:]*1E-6, r_1[1:]*1E3, label="Puerto +In")
    plt.plot(frec4_in[1:]*1E-6, r4_in[1:]*1E3, label="Conexión +In-Top")
    plt.plot(frec5_in[1:]*1E-6, r5_in[1:]*1E3, label="Montaje Completo")
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.legend()
    plt.title("Amplitud vs Frecuencia")
    plt.savefig("Fig3-Datos5.png")
    
    ii = frec_2 < 10E6
    frec_2 = frec_2[ii]
    r_2 = r_2[ii]
    
    plt.figure(figsize=(14,4.5))
    
    plt.subplot(1, 2, 1)
    plt.plot(frec_2[1:]*1E-6, r_2[1:]*1E3, label="Puerto +Out")
    plt.plot(frec4_out[1:]*1E-6, r4_out[1:]*1E3, label="Conexión +Out-Bottom")
    plt.plot(frec5_out[1:]*1E-6, r5_out[1:]*1E3, label="Montaje Completo")
    plt.xlabel(lab_x)
    plt.ylabel(lab_y)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(frec_2[1:], r_2[1:]*1E3, label="Puerto +Out")
    plt.plot(frec4_out[1:], r4_out[1:]*1E3, label="Conexión +Out-Bottom")
    plt.plot(frec5_out[1:], r5_out[1:]*1E3, label="Montaje Completo")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel(lab_y)
    plt.xscale("log")
    plt.legend()
    
    plt.suptitle("Amplitud vs Frecuencia")
    plt.savefig("Fig4-Datos5.png")
    
    return None

f_rms = 1.4

#Datos 1
df1 = pd.read_csv("Datos1.csv", sep=";")
df1 = np.array(df1)

frec_1 = df1[7,4:]
r_1 = df1[14,4:]*f_rms

#Datos1(frec_1, r_1)

#Datos 2

df2 = pd.read_csv("Datos2.csv", sep=";")
df2 = np.array(df2)

frec_2 = df2[7,4:]
r_2 = df2[14,4:]*f_rms

#Datos2(frec_2, r_2)

#Datos 3

#Datos3()

#Datos 4

df4_in = pd.read_csv("Datos4-IN.csv", sep=";")
df4_in = np.array(df4_in)

df4_out = pd.read_csv("Datos4-OUT.csv", sep=";")
df4_out = np.array(df4_out)

frec4_in = df4_in[7,4:]
r4_in = df4_in[14,4:]*f_rms

frec4_out = df4_out[7,4:]
r4_out = df4_out[14,4:]*f_rms

#Datos4(frec4_in, r4_in, frec4_out, r4_out)

#Datos 5

df5_in = pd.read_csv("Datos5-IN.csv", sep=";")
df5_in = np.array(df5_in)

df5_out = pd.read_csv("Datos5-OUT.csv", sep=";")
df5_out = np.array(df5_out)

frec5_in = df5_in[7,4:]
r5_in = df5_in[14,4:]*f_rms

frec5_out = df5_out[7,4:]
r5_out = df5_out[14,4:]*f_rms

#Datos5(frec4_in, r4_in, frec4_out, r4_out, frec5_in, r5_in, frec5_out, r5_out, frec_1, r_1, frec_2, r_2)