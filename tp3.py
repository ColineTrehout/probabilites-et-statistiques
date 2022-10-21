#Alice Gydé et Coline Trehout

#-----------------------------------------------------------------------------
# TP 3 : Intervalles de confiance
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy
import math

#Fonction calcul de la moyenne empirique
def moyenne(l):
    moy = sum(l)/len(l)
    return moy

#Fonction calcul de la variance empirique
def variance (l):
    moy = moyenne(l)
    taille = len(l)
    variance = 0
    for i in range(taille):
        variance = variance + pow(l[i]-moy,2)
    variance = variance/(taille)
    return variance

#Fonction calcul de l'intervalle de confiance avec variance inconnue
def intervalle_confiance (l, niveau):
    moy = moyenne(l)
    #fonction ppf (percent point fonction) pour la loi de student
    fractile = scipy.t.ppf(1 - niveau/2, df = len(l) - 1, loc=0, scale=1)
    t = math.sqrt(len(l))
    ecart_type = math.sqrt(variance(l))
    return [moy-(fractile*(ecart_type/t)),moy+(fractile*(ecart_type/t))]

#Fonction calcul de l'intervalle de confiance avec variance connue
def intervalle_confiance2 (l, niveau, var):
    moy = moyenne(l)
    #fonction ppf (percent point fonction) pour la loi normale
    fractile = scipy.norm.ppf(1 - niveau/2, loc=0, scale=1)
    t = math.sqrt(len(l))
    ecart_type = math.sqrt(var)
    return [moy-(fractile*(ecart_type/t)),moy+(fractile*(ecart_type/t))]


#-----------------------------------------------------------------------------
# Programme principal
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Problème 1
#-----------------------------------------------------------------------------

print("Problème 1 :")
print("Pot de confiture (kg) : ")

#données
p_confiture = [0.499, 0.509, 0.501, 0.494, 0.498, 0.497, 0.504, 0.506, 0.502, 0.496, 0.495, 0.493, 0.507, 0.505, 0.503, 0.491]

print("Moyenne empirique : " ,moyenne(p_confiture))

#Histogramme des fréquences
plt.hist(p_confiture, bins = np.arange(np.min(p_confiture), np.max(p_confiture)+0.001, 0.001), rwidth = 0.5, edgecolor = 'black', density = True, alpha = 0.4)
plt.xlabel('Poids/kg')
plt.ylabel('Nombre de pots')
plt.title('Histogramme du problème 1')
plt.show()

print("Intervalle de confiance à 95% : ",intervalle_confiance(p_confiture,0.05))
print("Intervalle de confiance à 99% : ",intervalle_confiance(p_confiture,0.01))

#données
p_avocat = [85.06, 91.44, 87.93, 89.02, 87.28, 82.34, 86.23, 84.16, 88.56, 90.45, 84.91, 89.90, 85.52, 86.75, 88.54, 87.90]

print("Avocat (g) : ")
print("Intervalle de confiance à 95% : ",intervalle_confiance(p_avocat,0.05))


#-----------------------------------------------------------------------------
# Problème 2
#-----------------------------------------------------------------------------

print("\nProblème 2 :")
#Remplissage de la population en fonction de la satisfaction (1 = satisfait, 0 = insatisfait)
satisfaction = []
for i in range(500) :
    if (i <= 95):
        satisfaction.append(1)
    else : 
        satisfaction.append(0)
print("Intervalle de confiance à 99% : ",intervalle_confiance(satisfaction,0.01))


#-----------------------------------------------------------------------------
# Problème 3
#-----------------------------------------------------------------------------

print("\nProblème 3 :")
print("Taille de l'echantillon ?")
#Saisi par l'utilisateur (conversion forcée en int)
n = int(input())
echantillon = scipy.bernoulli.rvs(0.5, size = n)
#Pour une loi de Bernoulli var(X) = p*(1-p)
var = 0.5*0.5
print(f"Pour {n}, intervalle de confiance à 95% : ", intervalle_confiance2(echantillon, 0.05 ,var))
