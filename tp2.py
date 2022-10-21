#Alice Gydé et Coline Trehout

#-----------------------------------------------------------------------------
# TP 2 : Étude et manipulation de lois de probabilités
#-----------------------------------------------------------------------------

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.stats as spicy
from scipy.stats import binom
import math as m


#-----------------------------------------------------------------------------
# 2.1. Loi Binomiale
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Données
#-----------------------------------------------------------------------------

#nombre d'expériences
n1 = 30
n2 = 30
n3 = 50

#probabilité de succès
p1 = 0.5
p2 = 0.7
p3 = 0.4

#-----------------------------------------------------------------------------
# Affichage des courbes
#-----------------------------------------------------------------------------

courbe1 = [binom.pmf(x, n1, p1) for x in range (100)]
courbe2 = [binom.pmf(x, n2, p2) for x in range (100)]
courbe3 = [binom.pmf(x, n3, p3) for x in range (100)]

plt.plot(courbe1, color = 'blue', label ="n = 30, p = 0.5")
plt.plot(courbe2, color = 'green', label ="n = 30, p = 0.7")
plt.plot(courbe3, color = 'red', label ="n = 50, p = 0.4")

plt.legend(loc = 'upper right')

plt.xlabel('x')
plt.ylabel('P(X = x)')

plt.savefig('binomiale.png')
plt.show()

#-----------------------------------------------------------------------------
# 2.2. Loi Normale univariée
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Données
#-----------------------------------------------------------------------------

#espérance
mu1 = 0
mu2 = 2
mu3 = 2

#écart type
sigma1 = 1
sigma2 = 1.5
sigma3 = 0.6

#bornes en abscisse
borneinf = -10
bornesup = 10

#-----------------------------------------------------------------------------
# Affichage des courbes
#-----------------------------------------------------------------------------

x = np.linspace(borneinf, bornesup, 1000)

y1 = spicy.norm.pdf(x, mu1, sigma1)
y2 = spicy.norm.pdf(x, mu2, sigma2)
y3 = spicy.norm.pdf(x, mu3, sigma3) 

plt.plot(x, y1, color = 'blue', label = "mu = 0, sigma = 1")
plt.plot(x, y2, color = 'green', label = "mu = 2, sigma = 1.5")
plt.plot(x, y3, color = 'red', label = "mu = 2, sigma = 0.6")

plt.legend(loc = 'upper left')

plt.xlabel('x')
plt.ylabel('P(X = x)')

plt.savefig('normale.png')
plt.show()


#-----------------------------------------------------------------------------
# 2.3. Simulation de données à partir d’une loi
#-----------------------------------------------------------------------------
# 2.3.1. Cas de la loi Normale
#-----------------------------------------------------------------------------

#loi normale centrée réduite : N(0,1)

for n in [100, 1000, 10000, 100000]:
    
    b = np.random.normal(mu1, sigma1, n)
    
    #histogramme des données générées
    plt.hist(b, bins = 30, density = True, edgecolor = 'black', alpha = 0.2,  label = f"échantillon taille {n}") 
    #alpa : transparence de l'histogramme
    
    #courbe de densité théorique de la loi normale centrée réduite
    x1 = np.linspace(-6, 6, 1000)
    plt.plot(x1, y1, color = 'darkblue', label = "mu = 0, sigma = 1")

    plt.legend(loc = 'upper right')

    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    plt.savefig('normale2.png')
    plt.show()


#-----------------------------------------------------------------------------
# 2.4. Estimation de densité
#-----------------------------------------------------------------------------
# 2.4.1. Cas de la loi Normale
#-----------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# Calcul des estimateurs
#-----------------------------------------------------------------------------

#renvoie la moyenne empirique
def moyenne(valeurs):
    return sum(valeurs) / len(valeurs)

#renvoie l'éacrt-type empirique
def ecart_type(valeurs):
    somme = 0
    x_bar = moyenne(valeurs)

    for i in range(len(valeurs)):
        somme = somme + (valeurs[i] - x_bar)**2

    return m.sqrt(somme / (len(valeurs) - 1))


#-----------------------------------------------------------------------------
# Comparaison des résultats
#-----------------------------------------------------------------------------

print(f"Estimation de densité pour la loi normale \n")

#abscisses
x2 = np.linspace(-5, 5, 1000)
    
for n in [20, 80, 150]:
    b1 = np.random.normal(mu1, sigma1, n)
    
    #estimation de la moyenne et de l'écart-type
    mu_estime = moyenne(b1)
    sigma_estime = ecart_type(b1)
    
    print(f"Échantillon de taille n = {n}")
    print(f"Moyenne empirique : {mu_estime}")
    print(f"Ecart-type empirique : {sigma_estime}")
    
    #courbe de densité empirique de la loi univariée centrée réduite
    y4 = spicy.norm.pdf(x2, mu_estime, sigma_estime)
    plt.plot(x2, y4, color = 'red', label = f"N(0,1) empirique n = {n}")

    #courbe de densité théorique de la loi univariée centrée réduite
    plt.plot(x2, y1, color = 'darkblue', label = "N(0,1) théorique")
    
    #tracé des courbes
    plt.legend(loc = 'upper right')

    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    plt.savefig('normale3.png')
    plt.show()
    

#-----------------------------------------------------------------------------
# 2.4.2. Cas de la loi exponentielle
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Fonction de densité
#-----------------------------------------------------------------------------

#paramètre de la loi exponentielle
lambda1 = 1

#abscisses
x3 = np.linspace(0, 8, 1000)

print(f"\nEstimation de densité pour la loi exponentielle \n")

for n in [20, 80, 150]:
    #échantillons générés
    b2 = np.random.exponential(scale = 1/lambda1, size = n)

    #histogramme des échantillons générés
    plt.hist(b2, bins = 30, density = True, edgecolor = 'black', alpha = 0.2,  label = f"échantillon n = {n}") 
    
    #estimation du paramètre lambda
    lambda_estime = 1 / moyenne (b2)
    
    print(f"Échantillon de taille n = {n}")
    print(f"Estimation du paramètre lambda : {lambda_estime}")
    
    #courbe de densité empirique de la loi exponentielle
    y5 = spicy.expon.pdf(x3, scale = 1 / lambda_estime)
    plt.plot(x3, y5, color = 'red', label = "E(1) empirique")
    
    #courbe de densité théorique de la loi exponentielle
    y6 = spicy.expon.pdf(x3, scale = 1/lambda1)
    plt.plot(x3, y6, color = 'darkblue', label = "E(1) théorique")
        
    plt.legend(loc = 'upper right')

    plt.xlabel('x')
    plt.ylabel('P(X = x)')

    plt.savefig('densite_exponentielle.png')
    plt.show()

#-----------------------------------------------------------------------------
# Fonction de répartition
#-----------------------------------------------------------------------------

print(f"\nEstimation de répartition pour la loi exponentielle \n")

for n in [20, 80, 150]:
    #échantillons générés
    b2 = np.random.exponential(scale = 1/lambda1, size = n)

    #histogramme des échantillons générés
    plt.hist(b2, bins = 30, density = True, edgecolor = 'black', alpha = 0.2,  label = f"échantillon n = {n}") 
    
    #estimation du paramètre lambda
    lambda_estime2 = 1 / moyenne (b2)
    
    print(f"Échantillon de taille n = {n}")
    print(f"Estimation du paramètre lambda : {lambda_estime2}")
    
    #courbe de répartition empirique de la loi exponentielle
    y6 = spicy.expon.cdf(x3, scale = 1/lambda_estime2)
    plt.plot(x3, y6, color = 'red', label = "F(x) empirique")
    
    #courbe de répartition théorique de la loi exponentielle
    y6 = spicy.expon.cdf(x3, scale = 1/lambda1)
    plt.plot(x3, y6, color = 'darkblue', label = "F(x) théorique")
        
    plt.legend(loc = 'upper right')

    plt.xlabel('x')
    plt.ylabel('F(x)')

    plt.savefig('repartition_exponentielle.png')
    plt.show()
    
    
#-----------------------------------------------------------------------------
# Variation du paramètre lambda
#-----------------------------------------------------------------------------

#paramètres de la loi exponentielle
lambda2 = 0.5
lambda3 = 1.5
lambda4 = 2

#tracé des courbes
y7 = spicy.expon.pdf(x3, scale = 1/lambda2)
plt.plot(x3, y7, color = 'darkblue', label = f"E({lambda2})")

y8 = spicy.expon.pdf(x3, scale = 1/lambda1)
plt.plot(x3, y8, color = 'green', label = f"E({lambda1})")

y9 = spicy.expon.pdf(x3, scale = 1/lambda3)
plt.plot(x3, y9, color = 'orange', label = f"E({lambda3})")

y10 = spicy.expon.pdf(x3, scale = 1/lambda4)
plt.plot(x3, y10, color = 'red', label = f"E({lambda4})")
    
plt.legend(loc = 'upper right')

plt.xlabel('x')
plt.ylabel('P(X = x)')

plt.savefig('exp_lambdas.png')
plt.show()
