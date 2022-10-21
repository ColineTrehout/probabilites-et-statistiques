#Alice Gydé et Coline Trehout

#-----------------------------------------------------------------------------
# TP 1 : Régression linéaire
#-----------------------------------------------------------------------------

import numpy as np
from numpy.lib.polynomial import poly
import matplotlib.pyplot as plt
from numpy.linalg import inv 
from numpy import linalg as la
import statsmodels.api as sm
import pylab
import pylab

#-----------------------------------------------------------------------------
# 1.1. Régression linéaire simple
#-----------------------------------------------------------------------------
# 1.1.1. Méthode des moindres carrés
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# calculB retourne les coefficients de régression linéaires calculés par 
# la méthode des moindres carrés
#-----------------------------------------------------------------------------

def calculB (x,y) :
    #moyennes
    xbarre = sum(x)/len(x)
    ybarre = sum(y)/len(y)
    covxy = 0
    covxx = 0
    for i in range (len(x)) :
        covxy = covxy+(x[i]-xbarre)*(y[i]-ybarre)
        covxx = covxx+(x[i]-xbarre)**2
    b1 = covxy/covxx
    b0 = ybarre-b1*xbarre
    return b0,b1


#-----------------------------------------------------------------------------
# Données
#-----------------------------------------------------------------------------

#y : pourcentage de rendement 
#x : température en degré celsius d’un procédé chimique
x = np.array([45,50,55,60,65,70,75,80,85,90])
y = np.array([43,45,48,51,55,57,59,63,66,68])


#-----------------------------------------------------------------------------
# Calcul et affichage des résultats
#-----------------------------------------------------------------------------

B = calculB(x,y)

print("Méthode des moindres carrés :\n")
print (f"résultat obtenu par la méthode des moindres carrés : B0 = {round(B[0],4)}, B1 = {round(B[1],4)}.")
poly = np.polyfit(x,y,1) #1 est le degré
print (f"résultat obtenu par la fonction polyfit : B0 = {round(poly[1],4)}, B1 = {round(poly[0],4)}.")

#comparaison des résultats
ecartB0 = abs(B[0]-poly[1])
ecartB1 = abs(B[1]-poly[0])

print(f"L'écart est de {format(ecartB0,'.2E')} pour B0 et {format(ecartB1,'.2E')} pour B1.\n")


#-----------------------------------------------------------------------------
# Affichage de la courbe
#-----------------------------------------------------------------------------

#nuage de points
plt.scatter(x,y, s = 100, c = 'darkblue', marker = '+' )
plt.title('Régression linéaire avec méthode vectorielle')
plt.xlabel('température en °C')
plt.ylabel('pourcentage de rendement')

#droite de la méthode des moindres carrés (avec lw épaisseur du trait)
plt.plot([30,100], [B[0]+B[1]*30, B[0]+B[1]*100], c ='red', lw = '1.5')

#droite de polyfit
#plt.plot([30,100], [poly[1]+poly[0]*30, poly[1]+poly[0]*100], c = 'blue', lw = '1.5')

plt.savefig('moindresCarres.png')
plt.show()

#-----------------------------------------------------------------------------
# 1.1.2. Méthode matricielle
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Calcul par la méthode matricielle et affichage des résultats
#-----------------------------------------------------------------------------

A = np.ones((len(x),2))

for i in range(len(x)):
    A[i][1] = x[i]
B = np.dot(inv(np.dot(A.T,A)),np.dot(A.T,y))

print("Méthode matricielle :\n")
print (f"résultat obtenu par la méthode matricielle : B0 = {round(B[0],4)}, B1 = {round(B[1],4)}.")
poly = np.polyfit(x,y,1)
print (f"résultat obtenu par la fonction polyfit : B0 = {round(poly[1],4)}, B1 = {round(poly[0],4)}.")

#comparaison des résultats
ecartB0 = abs(B[0]-poly[1])
ecartB1 = abs(B[1]-poly[0])

print(f"L'écart est de {format(ecartB0,'.2E')} pour B0 et {format(ecartB1,'.2E')} pour B1. \n")

#-----------------------------------------------------------------------------
# Affichage de la courbe obtenue avec la méthode matricielle
#-----------------------------------------------------------------------------

#nuage de points
plt.scatter(x,y, s = 100, c = 'darkblue', marker = '+' )
plt.title('Régression linéaire avec méthode matricielle')
plt.xlabel('température en °C')
plt.ylabel('pourcentage de rendement')

#droite avec lw épaisseur du trait
plt.plot([30,100], [B[0]+B[1]*30, B[0]+B[1]*100], c = 'red', lw = '1.5')

plt.savefig('matricielle.png')
plt.show()

#-----------------------------------------------------------------------------
# Affichage de la courbe du QQ-plot
#-----------------------------------------------------------------------------

pp_x = sm.ProbPlot(x, fit = True)
pp_y = sm.ProbPlot(y, fit = True)

pp_x.qqplot(other = pp_y, line = '45', c = 'black', marker = '+')

plt.title('Droite du QQ-Plot')
plt.savefig('qqplot.png')
pylab.show()

#-----------------------------------------------------------------------------
# 1.2. Régression linéaire par descente de gradient
#-----------------------------------------------------------------------------

print("Méthode de descente de gradient : \n")

#-----------------------------------------------------------------------------
# Calcul des dérivées partielles pour la méthode de descente de gradient
#-----------------------------------------------------------------------------

def h(b0,b1,x):
    return (b0+b1*x)
    
#dérivée partielle par rapport à B0
def derivee0(bt,x,y):
    d = 0
    for i in range(len(x)):
        d = d+h(bt[0],bt[1],x[i])-y[i]
    return(1/len(x)*d)

#dérivée partielle par rapport à B1    
def derivee1(bt,x,y):
    d = 0
    for i in range(len(x)):
        d = d+(h(bt[0],bt[1],x[i])-y[i])*x[i]
    return(1/len(x)*d) 

#-----------------------------------------------------------------------------
# Données
#-----------------------------------------------------------------------------

epsilon = 0.001 #seuil de tolérance
lamb = 0.0001 #pas compris entre 0 et 1
bt_1 = [17,17] #vecteur beta à l'itération précédente
bt = [0,0] #vecteur beta à l'itération actuelle
i = 1 #nombre d'itérations

#première coordonnée : B0
#deuxième coordonnée : B1

#-----------------------------------------------------------------------------
# Calcul par la méthode de descente de gradient
#-----------------------------------------------------------------------------

bt[0] = bt_1[0]-lamb*derivee0(bt_1,x,y)
bt[1] = bt_1[1]-lamb*derivee1(bt_1,x,y)
print(f"itération n°{i} : {bt}")

while (la.norm([bt[0]-bt_1[0],bt[1]-bt_1[1]]) > epsilon):
    save = bt.copy()
    bt_1 = save
    bt[0] = bt_1[0]-lamb*derivee0(bt_1,x,y)
    bt[1] = bt_1[1]-lamb*derivee1(bt_1,x,y)
    i = i+1
    print(f"itération n°{i} : {bt}")
    
print(f"\nrésultat final par l'algorithme de gradient au bout de {i} itérations : B0 = {round(bt[0],4)} B1 = {round(bt[1],4)}")

poly = np.polyfit(x,y,1)
print (f"résultat obtenu par la fonction polyfit : B0 = {round(poly[1],4)}, B1 = {round(poly[0],4)}.")

#-----------------------------------------------------------------------------
# Comparaison des résultats
#-----------------------------------------------------------------------------

#résultat avec méthode des moindres carrés
B = calculB(x,y)

#comparaison des résultats
ecartB0 = abs(bt[0]-poly[1])
ecartB1 = abs(bt[1]-poly[0])

print(f"L'écart est de {format(ecartB0,'.2E')} pour B0 et {format(ecartB1,'.2E')} pour B1.")

#-----------------------------------------------------------------------------
# Affichage des courbes
#-----------------------------------------------------------------------------

#nuage de points
plt.scatter(x,y, s = 100, c = 'darkblue', marker = '+' )
plt.title('Régression linéaire avec méthode de descente de gradient')
plt.xlabel('température en °C')
plt.ylabel('pourcentage de rendement')

#droite avec lw épaisseur du trait
#droite obtenue par la méthode des gradients
plt.plot([30,100], [bt[0]+bt[1]*30, bt[0]+bt[1]*100], c = 'darkgreen', lw = '1.5')

#droite obtenue par la méthode des moindres carrés
plt.plot([30,100], [B[0]+B[1]*30, B[0]+B[1]*100], c = 'red', lw = '1.5')

plt.savefig('gradient.png')
plt.show()