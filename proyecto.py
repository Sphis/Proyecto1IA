# importar modulos de interes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # Primer algoritmo a usar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans # Este es el segundo algoritmo a usar

################################### NOTA: ###################################
####                                                                     ####
# Para observar los resultados de los Anexos tenga en cuenta que:           #
# opcionA = 1 para ver los datos del AnexoA                                 #
# opcionA = 0 Para ver los datos del AnexoB                                 #
####                                                                     ####

opcionA = 1 # Para determinar que datos quiere analizar el usuario

# Se guarda los datos del Anexo A en un dataframe
if (opcionA == 1):
    AnexoA = pd.read_csv("DatosAnexoA.csv")
    # print(AnexoA)
else:
    AnexoA = pd.read_csv("DatosAnexoB.csv")
    # print(AnexoA)

# Ahora se guardan los datos de interés, es decir, las columnas de Abonados, DPI y FPI
columnas = ['Abonados', 'DPI', 'FPI']
datos_normalizados = AnexoA[columnas]

# Se normalizan los datos
scaler1 = StandardScaler()
normalizadoA = scaler1.fit_transform(datos_normalizados)
AnexoA[columnas] = normalizadoA

# Se toman los únicamente los datos normalizados para trabajar con ellos
datosA = AnexoA.iloc[:,2:6].values

############################## Aqui se implementa el caso de K-means ##############################

# Ahora se implementa el método codo para determinar cuántos cluseters utilizar
# Método codo para los datos introducidos
listaA = []
for i in range(1,11):
    k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
    k_means.fit(datosA)
    listaA.append(k_means.inertia_)

# Graficar para determinar numero de clusters visualmente
plt.figure(1)
plt.plot(np.arange(1,11),listaA)
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.xticks(np.arange(0, 12, 1.0))
if (opcionA == 1):
    plt.title("Método codo para datos del anexo A (k-means)")
    plt.show()
else:
    plt.title("Método codo para datos del anexo B (k-means)")
    plt.show()

# Para el Anexo A se determina que la cantidad de clusters ideal es de 2 con un valor
# silueta de 0.4698429269637665
# Para el Anexo B se determina que la cantidad de clusters ideal es de 2 con un valor
# silueta de 0.4874193356068225

# Ahora se aplica el algoritmo de Kmeans
k_means_optimum = KMeans(n_clusters = 2, init = 'k-means++',  random_state=42)
y = k_means_optimum.fit_predict(datosA)
print(k_means.cluster_centers_)

# Agregar otra columna para indicar a cuál de los 3 clusters pertenecen los datos (Anexo A)
AnexoA['cluster'] = y

# Indicar cual cluster pertence el dato
data1A = AnexoA[AnexoA.cluster == 0]
data2A = AnexoA[AnexoA.cluster == 1]

plt.figure(2)
kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)

# Graficar datos del anexo A
# Poner los datos en 3D y darle color a los clusters
kplot.scatter3D(data1A.Abonados, data1A.DPI, data1A.FPI, c='red', label = 'Cluster 0')
kplot.scatter3D(data2A.Abonados, data2A.DPI, data2A.FPI, c ='green', label = 'Cluster 1')

# Nombrar ejes
kplot.set_xlabel('Abonados')
kplot.set_ylabel('DPI')
kplot.set_zlabel('FPI')

plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()
if (opcionA == 1):
    plt.title("Método de Kmeans (Anexo A)")
    plt.show()
else:
    plt.title("Método de Kmeans (Anexo B)")
    plt.show()

# Finalmente se imprime el valor de puntuación de silueta para ver que tan bien ajustados están los 
# clusters que se escogieron
silueta = silhouette_score(datosA,y)
#print("\n Valor de puntuación de silueta con k-means \n" + str(silueta) + "\n")


############################## Aqui se implementa el caso de mini batch K-means ##############################

# Ahora se implementa el método codo para determinar cuántos cluseters utilizar
# Método codo para los datos introducidos
listaA = []
for i in range(1,11):
    k_means = MiniBatchKMeans(n_clusters=i,init='k-means++', random_state=42)
    k_means.fit(datosA)
    listaA.append(k_means.inertia_)
# Graficar para determinar numero de clusters visualmente
plt.figure(3)
plt.plot(np.arange(1,11),listaA)
plt.xlabel('Clusters')
plt.ylabel('SSE')
plt.xticks(np.arange(0, 12, 1.0))
if (opcionA == 1):
    plt.title("Método codo para datos del anexo A (mini batch k-means)")
    plt.show()
else:
    plt.title("Método codo para datos del anexo B (mini batch k-means)")
    plt.show()

# Para el Anexo A se determina que la cantidad de clusters ideal es de 3 con un valor
# silueta de 0.4140586834631853
# Para el Anexo B se determina que la cantidad de clusters ideal es de 2 con un valor
# silueta de 0.4874193356068225

# Ahora se aplica el algoritmo de Kmeans
if (opcionA == 1):
    k_means_optimum = MiniBatchKMeans(n_clusters = 3, init = 'k-means++',  random_state=42)
    y = k_means_optimum.fit_predict(datosA)
else:
    k_means_optimum = MiniBatchKMeans(n_clusters = 2, init = 'k-means++',  random_state=42)
    y = k_means_optimum.fit_predict(datosA)

# Agregar otra columna para indicar a cuál de los 3 clusters pertenecen los datos (Anexo A)
AnexoA['cluster'] = y

# Indicar cual cluster pertence el dato
data1A = AnexoA[AnexoA.cluster == 0]
data2A = AnexoA[AnexoA.cluster == 1]
data3A = AnexoA[AnexoA.cluster == 2]

plt.figure(4)
kplot = plt.axes(projection='3d')
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)
zline = np.linspace(0, 15, 1000)

# Graficar datos del anexo A
# Poner los datos en 3D y darle color a los clusters
# Esto es para las rotulaciones del Anexos A ya que es mejor usar 3 clusters
if (opcionA == 1):
    kplot.scatter3D(data1A.Abonados, data1A.DPI, data1A.FPI, c='red', label = 'Cluster 0')
    kplot.scatter3D(data2A.Abonados, data2A.DPI, data2A.FPI, c ='green', label = 'Cluster 1')
    kplot.scatter3D(data3A.Abonados, data3A.DPI, data3A.FPI, c ='blue', label = 'Cluster 2')
else: # Esto es para las rotulaciones del Anexos B ya que es mejor usar 2 clusters
    kplot.scatter3D(data1A.Abonados, data1A.DPI, data1A.FPI, c='red', label = 'Cluster 0')
    kplot.scatter3D(data2A.Abonados, data2A.DPI, data2A.FPI, c ='green', label = 'Cluster 1')

# Nombrar ejes
kplot.set_xlabel('Abonados')
kplot.set_ylabel('DPI')
kplot.set_zlabel('FPI')

plt.scatter(k_means_optimum.cluster_centers_[:,0], k_means_optimum.cluster_centers_[:,1], color = 'indigo', s = 200)
plt.legend()
if (opcionA == 1):
    plt.title("Método de mini batch K-means (Anexo A)")
    plt.show()
else:
    plt.title("Método de mini batch K-means (Anexo B)")
    plt.show()

# Finalmente se imprime el valor Puntuación de Silueta para ver que tan bien ajustados están los 
# clusters que se escogieron
silueta = silhouette_score(datosA,y)
#print("\n Valor de puntuación de silueta con mini batch k-means: \n" + str(silueta) + "\n")