import pandas as pd
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import const
from sklearn.cluster import KMeans
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import  GridSearchCV, PredefinedSplit
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
# from xgboost.sklearn import XGBClassifier
# import seaborn as sns
# import re

datasets = ['Intensité', 'Palette', 'K-means', 'Comparaison Palette / K-means']

# I. Fonctions permettant l'extraction des données d'une image


# Fonctions annexes
def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def iqr(x):
    return q3(x) - q1(x)


# Fonctions extraction répartition couleurs
def extract_centileRGB(file_path, without_bkg):
    """ Prend en argument :
            - le chemin de l'image
            - si l'image est détourée (True) ou non (False)

        Retourne une liste contenant la répartition en centile des couleurs
        BGR + moyenne et écart type
    """

    # lecture de l'image
    img = cv2.imread(file_path)

    # Contrôle taille image :
    if img.shape[0] != 224 or img.shape[1] != 224:
        img = cv2.resize(img, dsize=(224, 224))

    # Redimensionnement de l'array
    img = img.reshape([224*224, 3])

    # Suppression pixels [0, 0, 0] correspondant la partie détourée
    if without_bkg:
        img = img[(img[:, 0] > 0) | (img[:, 1] > 0) | (img[:, 1] > 0)]

    # Calcul répartition en centile des valeurs BGR par image
    rep_color = np.percentile(img, range(0, 101), axis=0)

    # Passage des centiles en colonne pour chaque couleur
    for col in range(0, rep_color.shape[1]):
        if col == 0:
            rep_color_img = list(rep_color[:, col].T)
        else:
            rep_color_img.extend(list(rep_color[:, col].T))

    # Ajout des informations moyenne et écart type
    rep_color_img.extend(list(img.mean(axis=0).round(1)))
    rep_color_img.extend(list(img.std(axis=0).round(1)))

    return rep_color_img


# Fonctions extraction palette couleurs
def extract_palette(file_path, nb_colors, without_bkg):
    """ Prend en argument :
            - le chemin de l'image
            - le nombre de couleurs à extraire
            - si l'image est détourée (True) ou non (False)

        Retourne une liste contenant les indicateurs de position (moyenne,
        médiane, Q1, Q3) et de dispersion (écart type et écart interquartile )
        par couleur"""

    img = cv2.imread(file_path)

    # Largeur et hauteur de l'image pour traitements
    img_width = 120
    img_height = 120

    # seuil érosion
    seuil_ero = 2

    # Redimensionnement de l'image
    img_color = cv2.resize(img, dsize=(img_width, img_height))

    # Passage en noir et blanc pour stats et érosion si image détourée
    # (without_bkg = True)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if without_bkg:

        # Érosion
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(img_gray, kernel)

        erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
        erosion = erosion.reshape(img_width*img_height, 3)

        # Application érosion à l'image en noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1)
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

        # Test découpage couleur (de la + foncée à la + claire) et ajustement
        # seuil si besoin
        decoup = list(np.percentile(img_sort_gray[:, 0],
                                    np.arange(0, 101, 100/nb_colors).round(1),
                                    axis=0))

        # Ajustement seuil érosion (5 maximum)
        while len(np.unique(decoup)) <= nb_colors and seuil_ero < 5:
            seuil_ero += 1
            img_sort_gray = img_gray.reshape(img_width*img_height, 1)
            img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                          (erosion[:, 1] > seuil_ero) |
                                          (erosion[:, 2] > seuil_ero)]
            decoup = list(np.percentile(img_sort_gray[:, 0],
                                        np.arange(0, 101, 100/nb_colors)
                                        .round(1),
                                        axis=0))

        # Exlusion des pixels totalement blancs (problème image)
        erosion[(erosion[:, 0] > (255 - seuil_ero)) &
                (erosion[:, 1] > (255 - seuil_ero)) &
                (erosion[:, 2] > (255 - seuil_ero))] = 0

        # Application érosion sur image couleur
        img_sort_color = img_color.reshape(img_width*img_height, 3).copy()

        img_sort_color = img_sort_color[(erosion[:, 0] > seuil_ero) |
                                        (erosion[:, 1] > seuil_ero) |
                                        (erosion[:, 2] > seuil_ero)]

        # Application érosion sur image noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1).copy()
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

    else:
        seuil_ero = 5

        # Exlusion des pixels totalement blancs (problème image)
        erosion = img_color.reshape(img_width*img_height, 3).copy()
        erosion[(erosion[:, 0] > (255 - seuil_ero)) &
                (erosion[:, 1] > (255 - seuil_ero)) &
                (erosion[:, 2] > (255 - seuil_ero))] = 0

        # Application érosion sur image couleur
        img_sort_color = img_color.reshape(img_width*img_height, 3)

        img_sort_color = img_sort_color[(erosion[:, 0] > seuil_ero) |
                                        (erosion[:, 1] > seuil_ero) |
                                        (erosion[:, 2] > seuil_ero)]

        # Application erosion à image en noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1)
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

    # Découpage sur R + G + B
    decoup_source = np.sum(img_sort_color, axis=1)
    decoup_source.shape = (len(decoup_source), 1)
    decoup_source = np.concatenate([img_sort_color, decoup_source], axis=1)
    decoup = list(np.percentile(decoup_source[:, 3],
                                np.arange(0, 101, 100/nb_colors).round(1),
                                axis=0))

    # Modification découpage si nombre de valeurs distinctes toujours pas ok
    cpt = 0
    if len(np.unique(decoup)) <= nb_colors:
        decoup_temp = decoup.copy()
        decoup.clear()
        for val in decoup_temp:
            # print(decoup)
            while val <= (cpt + 1):
                val = val + 2
            # print('Ajout')
            cpt = val
            decoup.append(int(val))

    # Passage en DataFrame
    df_img_gray = pd.DataFrame(img_sort_gray, columns=['Gray'])
    df_img_color = pd.DataFrame(decoup_source, columns=['B', 'G', 'R', 'TOT'])

    df_img = pd.concat([df_img_gray, df_img_color], axis=1)

    # Découpage couleurs
    nom_inter = []
    for inter in range(1, nb_colors+1, 1):
        nom_inter.append(f'C{str(inter)}')

    df_img['decoup'] = pd.cut(x=df_img['TOT'], bins=decoup,
                              labels=nom_inter, include_lowest=True)

    # Calcul des indicateurs
    calculs = {'R': ['mean', 'std', 'median', q1, q3, iqr],
               'G': ['mean', 'std', 'median', q1, q3, iqr],
               'B': ['mean', 'std', 'median', q1, q3, iqr],
               'Gray': ['mean', 'std', 'median', q1, q3, iqr]}

    df_img_sort = df_img.groupby('decoup', as_index=False).agg(calculs)\
        .round(1)

    # Renommage des colonnes
    df_img_sort = df_img_sort.set_axis(df_img_sort.columns.map('_'.join),
                                       axis=1)

    # Gestion des valeurs manquantes
    df_img_sort['Test_NaN'] = df_img_sort.isna().any(axis=1)

    for i in range(0, nb_colors, 1):
        # print(f"test C{i+1}")
        if df_img_sort[df_img_sort['decoup_'] == f'C{i+1}']['Test_NaN'].values:
            # Récupération données classe précédente
            df_img_sort.iloc[i, 1:25] = df_img_sort.iloc[i-1, 1:25]

    df_img_sort = df_img_sort.drop('Test_NaN', axis=1)

    # Passage des informations en liste avec toutes les infos sur 1 ligne
    rep_color_img = []
    for i in range(0, df_img_sort.shape[0]):
        if i == 0:
            rep_color_img = list(df_img_sort.iloc[i, -24:])
        else:
            rep_color_img.extend(list(df_img_sort.iloc[i, -24:]))

    return rep_color_img


# Fonctions extraction couleurs K-means
def extract_colors_img_kmeans(file_path, nb_colors, without_bkg):
    """ Prend en argument :
        - le chemin de l'image
        - le nombre de couleurs à extraire
        - si l'image est détourée (True) ou non (False)

    Retourne une liste contenant les indicateurs de position (moyenne, médiane,
    Q1, Q3) et de dispersion (écart type et écart interquartile ) par couleur
    définies selon algorithme des K-means"""

    img = cv2.imread(file_path)

    # Taille image resizée
    img_width = 120
    img_height = 120

    # Nb de couleurs pour nettoyage image / érosion
    nb_colors_clean = 20

    # Resize image et affichage
    img_color = cv2.resize(img, dsize=(img_width, img_height))

    # Passage en noir et blanc et affichage
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    if without_bkg:

        # seuil érosion pour image détourée
        seuil_ero = 2

        # Erosion
        kernel = np.ones((4, 4), np.uint8)
        erosion = cv2.erode(img_gray, kernel)

        erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
        erosion = erosion.reshape(img_width*img_height, 3)

        # Application erosion à image en noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1)
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

        # Test découpage et ajustement seuil si besoin
        decoup = list(np.percentile(img_sort_gray[:, 0],
                                    np.arange(0, 101, 100/nb_colors_clean),
                                    axis=0))

        # Ajustement seuil érosion (5 maximum)
        # print(seuil_ero)
        while len(np.unique(decoup)) <= nb_colors_clean and seuil_ero < 5:
            seuil_ero += 1
            img_sort_gray = img_gray.reshape(img_width*img_height, 1)
            img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                          (erosion[:, 1] > seuil_ero) |
                                          (erosion[:, 2] > seuil_ero)]
            decoup = list(np.percentile(img_sort_gray[:, 0],
                                        np.arange(0, 101, 100/nb_colors_clean),
                                        axis=0))

        # Exlusion des pixels totalement blancs (problème image)
        erosion[(erosion[:, 0] > (255 - seuil_ero)) &
                (erosion[:, 1] > (255 - seuil_ero)) &
                (erosion[:, 2] > (255 - seuil_ero))] = 0

        # Application érosion sur image couleur
        img_sort_color = img_color.reshape(img_width*img_height, 3).copy()

        img_sort_color = img_sort_color[(erosion[:, 0] > seuil_ero) |
                                        (erosion[:, 1] > seuil_ero) |
                                        (erosion[:, 2] > seuil_ero)]

        # Application érosion sur image noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1).copy()
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

    else:
        seuil_ero = 5

        # Exlusion des pixels totalement blancs (problème image)
        erosion = img_color.reshape(img_width*img_height, 3).copy()
        erosion[(erosion[:, 0] > (255 - seuil_ero)) &
                (erosion[:, 1] > (255 - seuil_ero)) &
                (erosion[:, 2] > (255 - seuil_ero))] = 0

        # Application érosion sur image couleur
        img_sort_color = img_color.reshape(img_width*img_height, 3)

        img_sort_color = img_sort_color[(erosion[:, 0] > seuil_ero) |
                                        (erosion[:, 1] > seuil_ero) |
                                        (erosion[:, 2] > seuil_ero)]

        # Application erosion à image en noir et blanc
        img_sort_gray = img_gray.reshape(img_width*img_height, 1)
        img_sort_gray = img_sort_gray[(erosion[:, 0] > seuil_ero) |
                                      (erosion[:, 1] > seuil_ero) |
                                      (erosion[:, 2] > seuil_ero)]

    # Algorithme de K-means
    kmeans = KMeans(n_clusters=nb_colors)
    kmeans.fit(img_sort_color)

    # Centroids and labels
    # centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    labels.shape = (len(labels), 1)
    decoup_source = np.concatenate([img_sort_color, labels], axis=1)

    # Passage en DataFrame
    df_img_gray = pd.DataFrame(img_sort_gray, columns=['Gray'])
    df_img_color = pd.DataFrame(decoup_source,
                                columns=['B', 'G', 'R', 'Decoup'])

    df_img = pd.concat([df_img_gray, df_img_color], axis=1)

    calculs = {'R': ['mean', 'std', 'median', q1, q3, iqr],
               'G': ['mean', 'std', 'median', q1, q3, iqr],
               'B': ['mean', 'std', 'median', q1, q3, iqr],
               'Gray': ['mean', 'std', 'median', q1, q3, iqr],
               'Decoup': 'count'}

    df_img_sort = df_img.groupby('Decoup', as_index=False) \
        .agg(calculs).round(1)

    df_img_sort = df_img_sort.set_axis(df_img_sort.columns.map('_'.join),
                                       axis=1)

    df_img_sort = df_img_sort.sort_values(by='Decoup_count', ascending=False)
    df_img_sort['Color'] = [f'C{i}' for i in range(1, nb_colors + 1)]

    # Passage des poids en %
    df_img_sort['poids_%'] = round((df_img_sort['Decoup_count']
                                    / df_img_sort['Decoup_count']
                                    .sum())*100, 1)
    df_img_sort = df_img_sort.drop('Decoup_count', axis=1)
    df_img_sort = df_img_sort.set_index('Color')

    # Passage des informations en colonne pour chaque couleur
    rep_color_img = []
    for i in range(0, df_img_sort.shape[0]):
        if i == 0:
            rep_color_img = list(df_img_sort.iloc[i, -25:])
        else:
            rep_color_img.extend(list(df_img_sort.iloc[i, -25:]))

    return rep_color_img


# II. Fonctions permettant la récupération des données suite extraction
# Répartition couleurs
def recup_centileRGB(res):
    centile_r, centile_g, centile_b = [], [], []
    for i in range(0, 101):
        centile_b.append(res[i])
        centile_g.append(res[i + 101])
        centile_r.append(res[i + 202])

    return centile_b, centile_g, centile_r


# Palette
def recup_palette(res, nb_colors):
    recup_res = []
    for num, i in enumerate(range(0, len(res), 24)):
        # print(i, i+6, i+12)
        recup_res.append([f'C{num+1}', res[i], res[i+6], res[i+12]])

    df_img_sort = pd.DataFrame(recup_res, columns=['decoup', 'R_mean',
                                                   'G_mean', 'B_mean'])

    # Pour calcul palette
    dim1 = 40
    dim2 = 200
    coeff_pal = int(dim2/nb_colors)

    demo_coul_moy = np.zeros(shape=(dim1, dim2, 3), dtype=int)
    demo_coul_moy[:, :, :] = 255

    deb = 0
    fin = coeff_pal

    for i in range(0, nb_colors, 1):
        demo_coul_moy[:, deb:fin, :] = np.array(df_img_sort
                                                [df_img_sort['decoup'] ==
                                                 f'C{i+1}']
                                                [['R_mean',
                                                  'G_mean',
                                                  'B_mean']].round(0))
        deb = fin + 1
        fin += coeff_pal

    return demo_coul_moy


# Kmeans
def recup_kmeans(res):
    recup_res = []
    for num, i in enumerate(range(0, len(res), 25)):
        # print(num, i, i+6, i+12)
        # print(num, res[i], res[i+6], res[i+12])
        recup_res.append([f'C{num+1}', res[i], res[i+6], res[i+12], res[i+24]])

    df_img_sort = pd.DataFrame(recup_res, columns=['decoup_', 'R_mean',
                                                   'G_mean', 'B_mean',
                                                   'Poids_%'])

    return df_img_sort


# III. Fonctions permettant la définition de la figure
# Répartition couleurs
def fig_centileRGB(img_path, centile_b, centile_g, centile_r):
    img = cv2.imread(img_path)

    fig = plt.figure(figsize=(11, 4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image originale')
    ax1.axis('off')

    # Répartition en centile des valeurs RGB
    ax2 = fig.add_subplot(122)
    barWidth = 0.3
    x1 = range(0, 101)
    x2 = [i + barWidth for i in x1]
    x3 = [i + barWidth for i in x2]
    ax2.bar(x1, centile_r, alpha=0.7, color='red')
    ax2.bar(x2, centile_g, alpha=0.7, color='green')
    ax2.bar(x3, centile_b, alpha=0.7, color='blue')
    ax2.set_xlim(0, 101)
    ax2.set_ylim(0, 255)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylabel('Valeurs RVB (0-255)')
    ax2.set_xlabel('Centile')
    ax2.set_title("Répartion par centile des valeurs RVB de l'image")
    ax2.text(5, 210, f'Médiane : {centile_r[50]}', color='red')
    ax2.text(5, 195, f'Médiane : {centile_g[50]}', color='green')
    ax2.text(5, 180, f'Médiane : {centile_b[50]}', color='blue')

    return fig


# Palette
def fig_palette(img_path, demo_coul_moy):
    fig = plt.figure(figsize=(10, 5))

    # Image originale
    ax1 = fig.add_subplot(121)
    img = cv2.imread(img_path)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image originale')

    ax2 = fig.add_subplot(122)
    ax2.text(0.15, 0.9, 'Palette couleur moyenne', fontsize=14)
    imagebox1 = AnnotationBbox(OffsetImage(demo_coul_moy), [0.5, 0.5])
    ax2.add_artist(imagebox1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    return fig


# Kmeans
def fig_kmeans(img_path, df_img_sort, nb_colors):

    colors = []
    explode = []
    seuil_legend = 0
    for i in range(0, nb_colors):
        nt = (df_img_sort.iloc[i, 1] / 255, df_img_sort.iloc[i, 2] / 255,
              df_img_sort.iloc[i, 3] / 255)
        colors.append(nt)
        if i <= 4:
            explode.append(0.1)
            seuil_legend = round(df_img_sort.iloc[i, 4], 2) - 0.01
        else:
            explode.append(0)

    fig = plt.figure(figsize=(10, 5))

    # Image originale
    img = cv2.imread(img_path)
    ax1 = fig.add_subplot(121)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image originale')

    # Extraction couleur K-means
    ax2 = fig.add_subplot(122)
    ax2.pie(df_img_sort['Poids_%'],  # labels=df_img_sort['Color'],
            colors=colors,
            explode=explode,
            labeldistance=0.7,
            autopct=lambda v: f'{v:.2f}%' if v >= seuil_legend else None,
            pctdistance=1.15,
            wedgeprops={"edgecolor": "k", 'linewidth': 1})
    titre = ax2.set_title("Couleurs extraites et poids associés")
    titre.set(color="black", fontsize="14", fontfamily="serif")

    return fig


# Comparaison Palette / K-means
def fig_pal_kmeans(img_path, demo_coul_moy, df_img_sort, nb_colors):
    fig = plt.figure(figsize=(15, 5))

    # Image originale
    ax1 = fig.add_subplot(131)
    img = cv2.imread(img_path)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Image originale')

    ax2 = fig.add_subplot(132)
    ax2.text(0.15, 0.9, 'Palette couleur moyenne', fontsize=14)
    imagebox1 = AnnotationBbox(OffsetImage(demo_coul_moy), [0.5, 0.5])
    ax2.add_artist(imagebox1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    colors = []
    explode = []
    seuil_legend = 0
    for i in range(0, nb_colors):
        nt = (df_img_sort.iloc[i, 1] / 255, df_img_sort.iloc[i, 2] / 255,
              df_img_sort.iloc[i, 3] / 255)
        colors.append(nt)
        if i <= 4:
            explode.append(0.1)
            seuil_legend = round(df_img_sort.iloc[i, 4], 2) - 0.01
        else:
            explode.append(0)

    ax3 = fig.add_subplot(133)
    ax3.pie(df_img_sort['Poids_%'],  # labels=df_img_sort['Color'],
            colors=colors,
            explode=explode,
            labeldistance=0.7,
            autopct=lambda v: f'{v:.2f}%' if v >= seuil_legend else None,
            pctdistance=1.15,
            wedgeprops={"edgecolor": "k", 'linewidth': 1})
    titre = ax3.set_title("Couleurs extraites et poids associés")
    titre.set(color="black", fontsize="14", fontfamily="serif")

    return fig


def graph_demo(type_img_sel, dataset, nb_colors=None, random_sel=True,
               img_path=''):

    if type_img_sel == 'Originale':
        set_path = const.DATASET_ORIGINAL_PATH
        without_bkg = False
    else:
        set_path = const.DATASET_CLEAN_WO_BACKGROUND_PATH
        without_bkg = True

    if random_sel:
        # Choix aléatoire d'images
        ech_rand = random.choice(os.listdir(set_path))
        ech_path = os.path.join(set_path, ech_rand)
        classe_rand = random.choice(os.listdir(ech_path))
        class_path = os.path.join(ech_path, classe_rand)
        image_rand = random.choice(os.listdir(class_path))
        img_path = os.path.join(class_path, image_rand)

    if dataset == datasets[0]:
        res = extract_centileRGB(img_path, without_bkg=without_bkg)
        centile_b, centile_g, centile_r = recup_centileRGB(res)
        fig = fig_centileRGB(img_path, centile_b, centile_g, centile_r)
    elif dataset == datasets[1]:
        res = extract_palette(img_path, nb_colors, without_bkg=without_bkg)
        demo_coul_moy = recup_palette(res, nb_colors)
        fig = fig_palette(img_path, demo_coul_moy)
    elif dataset == datasets[2]:
        res = extract_colors_img_kmeans(img_path, nb_colors,
                                        without_bkg=without_bkg)
        df_img_sort = recup_kmeans(res)
        fig = fig_kmeans(img_path, df_img_sort, nb_colors)
    elif dataset == datasets[3]:
        res = extract_palette(img_path, nb_colors, without_bkg=without_bkg)
        demo_coul_moy = recup_palette(res, nb_colors)
        res = extract_colors_img_kmeans(img_path, nb_colors,
                                        without_bkg=without_bkg)
        df_img_sort = recup_kmeans(res)
        fig = fig_pal_kmeans(img_path, demo_coul_moy, df_img_sort, nb_colors)

    return img_path, fig
