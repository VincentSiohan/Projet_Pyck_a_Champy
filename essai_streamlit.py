# import different modules
import streamlit as st
import os
import pandas as pd
import numpy as np
import time
import datetime
import itertools
import glob
import re

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')

from skimage import io

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Activation,Input,Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D


df=pd.read_csv("image_and_json_data.csv")

# side bar
with st.sidebar:
    st.info("Notebooks disponibles sur GitHub: [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy)")
    st.header("Pyck a Champy")
    st.subheader("Mushroom recognition")
    chap=st.radio("Sélection du chapitre",('0_Introduction','1_Exploration_des_données','2_Segmentation','3_Machine_learning','4_Deep_learning','5_Grad_cam','6_Difficultés_rencontrées','7_Pour_aller_plus_loin','8_Bibliographie'))
#st.write(chap)

if chap=='0_Introduction':
    st.title("Pyck a Champy - Mushroom recognition")
    st.image("images.jpg")
    st.markdown("## But: Prédire la famille d'un champignon parmi une liste de 7 familles à partir d'une image")
    st.write("""
             Promotion:    Data Scientist - Bootcamp - décembre 2021
             \nParticipants: Sébastien Thibert, Antoine Poirot-Bourdain et Vincent Siohan
             \nTuteur:       Louis de datascientest
             """)
    st.info("Notebooks disponibles sur GitHub: [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy)")
    
    st.header("CONTEXTE")
    st.write("""
             La définition d’un objectif pour ce projet était en fait le premier point à définir pour ce projet. En effet, la reconnaissance d’un champignon à partir d’une image ou photo n’est pas un objectif assez précis. En se basant sur le site [mushroomobserver.org](https://mushroomobserver.org), on se rend compte qu’il n’existe pas une liste exhaustive de champignons. D’après le site, moins de 5% des espèces de champignons qui existent dans le monde seraient connues.
             \nDe plus, la définition même d’un champignon ne semble pas non plus être clairement définie: par exemple, les lichens, les rouilles ou les moisissures peuvent être considérés comme des champignons.
             \nEn se basant sur ce site, nous allons restreindre notre projet aux champignons charnus qui sont les champignons  (comestibles ou non) qu’on peut trouver dans les forêts.
             \nNéanmoins, l’objectif n’est toujours pas assez précis. Nous allons donc définir un objectif atteignable et assez précis en faisant une première analyse des données disponibles.
             \nCe projet a donc été traité plutôt d’un point de vue technique.
             \nD’un point de vue économique, si le modèle était plus poussé (beaucoup plus de types de familles reconnues ou en ajoutant une cible comme par exemple si le champignon est comestible ou non) une application sur smartphone pourrait être créée qui reconnaîtrait le champignon qu’on prendrait en photo. Cependant, ce sont des compétences que nous ne possédons pas à ce jour. Par ailleurs,des applications existent déjà et sont disponibles sur l’App Store ou Google Play: “Champignouf”, “Champignong pro”, “IK-Champi”, “Shroomify”, etc. Il en existe une dizaine environ. Les plus performantes (comme “Shroomify”) peuvent identifier jusqu’à environ 400 champignons communs.
             \nLe modèle de reconnaissance de champignons pourrait aussi être utile d’un point de vue scientifique, tout simplement en indiquant aux chercheurs si le champignon qu’ils prennent en photo est déjà connu ou non. Mais, comme pour le point de vue économique, le modèle devrait pouvoir reconnaître beaucoup plus de caractéristiques.
             """)
    
    st.header("OBJECTIFS")
    st.write("""
             Ci-dessous les principales étapes du projet:
             \n1. Recherche sur les données disponibles et de quelle façon les récupérer
             \n2. Exploration et visualisation des données récupérées
             \n3. Définition de l’objectif
             \n4. Modélisations simples (1ère itération) avec du machine learning
             \n5. Modélisations simples (2ème itération) avec du deep learning
             \n6. Essais de différentes modélisations plus adaptées à la computer vision
             \n7. Optimisations des modèles les plus performants
             \n8. Interprétation des résultats avec différentes méthodes (grad-cam, SHAP) et étude de la segmentation (clustering KMeans et MeanShift)
            \nPour atteindre ces objectifs, nous nous sommes bien entendu basé sur les cours de Datascientest, dont particulièrement sur le deep learning (tensorflow, keras).
            \nL’aide de notre tuteur nous a aussi bien aidé en nous permettant de résoudre certains problèmes d’exécution, ainsi qu’en suggérant certaines pistes intéressantes à étudier.
            \nEnfin, l’aide de la communauté web a été indispensable (ex: stackoverflow, kaggle, github, etc.). Voir les liens dans la bibliographie, à la fin du rapport.
             """)


if chap=='1_Exploration_des_données':
    st.title("EXPLORATION DES DONNEES")
    st.info("Données basées sur le site: [mushroomobserver.org](https://mushroomobserver.org)")
    
    st.header("DATA")
    st.write("""
             Plusieurs méthodes étaient possibles pour récupérer des données:
             \n- Web scraping du site  [mushroomobserver.org](https://mushroomobserver.org): cette méthode n’a pas été explorée principalement à cause du temps qu’il aurait fallu pour récupérer les données (images et informations). En effet, pour des raisons d’accessibilité au site, nous ne pouvons faire que 20 requêtes par minute (voir site [lien](https://github.com/MushroomObserver/mushroom-observer/blob/master/README_API.md#csv-files). Si nous voulions récupérer par exemple 200,000 images, il aurait fallu au moins une semaine entière de temps d’exécution. En février 2022, le site compte plus de 530,000 images.
             \n- Un fichier csv avec les liens de plus de 250,000 images était proposé par le site. Mais, nous aurions été de nouveau été contraint aux nombres de requêtes limités par le site
             \n- À l'aide de notre tuteur, nous avons pu récupérer des images et leurs informations à partir d’un ancien projet GitHub ([lien](https://github.com/bechtle/mushroomobser-dataset)). Celui-ci amène au lien suivant qui permet de récupérer les données ([lien](https://www.dropbox.com/sh/m1o91dwd1nto6w0/AABuDQVJWTq04lL_yaF_G2MFa?dl=0)).
             \nCe jeu de données est constitué de 11 dossiers images (un dossier par année; de 2006 à 2016 inclus) et de 12 dossiers json. Les images représentent environ 5 Go et les json environ 440 Mo.
                """)
    
    st.header("Liste des paramètres images et json")
    if st.checkbox("Afficher liste des paramètres du jeux de données de départ (images et json)"):
        st.dataframe(df.head())
    st.write("Nombre d'observations:",len(df))
    
    st.header("Résolution des images")
    st.write("Ci-dessous l’affichage des 5 résolutions les plus présentes:")
    res_count = df['resolution'].value_counts()[:5]
    fig_1=plt.figure(figsize=(20,5))
    ax_1=fig_1.add_subplot(111)
    sns.barplot(x=res_count.index, y=res_count.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title("top5 resolution")
    plt.xlabel("resolution")
    plt.ylabel("count")
    st.pyplot(fig_1)
    st.write("""
             Par la suite, nous ne garderons que la principale résolution (320, 240) = (width, height).
             \nCela nous évite de changer la taille des images, évitant ainsi d’ajouter un biais en “déformant” les images. De plus, nous évitons la tâche fastidieuse de devoir sauvegarder les images transformées.
             """)
    df = df[df['resolution']=='(320, 240)']
    st.write("Nombre d'observations après filtre sur resolution:",len(df))
    
    
    #nb modalités
    st.header("Taxonomie")
    nb_species=len(df['gbif_info.species'].value_counts())
    nb_genus=len(df['gbif_info.genus'].value_counts())
    nb_order=len(df['gbif_info.order'].value_counts())
    nb_family=len(df['gbif_info.family'].value_counts())
    nb_kingdom=len(df['gbif_info.kingdom'].value_counts())
    nb_class=len(df['gbif_info.class'].value_counts())
    nb_phylum=len(df['gbif_info.phylum'].value_counts())
    fig_2 = plt.figure(figsize=(20,5))
    ax_2=fig_2.add_subplot(111)
    sns.barplot(x=['species','genus','order','family','class','phylum','kingdom'],
                y=[nb_species,nb_genus,nb_family,nb_order,nb_class,nb_phylum,nb_kingdom], alpha=0.8)
    plt.xticks(rotation=90);
    plt.title('Number of modalities per rank')
    plt.xlabel('rank')
    plt.ylabel("count")
    st.pyplot(fig_2)
    
    st.subheader("Visualisation de la distribution par hiérachie")
    real_name='gbif_info.species'
    # df_taxonomie=pd.DataFrame({'name':['species','genus','order','family','class','phylum'],
    #                            'real_name':['gbif_info.species','gbif_info.genus','gbif_info.order','gbif_info.family','gbif_info.class','gbif_info.phylum']})
    option=st.selectbox('Sélectionner la hiérachie',('species','genus','order','family','class','phylum'))
    if option=='species':
        real_name='gbif_info.species'
    if option=='genus':
        real_name='gbif_info.genus'
    if option=='order':
        real_name='gbif_info.order'
    if option=='family':
        real_name='gbif_info.family'
    if option=='class':
        real_name='gbif_info.class'
    if option=='phylum':
        real_name='gbif_info.phylum'
    
    val_count = df[real_name].value_counts()[:10]
    fig_3=plt.figure(figsize=(20,5))
    ax_3=fig_3.add_subplot(111)
    sns.barplot(x=val_count.index, y=val_count.values, alpha=0.8)
    plt.xticks(rotation=90)
    plt.title("top10 Number of images per {}".format(option))
    plt.xlabel(option)
    plt.ylabel("count")
    st.pyplot(fig_3)
    
    st.header("Choix cible")
    st.write("""
             Le choix de la cible est le top 7 des familles.
             \n Les 7 familles les plus importantes se détachent des autres,
             avec plus de 4000 images par famille,
             ce qui semble suffisant pour faire de la computer vison,
             sans avoir un jeu trop déséquilibré.
             """)
    
    #select only images with resolution (320,240)= (width,height) => main resolution available
    df = df[df['resolution']=='(320, 240)']
    # drop na on gbif_info.family
    df = df[df['gbif_info.family'].notna()]
    # keep confidence level over 90 
    df = df.loc[df['gbif_info.confidence']>90]
    # keep only most common class = 'Agaricomycetes'
    top_class = df['gbif_info.class'].value_counts().index[0]
    df = df[df['gbif_info.class'] == top_class]
    # keep only top 7 families 
    top_fam = df['gbif_info.family'].value_counts().index[:7].values
    df = df[df['gbif_info.family'].isin(top_fam)]
    # subset col of interest + label encoding
    df_temp = df[['file_path','gbif_info.family']].copy()
    df_temp['label'] = df_temp['gbif_info.family'].replace(df_temp['gbif_info.family'].unique(),
                                             [*range(len(df_temp['gbif_info.family'].unique()))]).astype(str) 
    dict_label_df = pd.DataFrame(df_temp.groupby(['label','gbif_info.family'], as_index=False).size())
    st.dataframe(dict_label_df)
    
    st.write("A ce stade, nous avons réduit notre jeu de données à {} images.".format(len(df_temp)))
    

if chap=='2_Segmentation':    
    st.title("SEGMENTATION")
    st.write("""
             Une partie de ce projet a concerné l’exploration de différentes solutions de segmentation d’image faisant intervenir des algorithmes de clustering (voir [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_5/%5BColab%5D%20%20Image%20segmentation%20.ipynb) et [version colab avec widgets](https://colab.research.google.com/drive/1ZO6_Zz3W8BX_RBihBwXktoyOwbchFf0T?usp=sharing)). 
             """)
    
    st.header("Kmeans")
    st.write("""
             L’image ci-dessous montre qu’il est possible de segmenter l’image en différents clusters correspondant aux différents champignons et au background  grâce à l’algorithme Kmeans.
             \nPar ailleurs, les propriétés de la segmentation dépendent du nombre de features attribuées à chaque pixel :  valeurs des 3 canaux RGB ou valeurs des 3 canaux RGB + valeurs des coordonnées x/y.
             """)
    st.image("segmentation_kmeans.jpg")
    
    st.header("MeanShift")
    st.write("""
             L’algorithme MeanShift permet également d’obtenir des résultats intéressants comme le montre la figure ci-dessous. Mais, il est pénalisé par son temps de calcul qui nécessite une compression des images, notamment pour estimer les hyper paramètres optimaux qui déterminent le nombre de clusters  (bandwidth). 
             """)
    st.image("segmentation_meanshift.jpg")
    
    st.header("Autres méthodes")
    st.write("""
             Finalement, des méthodes plus élaborées basées sur la détection des bords  et l’histogramme d’intensité des pixels en niveau de gris permettent aussi d’obtenir des champignons détourés comme le démontre l’exemple ci-dessous.
             \nToutefois, toutes les méthodes étudiées souffrent de leur temps de calcul et surtout du manque d’une méthode automatisée permettant de trouver les hyper paramètres optimaux pour chacune des images vu la grande diversité de ces dernières. Or, ce sont ces hyper paramètres qui déterminent le nombre de clusters et la qualité de la segmentation.
             \nDes méthodes basées sur du deep learning avancé existent (cf [lien](https://towardsdatascience.com/background-removal-with-deep-learning-c4f2104b3157)). Mais ces dernières étaient hors de portée pour ce projet de 5 semaines. 
             """)
    st.image("segmentation_autres.jpg")


if chap=='3_Machine_learning':
    st.title("MACHINE LEARNING")
    st.header("Influence des paramètres de base")
    st.write("""
             Dans un premier temps, une approche utilisant uniquement des algorithmes de machine learning a été utilisée suivant cette méthode:
             \n- chacune des images est compressée en réduisant sa résolution
             \n- les images sont ensuite transformées en une matrice de dimensions N = colonnes de pixels x lignes de pixels x 3 canaux RGB 
             \n- toutes les images sont concaténées dans une matrice de dimensions N x nombre d’images (un pixel ⇔ un feature , une ligne ⇔ une image)
             \n- un split train/test de 80/20% est appliqué avec option stratify
             \n- un algorithme de machine learning est entraîné sur le train test 
             \n- l’algorithme est finalement évalué sur le test set
             \nLa figure ci-dessous (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_2/%5BDRIVE%5D%20DOE%20resolution%20and%20class%20number_Random%20forest.ipynb)) montre qu’avec un algorithme de type random forest (avec les hyper paramètres de base), une accuracy de l’ordre 0,27 est obtenue avec 5 familles contre 0,16 avec 21 familles. Par contre, la résolution n’a qu’un faible effet sur la précision du modèle alors qu’elle a un effet très important sur le temps d'entraînement et le besoin de RAM. C’est pour cette raison (crash du kernel)  qu’il n’y a pas de point pour la résolution (72, 96) avec plus de 11 familles (courbes bleues).  Des résultats similaires ont été obtenus avec un classifieur de type SVC. 
             """)
    st.image("ml_rf_accuracy_vs_resolution.jpg")
    st.image("ml_rf_time to fit_vs_resolution.jpg")
    
    st.header("XGBoost & Optuna")
    st.write("""
             Malgré les limitations évidentes d’une approche machine learning pour ce type de problème, des essais ont été menés avec le classifier XGBOOST et la librairie d’optimisation bayésienne des hyper paramètres Optuna (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_2/Copie%20de%20%5BDRIVE%5D%20ML%20hyperparameter%20optimization_0.32%20acc%20with%207%20classes.ipynb)).  Ce package est un logiciel d'optimisation automatique des hyperparamètres, particulièrement conçu pour l'apprentissage automatique. Il est basé sur l’optimisation bayésienne qui est une stratégie de conception séquentielle pour l'optimisation globale de fonctions boîte noire qui ne suppose aucune forme fonctionnelle. Elle est généralement employée pour optimiser des fonctions coûteuses à évaluer (cf [lien](https://en.wikipedia.org/wiki/Bayesian_optimization)). Le tableau ci-dessous montre que la combinaison de XGBOOST et de cette  méthode a permis d’obtenir une accuracy de 0,32 sur 7 familles.
             """)
    st.image("ml_xgboost et optuna_cnf matrix.jpg")
    st.write("""
             Pour comprendre comment XGBOOST fait ses prédictions, l’importance de chacune des features (ici les pixels) a été extraite puis transformée en une image en niveaux de gris (couleur d’un pixel proportionnelle à la moyenne des 3 niveaux RGB). La figure ci- dessous montre ainsi que les pixels au centre sont les plus utilisés lors de la classification. Cela semble pertinent, car les photos sont en général réalisées avec le champignon en son centre. 
             """)
    st.image("ml_importance features with xgboost.jpg")


if chap=='4_Deep_learning':
    st.title("DEEP LEARNING")
    st.header("Comparaison des modèles de base")
    st.write("""
             Dans le but d’identifier rapidement les architectures des réseaux qui fonctionnent bien sur notre jeu de données, différents modèles CNN ont été testés sur une partie de notre jeu de données limitée à 5000 images pour limiter le temps de calcul ([lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_3/%5BDRIVE%5D%20DL%20models%20comparison.ipynb)). Les modèles de base tels que EfficientNetB1 ou Inception ont été chargés avec les poids ‘imagenet’ dans une architecture qui leur ajoute une couche fully connected (2 couples de couches dense/dropout + couche de prédiction). La figure suivante montre que  le modèle EfficientNetB1 (EFB1) permet d’obtenir les meilleures performances en un temps raisonnable.
             \nOn peut toutefois remarquer que le modèle MobileNetV3Small est le plus performant en temps de calcul avec une accuracy légèrement plus faible que celle d’EfficientNetB1.
             """)
    st.image("dp_comparaison modèles_accuracy.jpg")
    st.image("dp_comparaison modèles_duration.jpg")
    st.write("temps de calculs (=fit_duration) en secondes - obtenus avec google colab.")
    st.subheader("Générateurs d'images")
    st.write("""
             Par la suite, le jeu de données retenues (35787 images) a été divisé en trois jeux (train, validation et test; 64%, 16% et 20 du jeu).
             \nDeux méthodes de générateur d’Images ont été étudiées:
             \n- flow_from_dataframe
             \n- flow_from_directory
             \nLa première méthode est la moins contraignante puisqu’elle nous permet d’accéder aux images grâce à la colonne “file_path” d’un dataframe. C’est celle-ci qui a été privilégiée par la suite.
             \nLa deuxième méthode, nous oblige à enregistrer les images suivant la répartition des jeux train, validation et test. Cette méthode classe par défaut les données cibles dans l’ordre alphabétique (ce qui n’est pas le cas de la première méthode). Un exemple d’utilisation est disponible sous le lien suivant: [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Vincent/MobileNetV3Small/MobileNetV3Small_optimization.ipynb). 
             """)
    
    st.header("EfficientNetB1")
    st.write("""
             Suite aux résultats précédents, le modèle EFB1 a été retenu pour tenter d’améliorer l’accuracy. La figure ci-dessous montre qu’en fonction du problème, différentes approches peuvent être retenues. Le jeu de données qui fait l’objet des prochaines expériences est constitué d’au moins 4000 images pour la classe minoritaire (ramené à 600 quand la taille du jeu de données est restreinte à 5000 images pour accélérer le temps de calcul) ce qui limite le choix à la partie supérieure. 
             \nPar ailleurs, des tests préliminaires ont montré qu’il était bien plus efficace de partir d’un réseau pré-entraîné (transfer learning) sur imagenet qu’à partir d’un réseau initialisé de façon aléatoire. C’est cette approche qui a été retenue. Cependant, comme il est difficile d’évaluer la ressemblance entre le jeu de données et la base de données imagenet, les deux méthodes (quadrants 1 & 2) de la figure suivante ont été comparées. 
             """)
    st.image("dp_efb1_diagramme aide au choix.jpg")
    st.write("""
             Tout au long des expériences suivantes, le package Optuna a été extensivement utilisé sur le jeu de données restreint à 5000 images  pour optimiser les hyper paramètres (learning rate , batch size, architecture de la couche fully connected, utilisation de la data augmentation, …). Par exemple, sur la figure suivante extraite du notebook [DL Model optimization with optuna + unfreeze base model](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BDRIVE%5D%20DL%20Model%20optimization%20with%20optuna%20%2B%20unfreeze%20base%20model.ipynb), le learning rate a plus d’importance, que le batch_size et ainsi de suite. Il est ensuite possible de choisir l’essai ayant donné les meilleurs résultats ou de relancer une recherche dans un espace restreint (nombre d'hyper paramètres ou intervalle de recherche). 
             """)
    st.image("dp_hyperparam importance.jpg")
    st.image("dp_hyperparam importance_2.jpg")
    st.write("""
             Les résultats présentés par la suite se concentrent donc sur les meilleurs modèles issus des expériences menées avec Optuna sur 5000 images, puis  appliqués au jeu de données complet (un essai a montré qu'utiliser Optuna sur le jeu complet n'amène pas d’amélioration). Il faut également noter que tous ces tests ont été menés en synergie avec l'utilisation de deux callbacks: 
             \n- ReduceLROnPlateau: pour réduire le learning rate lorsque l’accuracy du jeu de données de validation atteint un plateau  ou augmente  durant N epochs (en général d’un facteur 10 au bout de 2 epochs dans ces travaux).
             \n- EarlyStopping: pour stopper l'entraînement et limiter le surapprentissage lorsque l’accuracy du jeu de données de validation atteint un plateau ou augmente durant N epochs (en général au bout de 3 epochs dans ces travaux).
             """)
    st.subheader("Approche 'unfreeze at start'")
    st.write("""
             Pour réaliser cette méthode, après une initialisation avec les poids imagenet, toutes les couches du réseau de neurones sont dégelées dès le début de l'entraînement. La figure ci-dessous, issue du notebook [Best EFB1 model on all images_unfreeze at start](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BDRIVE%5D%20Best%20EFB1%20model%20on%20all%20images_unfreeze%20at%20start%20.ipynb), montre qu’une accuracy de 0,76  a pu être mesurée sur le test set. Elle a été obtenue en adjoignant une couche fully connected composée de deux couples de couches dense/dropout (128 neurones/25% - 32 neurones/25%) et d’une couche de prédiction  à la partie convolutive d’EFB1.  Malgré l’utilisation des callbacks, les courbes d'entraînement montrent un certain overfit (gap entre les courbes obtenues sur les deux jeux de données). 
             """)
    st.image("dp_efb1_unfreeze at start.jpg")
    
    st.write("""
             Certains des essais menés avec Optuna ont montré que l’overfit pouvait être réduit, mais c’est au détriment des performances sur le test set. Des essais ont également été menés en utilisant de la data augmentation. Mis à part une augmentation considérable du temps d'entraînement, ces derniers n’ont ni montré une amélioration sur le test set, ni une réduction de l’overfit. Cette observation montre ainsi qu’avec plus de 4000 images par classe, le jeu de données est assez diversifié et ne bénéficie pas d’une data augmentation supplémentaire. Celà est confirmé en visualisant les images sur la figure ci-dessous qui montre la grande diversité de celles-ci : 
             \n- vue du dessus
             \n- vue du dessous
             \n- vue sur le côté
             \n- champignon coupé
             \n- champignon tout seul sur une table
             \n- champignon tenu par une main
             \n- plusieurs champignons sur la même image
             \n- etc.
             """)
    st.image("images.jpg")
    st.write("""
             Finalement, des expériences ont été conduites en essayant d’appliquer soit une couche GlobalAverage à la place de la couche fully connected, soit un profil de learning rate (inspiré de ce notebook [lien](https://colab.research.google.com/drive/18rmiOPkoYg1MZ6ajhg3UVP1BE5Ur92pF) et optimisé via Optuna [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BDRIVE%5D%20%20DL%20model%20optimization%20on%205k%20images%20%20diff%C3%A9rent%20lr%20profiles%20with%20FC%20architecture%20%26%20unfreeze%20at%20start%20.ipynb)). Bien qu’elles aient permis de réduire considérablement l’overfit, les performances finales sur le jeu de test étaient moins bonnes (0,57 vs 0,65 sur le jeu de données réduit) . 
             """)
    
    st.subheader("Approche 'freeze at start + fine tuning'")
    st.write("""
             Durant cette approche, les essais ont été réalisés en deux temps:
             \n- Dans un premier temps, la couche fully connected a été optimisée en fixant son architectures à deux deux couples de couches dense/dropout et en laissant libre leur nombre de neurones et leur dropout rate, mais aussi le learning rate et le batch size. Durant cette étape, le réseau EFB1 a été initialisé avec les poids imagenet et ses couches ont été gelées. Le modèle a pu s'entraîner jusqu’à la détection de la convergence via un callback early stopping.
             \n- Dans un second temps, toutes les couches du modèle sont dégelées avant de relancer un entraînement en commençant avec le learning rate de l’étape précédente et en le réduisant grâce au callback ReduceLROnPlateau.
             \nLa figure ci-dessous montre qu’avec cette méthode, une accuracy de 0,65 est obtenue à la fin de la première phase, puis celle-ci augmente au-dessus de 0,70 dès que les couches du réseau convolutif sont dégelées (ligne verticale noire). Malgré l’overfit qui est supérieur à celui obtenu avec la méthode précédente, un score de 0,79 a  été mesuré sur le test set (cf [[Colab] FINAL EFB1 model on all images_freeze at start + fine tuning](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Modelisation/%5BColab%5D%20FINAL%20EFB1%20model%20on%20all%20images_freeze%20at%20start%20%2B%20fine%20tuning%20.ipynb)). 
             """)
    st.image("dp_efb1_freeze at start_&_fine tuning.jpg")
    st.image("dp_efb1_freeze at start_&_fine tuning_2.jpg")
    
    st.write("""
             Pour tenter de réduire l’overfit, différents essais ont été menés toujours avec l’aide d’Optuna:
             \n- Ajout de la data augmentation via un data generator 
             \n- Optimisation de l’étape de fine tuning en laissant Optuna choisir les meilleurs paramètres parmi ceux d’un callback LearningRateScheduler à décroissance exponentielle et le nombre de couches à dégeler (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BDRIVE%5D%20DL%20Model%20optimization%20on%205k%20images_transfer%20learning%20%2B%20fine%20tuning%20with%20FC%20classifier%20.ipynb))
             \nAucune de ces approches n’a permis d’égaler les performances présentées dans le paragraphe précédent. Par exemple, la figure ci-dessous (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BColab%5D%20Best%20EFB1%20with%20partial%20unfreeze%20and%20exponential%20lr%20decrease%20.ipynb)) montre qu’en dégelant uniquement 250 sur les 339 couches et en appliquant une décroissance exponentielle du learning rate après décongélation des couches, l’overfit est légèrement réduit (gap entre les courbes de training et de validation), mais l’accuracy sur le test set subit une perte de 4% par rapport aux conditions précédentes.
             """)
    st.image("dp_efb1_freeze at start_&_fine tuning_3a.jpg")
    st.image("dp_efb1_freeze at start_&_fine tuning_3b.jpg")
    
    st.header("EfficientNetB1 + SVC")
    st.write("""
             La méthode consistant à utiliser un classifieur de type machine learning après extraction des features par le réseau de neurone a également été étudiée ([lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_5/%5BColab%5D%20%20Final%20DL%20EFB1%20model%20%2B%20SVC%20on%205k%20images%20.ipynb)). Le modèle de la section précédente a d’abord été entraîné, puis un SVC a été utilisé pour classer les features extraits à différents endroits du réseau. Sur la figure ci-dessous, Il est intéressant de noter que  l’accuracy augmente avec la position de la couche utilisée pour extraire les features. Cela démontre que plus les couches sont loin dans le réseau, plus elles sont spécialisées au problème étudié  (N.B. l’accuracy est ici surévalué car le SVC a été appliqué sur une cross-validation de la combinaison du train set et du set d'évaluation utilisés pour entraîner le réseau de neurones).
             """)
    st.image("dp_efb1 & svc_1.jpg")
    st.write("""
             En appliquant le meilleur set d’hyper paramètres (couche global average en sortie de la partie convolutive et C = 1) sur tout le jeu de données, cette méthode a donné des résultats comparables à celle de la section précédente avec une accuracy de 0,79  sur le test set ([lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_5/%5BColab%5D%20FINAL%20EFB1%20%2B%20SVC%20%20model%20on%20all%20images%20.ipynb)).
             """)
    st.image("dp_efb1 & svc_2a.jpg")
    st.image("dp_efb1 & svc_2b.jpg")
    
    st.header("Stacking")
    st.write("""
             Pour tenter d’améliorer encore l’accuracy, la méthode précédente (optimisation de l’approche ‘freeze at start + fine tuning’ sur 5000 images avec Optuna) a été reproduite avec les différents modèles ayant montré des bonnes performances dans la section Comparaison des modèles de base. Par la suite,  chacun des modèles optimisés (VGG19, EFB1, MobileNetV3Large, resnet50, VGG16) a été entraîné sur le jeu d'entraînement, puis ils ont été utilisés pour générer des prédictions (N images x 7 probabilités d’appartenance aux différentes familles). Individuellement, ils montrent une accuracy sur le test set s’étalant de 0,68 pour VGG19  à 0,79 pour EFB1.
             \nFinalement, un modèle XGBOOST a permis de tirer parti de ces prédictions pour réaliser un stacking de ces modèles. La figure ci-dessous (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Modelisation/%5BColab%5D%20Stacking%20DL%20model%20%20on%20all%20images_5%20models%20.ipynb)) montre qu’avec cette approche un gain substantiel de 1% d’accuracy par rapport au meilleur modèle peut être obtenu. Peut être que l'utilisation de modèles plus diversifiés (e.g. meilleurs pour prédire les classes 1, 4 et 5) ou une pondération de l'importance des prédictions en fonction des réseaux pourrait  améliorer encore l’accuracy. Cependant, dans les quelques notebooks kaggle étudiés, le gain entre le meilleur modèle et le stacking de plusieurs modèles CNN restent souvent limités à 1 ou 2%.
             """)
    st.image("df_stacking.jpg")
    
    st.header("Autre bibliothèque utilisant la 'Bayesian optimization'")
    st.write("""
             Une autre bibliothèque plus simple qu’Optuna est la bibliothèque 'bayesian-optimization'.
             \nLa description est disponible sur GitHub: [https://github.com/fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization).
             \nCette piste a été étudiée en parallèle sur un modèle MobileNetV3Small. Nous avons choisi ce modèle pour le faible temps d’exécution. L’objectif ici n’était pas d’atteindre la meilleure optimisation possible (qui semblait être atteinte avec l’EfficientNetB1 + optimisation).  Mais, seulement d’explorer cette autre méthode.
             \nLe notebook est disponible sous GitHub ([lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Vincent/Bayesian_optimization/MobileNetV3Small_Bayesian%20opti.ipynb)).
             \nSon application est relativement aisée.
             \nMais, il faut faire attention à notamment deux points:
             \n- Par défaut, cette méthode va chercher la valeur maximale de performance. Il faut donc choisir la métrique “accuracy”.
             \n- Par défaut, à chaque itération les valeurs des hyper paramètres sont choisies dans l’étendue définie. Mais elles ne prennent pas de valeurs discrètes. Si on veut ajouter des paramètres discrets (comme batch_size ou epochs), il faut passer par la définition d’une fonction spécifique (voir: [lien](https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb); cf. paragraphe 2).
             """)
    
    st.header("Comparaison avec 'AutoML' et précédents travaux")
    st.write("""
             Pour pouvoir comparer les performances des modèles développés lors de cette étude, la librairie d’AutoML Autogluon a été utilisée dans les mêmes conditions que les expériences précédentes (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_4/%5BDRIVE%5D%20Autogluon%20classification.ipynb)). La figure ci-dessous montre qu’une accuracy de 0,70, a été mesurée, soit 10% en dessous des meilleurs résultats présentés dans ce rapport.  
             """)
    st.image("dp_automl_1.jpg")
    st.image("dp_automl_2.jpg")
    st.write("""
             Par ailleurs, un groupe d’anciens élèves avait développé un modèle permettant d’obtenir un score de 0,77 sur 5 familles (cf [lien](https://github.com/thibaultkaczmarek/MushPy/blob/main/rapport_final/20210716_Rapport_MushPy_final.pdf)) avec le modèle EFB1.  En limitant à 5 familles, les modèles basés sur EFB1 et développés dans les sections précédentes permettent d’obtenir une accuracy supérieure à 0,83 (cf [lien](https://github.com/DataScientest-Studio/Pyck_a_Champy/blob/main/Sebastien/Iteration_5/%5BColab%5D%20FINAL%20EFB1%20model%20on%205%20families%20all%20images_freeze%20at%20start%20%2B%20fine%20tuning%20.ipynb)), le gain de 6% étant probablement dû à l’utilisation d’Optuna pour optimiser les hyper paramètres des modèles utilisés.
             """)
    

if chap=='5_Grad_cam':
    st.title("GRAD-CAM")
    st.header('Principe du Grad-Cam')
    st.info("Voir exemple sur Keras: [lien](https://keras.io/examples/vision/grad_cam/)")
    st.write("""
             Le principe de cette méthode est d'identifier la dernière couche de convolution.
             \nPuis, de calculer un **heatmap** qui est représentation visuelle de l'activation des pixels de la dernière couche de convolution.
             Plus la valeur associée au pixel est élevée, plus la probabilité d'activer ce pixel est élevée.
             \nLe calcul de ce **heatmap** se fait par une descente de gradient.
             \nCi-dessous un exemple de **heatmap** suivant deux cmap différents (le calcul a été fait à partir de la même image de départ)
             """)
    st.subheader("Exemple Image d'entrée")
    st.image("gradcam_input_img.jpg")
    st.subheader("Exemples **heatmap** avec 2 cmap")
    st.image("heatmap.jpg")
    st.write("""
             Pour l'image à gauche, les pixels activés sont ceux proches du vert ou jaune.
             \nPour l'image à droite, les pixels activés sont ceux proches du roughe ou jaune.
             """)
    st.subheader("Exemple sortie")
    st.image("gradcam_output.jpg")
    st.write("""
             Les pixels activés correspondent plutôt bien au champignon présent sur l'image.
             """)
    
    st.header('Exemples résultats sur la class3 (meilleur f1-score à 0.87)')
    st.image("gradcam_output_class3.jpg")
    st.header('Exemples résultats sur la class2 (2ème meilleur f1-score à 0.84)')
    st.image("gradcam_output_class2.jpg")
    st.header('Exemples résultats sur la class4 (moins bon f1-score à 0.68)')
    st.image("gradcam_output_class4.jpg")
    st.header('Exemples résultats sur chaque classe à partir de la même image')
    st.image("gradcam_output_all_class.jpg")
    st.write("""
             L'image d'origine est dans le Label3 = Boletaceae.
             \nLe modèle final prédit un Label3.
             \nLe Grad-Cam montre aussi que le Label3 est le résultat le plus proche de la réalité.
             """)
    
    
    st.header("Appliquer le Grad-Cam sur une image du jeu de test")
    st.warning("""
             **ATTENTION***:
             \n    1. le chargement du modèle peut prendre environ 30 secondes
             \n    2. Le calcul des prédictions peut prendre environ 30 secondes
             \nCliquer sur les checkbox dans l'ordre!
             """)
    st.subheader("Chargement du modèle final")
    #select only images with resolution (320,240)= (width,height) => main resolution available
    df = df[df['resolution']=='(320, 240)']
    # drop na on gbif_info.family
    df = df[df['gbif_info.family'].notna()]
    # keep confidence level over 90 
    df = df.loc[df['gbif_info.confidence']>90]
    # keep only most common class = 'Agaricomycetes'
    top_class = df['gbif_info.class'].value_counts().index[0]
    df = df[df['gbif_info.class'] == top_class]
    # keep only top 7 families 
    top_fam = df['gbif_info.family'].value_counts().index[:7].values
    df = df[df['gbif_info.family'].isin(top_fam)]
    # subset col of interest + label encoding
    df_temp = df[['file_path','gbif_info.family']].copy()
    df_temp['label'] = df_temp['gbif_info.family'].replace(df_temp['gbif_info.family'].unique(),
                                             [*range(len(df_temp['gbif_info.family'].unique()))]).astype(str) 
    dict_label_df = pd.DataFrame(df_temp.groupby(['label','gbif_info.family'], as_index=False).size())

    # Train/test
    df_train,df_test=train_test_split(df_temp, train_size=0.8, stratify =df_temp['label'], random_state=42)
    # train / val 
    df_train_,df_val_=train_test_split(df_train, train_size=0.8,stratify =df_train['label'],random_state=42)
    
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_data = ImageDataGenerator().flow_from_dataframe(dataframe = df_test,
                                                         x_col = "file_path",y_col="label", 
                                                         shuffle = False, # to match df_test order 
                                                         target_size=(240,320), # (height,width)
                                                         batch_size=16,
                                                         class_mode='sparse')
    
    checkbox_load_model=st.checkbox("1_Charger le modèle")
    if checkbox_load_model:
        with st.spinner('Chargement du modèle en cours'):
        #     start=time.time()
        #     #load the final optimized model
        #     efficientnet_final = tf.keras.models.load_model('EfficientNetB1_final_model_seb')
        #     total_time=time.time()-start
        # st.success('model chargé en {:.1f} secondes'.format(total_time))
    
    #ne marche pas: conflit entre st.cache et load_model de keras
            @st.cache(allow_output_mutation=True)
            def load_model(filename):
                efficientnet_final=tf.keras.models.load_model(filename)
                return efficientnet_final
    
            start=time.time()
            efficientnet_final=load_model('EfficientNetB1_final_model_seb')
            total_time=time.time()-start
            st.write('model chargé en {:.1f} secondes'.format(total_time))
    
    
    from tensorflow import keras
    from IPython.display import Image, display
    import matplotlib.cm as cm
    #function to preprocessed correctly the image
    def get_img_array(img):
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array
    
    st.subheader("Choisir l'image à traiter")
    indice=st.number_input("Choix de l'image (entre 0 et 7157)", min_value=0,max_value=7157,value=4,step=1)
    batch=indice//16 #operation "//16" as the test_data was divided into batches of 16 images
    i=indice%16 #to obtain the real index of the test_data 
    im=np.array(test_data[batch][0][i],dtype=np.uint8) #[batch][0][index]

    fig_4=plt.figure(figsize=(20,5))
    ax_4=fig_4.add_subplot(111)
    plt.imshow(im)
    plt.axis('off')
    st.pyplot(fig_4)
    # Prepare the selected image
    img_array = get_img_array(img=im)     
    
    
    st.subheader("Calcul des prédictions sur le jeu de test")
    input_tensor = tf.keras.Input(shape=(240,320,3)) #input with the shape of the images to be processed
    #base_model = EfficientNetB1 without top layers of classification
    from tensorflow.keras.applications import EfficientNetB1
    base_model=EfficientNetB1(weights='imagenet',
                              include_top=False,
                              input_tensor = input_tensor)
    #apply the weights of the the first layer of efficientnet_final model to the base_model
    base_model.load_weights('EfficientNetB1_final_model_seb_weights/base_model/my_checkpoint')
    
    from tensorflow.keras.layers import BatchNormalization, Activation
    last_conv = base_model.get_layer('top_conv') #identify the last convolution layer
    
    #define the model from the last_conv layer
    #all the layers are the same as the customised efficientnet classification top layers
    #with weights from efficientnet_final
    x=BatchNormalization(weights=base_model.layers[-2].weights)(last_conv.output)
    x=Activation('swish',weights=base_model.layers[-1].weights)(x)
    x= GlobalAveragePooling2D(weights=efficientnet_final.layers[1].weights)(x)
    x= Dense(1024, activation='relu',weights=efficientnet_final.layers[2].weights)(x)
    x= Dropout(0.3,weights=efficientnet_final.layers[3].weights)(x)
    x= Dense(512, activation='relu',weights=efficientnet_final.layers[4].weights)(x)
    x= Dropout(0.2,weights=efficientnet_final.layers[5].weights)(x)
    x= Dense(7, activation='softmax',weights=efficientnet_final.layers[6].weights)(x)
    new_model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
    
    if st.checkbox("Calculer les prédictions"):
        with st.spinner("Calcul des prédictions en cours"):
            result_new_model=new_model(img_array)
            st.write(result_new_model)
            label_pred=tf.argmax(new_model(img_array),axis=1).numpy()
            st.write('Label prédit:',label_pred)
            real_label=df_test.label.iloc[indice]
            st.write('Label réel:',real_label)
    
    st.subheader("Affichage du heatmap")
    
    
    #function to calculate the heatmap of the image output at the "top_conv" layer
    def make_gradcam_heatmap(img_array, model, last_conv_name):
        grad_model = Model([model.inputs],[model.get_layer(name=last_conv_name).output,model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output,preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    if st.checkbox("Calculer et afficher le heatmap"):
        with st.spinner("Calcul du heatmap en cours"):
            # Generate class activation heatmap
            heatmap = make_gradcam_heatmap(img_array, new_model,'top_conv')
            # Rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)
            
            # Use jet colormap to colorize heatmap (blue for low values and red for high values)
            jet = cm.get_cmap("jet")
            
            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]
            
            # Create an image with RGB colorized heatmap
            jet_heatmap=keras.preprocessing.image.array_to_img(jet_heatmap)
            jet_heatmap=jet_heatmap.resize((img_array.shape[2], img_array.shape[1])) #apply same resolution as the image
            jet_heatmap=keras.preprocessing.image.img_to_array(jet_heatmap)
            
            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * 0.4 + img_array[0] #0.4 = transparency of the heatmap
            superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
            
            # Save the superimposed image
            superimposed_img.save("cam.jpg")
            
            # Display Grad CAM
            st.image("cam.jpg")


if chap=='6_Difficultés_rencontrées':
    st.title("DIFFICULTES RENCONTREES")
    st.write("""
             Globalement, il n’a pas été évident de mener en parallèle le projet ainsi que les cours et les examens. Un investissement particulier notamment durant les week-ends est quasiment nécessaire.
             \nDe plus, certaines compétences techniques comme le deep learning n’étaient pas forcément acquises par rapport au cursus normal. Il a donc fallu redoubler d’efforts pour les acquérir en avance par rapport au planning normal pour être en phase avec le planning du projet.
             \nD’autres compétences techniques n’étaient tout simplement pas dans le programme de formation telles que les optimisations Optuna et Bayesian ainsi que certaines méthodes d’interprétabilité comme Grad-cam.
             \nNous nous sommes aussi rendu compte de l’importance du matériel informatique pour le temps d’exécution. Chacun des membres ayant un ordinateur avec des caractéristiques différentes, nous avons pu comparer le temps d’exécution sur les mêmes modèles. Par exemple, pour entraîner un VGG19 sur 5000 images, il a fallu 6 min avec Google colab (avec l’aide d’un GPU performant). Alors qu'il a fallu 5h30 avec un dell (Processeur i7-8550U CPU @ 1.80Ghz 1.99GHz; 16Go RAM; GPU 4Go). La gestion du temps en prenant en compte les performances du matériel informatique disponible est donc un point très important.
             \nRemarque sur Google colab: l’utilisation d’un GPU étant limitée, il peut être nécessaire de prendre un abonnement payant pour enlever cette limitation. Ou alors investir dans un ordinateur performant.
             \nFinalement, ce projet nous a permis de consolider nos compétences acquises lors de notre formation. Mais, il nous a aussi permis d’acquérir des compétences supplémentaires particulièrement sur le traitement des images et leur interprétations.
             """)


if chap=='7_Pour_aller_plus_loin':
    st.title("POUR ALLER PLUS LOIN")
    st.write("""
             Il reste un grand nombre d’approches qui n’ont pas été testées et qui pourraient permettre d’améliorer un peu les performances sur ce jeu de données. L’utilisation d’un modèle EFB0 pour limiter l’overfit, l’application de différents profils de learning rate (cf [lien](https://www.jeremyjordan.me/nn-learning-rate/)), l’initialisation avec des poids autres que ‘imagenet’ ou encore le stacking de modèles plus diversifiés en font partie.
             \nToutefois, les résultats de la section interprétabilité semblent montrer que la principale limitation est dû au fait que les modèles ont parfois du mal à identifier la partie de l’image correspondant au champignon. Par conséquent, une approche combinant de la segmentation et du deep learning pourrait être envisagée. Comme la diversité des images semble empêcher l’utilisation d’algorithmes de segmentation simples, il faudrait se tourner vers les solutions plus élaborées telle que celle proposée dans ce [lien](https://towardsdatascience.com/background-removal-with-deep-learning-c4f2104b3157) pour supprimer le background des images avant leur classification. 
             """)


if chap=='8_Bibliographie':
    st.title("BIBLIOGRAPHIE")
    st.header("Taxonomie et récupération des données:")
    st.write("""
             \n1. [https://mushroomobserver.org/](https://mushroomobserver.org/)
             \n2. [https://github.com/MushroomObserver/mushroom-observer/blob/master/README_API.md#csv-files](https://github.com/MushroomObserver/mushroom-observer/blob/master/README_API.md#csv-files)
             \n3. [https://github.com/bechtle/mushroomobser-dataset](https://github.com/bechtle/mushroomobser-dataset)
             \n4. [https://www.dropbox.com/sh/m1o91dwd1nto6w0/AABuDQVJWTq04lL_yaF_G2MFa?dl=0](https://www.dropbox.com/sh/m1o91dwd1nto6w0/AABuDQVJWTq04lL_yaF_G2MFa?dl=0)
             """)
    
    st.header("Google colab")
    st.write("""
             5. [https://ledatascientist.com/google-colab-le-guide-ultime/](https://ledatascientist.com/google-colab-le-guide-ultime/)
             """)
    
    st.header("Bibliothèques et tutoriels:")
    st.subheader("Gestion des images")
    st.write("""
             6. [https://he-arc.github.io/livre-python/pillow/index.html](https://he-arc.github.io/livre-python/pillow/index.html)
             \n7. [https://pypi.org/project/python-resize-image/](https://pypi.org/project/python-resize-image/)
             \n8. [https://stackoverflow.com/questions/48001890/how-to-read-images-from-a-directory-with-python-and-opencv](https://stackoverflow.com/questions/48001890/how-to-read-images-from-a-directory-with-python-and-opencv)
             \n9. [https://stackoverflow.com/questions/44078327/fastest-approach-to-read-thousands-of-images-into-one-big-numpy-array](https://stackoverflow.com/questions/44078327/fastest-approach-to-read-thousands-of-images-into-one-big-numpy-array)
             """)
    st.subheader("Gestion des dossiers")
    st.write("""
             10. (https://docs.python.org/3/library/glob.html?highlight=glob#module-glob](https://docs.python.org/3/library/glob.html?highlight=glob#module-glob)
             \n11. [https://docs.python.org/fr/3/library/json.html](https://docs.python.org/fr/3/library/json.html)
             """)
    st.subheader("Mesurer le temps d'exécution d'une commande")
    st.write("""
             12. [https://pypi.org/project/tqdm/](https://pypi.org/project/tqdm/)
             \n13. [https://saladtomatonion.com/blog/2014/12/16/mesurer-le-temps-dexecution-de-code-en-python/](https://saladtomatonion.com/blog/2014/12/16/mesurer-le-temps-dexecution-de-code-en-python/)
             """)
    st.subheader("Classification des images")
    st.write("""
             14. [https://www.kaggle.com/zayon5/image-classification-dog-and-cat-images](https://www.kaggle.com/zayon5/image-classification-dog-and-cat-images)
            \n15. [ https://github.com/fmfn/BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
             """)
    st.subheader("Interprétabilité")
    st.write("""
             16. [https://keras.io/examples/vision/grad_cam/](https://keras.io/examples/vision/grad_cam/)
             \n17. [https://pypi.org/project/grad-cam/](https://pypi.org/project/grad-cam/)
             \n18. [https://shap.readthedocs.io/en/latest/]https://shap.readthedocs.io/en/latest/()
             \n19. [https://christophm.github.io/interpretable-ml-book/](https://christophm.github.io/interpretable-ml-book/)
             """)
    st.subheader("Notebooks")
    st.write("""
             20. [https://github.com/DataScientest-Studio/Pyck_a_Champy](https://github.com/DataScientest-Studio/Pyck_a_Champy)
             \n21. [https://colab.research.google.com/drive/1ZO6_Zz3W8BX_RBihBwXktoyOwbchFf0T?usp=sharing](https://colab.research.google.com/drive/1ZO6_Zz3W8BX_RBihBwXktoyOwbchFf0T?usp=sharing)
             """)