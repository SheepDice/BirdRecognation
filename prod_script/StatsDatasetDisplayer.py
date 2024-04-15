import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from IPython.display import display, HTML
import json
from PIL import Image

class StatsDatasetDisplayer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.samples = [setPath for setPath in os.listdir(self.dataset_path)]
        self.count_image_of_sets = self.count_image_by_set()
        self.qtt = list()
        self.mini = list()
        self.maxi = list()
        self.moy = list()
        self.calcul_metrics()
        self.total = sum(self.qtt)
        self.df_train = self.create_df_train()
        self.mean_by_bird_r = None
        self.mean_by_bird_g = None
        self.mean_by_bird_b = None
        self.read_rgb_json()
        
    def count_image_by_set(self):
        temp_dict = dict() 
        for source in self.samples:
            #Chemin de l'échantillon
            path_sample = os.path.join(self.dataset_path, source)

            # Liste des classes disponibles pour l'échantillon concerné (1 dossier = 1 classe)
            classe = []
            for item in os.listdir(path_sample):
                classe.append(item)

            # Comptage du nombre d'images par classe pour l'échantillon concerné
            images_count = []
            for folder in classe:
                items = os.listdir(os.path.join(path_sample, folder))
                images_count.append(len(items))
            
            # Ajout des informations au dictionnaire
            temp_dict.update({source:{'classe': classe,
                            'img_nbr': images_count}})
        return temp_dict
    
    def calcul_metrics(self):
        copy_of_dict_count = self.count_image_of_sets
        for k in self.count_image_of_sets.keys():
            self.qtt.append(sum(copy_of_dict_count[k]['img_nbr']))
            self.mini.append(min(copy_of_dict_count[k]['img_nbr']))
            self.maxi.append(max(copy_of_dict_count[k]['img_nbr']))
            self.moy.append(sum(copy_of_dict_count[k]['img_nbr'])/
                            len(copy_of_dict_count[k]['img_nbr']))
            
    def create_df_train(self):
        test = pd.Series(self.count_image_of_sets['test']['img_nbr'], 
                         index=self.count_image_of_sets['test']['classe'], 
                         name='nb_img_test')
        train = pd.Series(self.count_image_of_sets['train']['img_nbr'], 
                          index=self.count_image_of_sets['train']['classe'], 
                          name='nb_img_train')
        valid = pd.Series(self.count_image_of_sets['valid']['img_nbr'], 
                          index=self.count_image_of_sets['valid']['classe'], 
                          name='nb_img_valid')
        df_glob = pd.concat([test, train, valid], axis=1)
        df_glob['nb_img_tot'] = df_glob.sum(axis = 1)
        df_glob.index.name = 'classe'

        # Calcul d'un top 0/1 pour non présence/présence de la classe dans l'échantillon
        df_glob['pres_test'] = np.where(df_glob['nb_img_test'].isna(), 0, 1)
        df_glob['pres_train'] = np.where(df_glob['nb_img_train'].isna(), 0, 1)
        df_glob['pres_valid'] = np.where(df_glob['nb_img_valid'].isna(), 0, 1)

        return pd.DataFrame(df_glob.loc[df_glob['pres_train']==1]['nb_img_train'])
        
    def aff_per(self, x):
        # Fonction pour affichage personnalisé pie chart
        return '{:.1f}%\n({:,.0f})'.format(x, self.total*x/100)

    def sample_distrib_plot(self):
        # Création des graphiques
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        coul = ['orange','green','yellow']

        # Pie chart
        ax1.pie(self.qtt, labels=self.samples,
                colors=coul,
                explode=[0, 0.3, 0.3],
                labeldistance=0.7,
                autopct=self.aff_per,
                pctdistance=1.2,
                wedgeprops={"edgecolor":"k",'linewidth': 1})
        ax1.legend(loc='best')
        titre = ax1.set_title("Poids des différents échantillons")
        titre.set(color="black", fontsize="16", fontfamily="serif")

        # Plot bar
        barWidth = 0.3 
        for x in range(len(self.samples)):
            axe = [i + barWidth * (x - 1) for i in range(len(self.samples))]
            ax2.bar(axe, [self.mini[x], self.moy[x], self.maxi[x]], 
                    color=coul[x], width=barWidth, 
                    edgecolor='black', label=self.samples[x])
            # Ajout des valeurs sur les bars
            for index, value in zip(axe, [self.mini[x], self.moy[x], self.maxi[x]]):
                plt.text(index - 0.05, value + 5,
                        str(int(value)))
        ax2.set_xticks(np.arange(0.0, 3, 1))
        ax2.set_xticklabels(['Minimum', 'Moyenne', 'Maximum'])
        # ax2.set_xticklabels(['','', 'Minimum', '', 'Moyenne', '', 'Maximum'])
        ax2.legend(loc='upper left', edgecolor='black')
        titre = ax2.set_title("Nombre d'images par échantillon")
        titre.set(color="black", fontsize="16", fontfamily="serif")

        # plt.show()
        return fig

    def nb_image_by_class_train_set(self):
        # on affiche une courbe représentant le nombre d'images par classe pour train
        fig = plt.figure(figsize = (25,10))
        df_train = self.df_train.sort_values(by = 'nb_img_train')
        plt.plot(df_train.index, df_train.nb_img_train, linewidth = 5, color = 'green')
        plt.title("Répartition du nombre d'images par classe du set train", fontsize = 22)
        plt.xlabel("Classes", fontsize = 22)
        plt.ylabel("Nombre d'images par classe", fontsize = 22)
        plt.xticks([])
        plt.yticks(fontsize = 18)
        return fig

    def read_rgb_json(self):
        with open(self.dataset_path + "_mean_rgb.json", "r") as f:
            data = json.load(f)
        self.mean_by_bird_r = data['mean_by_bird_r']
        self.mean_by_bird_g = data['mean_by_bird_g']
        self.mean_by_bird_b = data['mean_by_bird_b']

    def rgb_repartition(self):
        fig = plt.figure(figsize=(10, 6))
        plt.title("Distribution des valeurs moyennes RVB pour toutes les images")
        plt.xlabel("Valeurs RVB (0-255)")
        plt.ylabel("Répartition des valeurs RVB")
        plt.hist(self.mean_by_bird_r, bins=50, alpha=0.7, color='red', label='Rouge')
        plt.hist(self.mean_by_bird_g, bins=50, alpha=0.7, color='green', label='Vert')
        plt.hist(self.mean_by_bird_b, bins=50, alpha=0.7, color='blue', label='Bleu')
        plt.legend()
        plt.grid(True)
        return fig

    def rgb_repartition_distinct(self):
        fig = plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.hist(self.mean_by_bird_r, bins=255, alpha=0.7, color='red', label='Rouge')
        plt.title("Répartition des valeurs RVB moyennes (rouge)")
        plt.xlabel("Valeur de rouge (0-255)")
        plt.ylabel("Nombre d'images")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.hist(self.mean_by_bird_g, bins=255, alpha=0.7, color='green', label='Vert')
        plt.title("Répartition des valeurs RVB moyennes (vert)")
        plt.xlabel("Valeur de vert (0-255)")
        plt.ylabel("Nombre d'images")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.hist(self.mean_by_bird_b, bins=255, alpha=0.7, color='blue', label='Bleu')
        plt.title("Répartition des valeurs RVB moyennes (bleu)")
        plt.xlabel("Valeur de bleue (0-255)")
        plt.ylabel("Nombre d'images")
        plt.grid(True)
        plt.tight_layout()
        return fig

    def rvb_mean(self):
        all_bird_mean_r = np.mean(self.mean_by_bird_r)
        all_bird_mean_g = np.mean(self.mean_by_bird_g)
        all_bird_mean_b = np.mean(self.mean_by_bird_b)

        fig = plt.figure(figsize=(10, 6))
        plt.title("Moyennes de la répartition des couleurs RVB")
        plt.ylabel("Valeurs RVB moyennes")
        plt.bar(['Rouge', 'Vert', 'Bleu'], [all_bird_mean_r, all_bird_mean_g, all_bird_mean_b], color=['red', 'green', 'blue'])
        plt.grid(True)
        plt.ylim(0,255)
        return fig

    def box_plot(self):
        fig = plt.figure(figsize=(10, 6))
        plt.title("Répartition des couleurs RVB pour chaque espèce d'oiseau")
        plt.ylabel("Valeurs RVB")
        plt.boxplot([self.mean_by_bird_r, self.mean_by_bird_g, self.mean_by_bird_b], labels=['Rouge', 'Vert', 'Bleu'])
        plt.grid(True)
        return fig


if __name__ == "__main__":
    displayer = StatsDatasetDisplayer("data\\dataset_birds_original")
    displayer.sample_distrib_plot()
    
    