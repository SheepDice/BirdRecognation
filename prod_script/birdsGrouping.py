import os
import const_prod
import csv
import pandas as pd
from tqdm import tqdm
import shutil


class birdsGrouping:
    def __init__(self, db_to_sort):
        self.db_to_sort = db_to_sort
        self.birdList = list()
        self.df = self.read_csv()
        self.nb_order = self.df['order'].nunique()

    def read_csv(self):
        """
        Lit l'ensemble du csv en supprimant les espaces en trop
        Initialise les valeurs liées au dataframe
        """
        with open(os.path.join(const_prod.DATA_PATH, "taxo.csv"), 'r', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            cols = ['init_name', 'avi_name', 'order', 'family']
            for row in reader:
                bird = dict()
                for i in range(len(row)):
                    bird[cols[i]] = ' '.join(row[i].split())
                self.birdList.append(bird)
        del(self.birdList[0])
        return pd.DataFrame(self.birdList)
    
    def sort_set_by_order_or_family(self, set, level_of_taxo):
        """
        Parcourt l'ensemble d'un set (test, train ou valid) et rassemble toutes les images d'un(e) même
        ordre/famille dans le même répertoire. 
        Les fichiers prennent le nom de leur oiseau suivi de _ puis leur nom d'origine (nombre.jpg)
        Supprime les répertoires vides
        """
        for classe in tqdm(os.listdir(set), "Parcours des classes pour le set " + set):
            old_path = os.path.join(set, classe)
            if classe not in self.df['init_name'].values:
                continue
            new_path = os.path.join(set, self.df[self.df['init_name'] == classe ][level_of_taxo].values[0])
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            for file in os.listdir(old_path):
                old_file_path = os.path.join(old_path, file)
                new_file_path = os.path.join(new_path, classe + "_" + file)
                os.rename(old_file_path, new_file_path)
        self.clean_empty_dir(set)

    def sort_set_by_species(self, set):
        """
        Parcourt l'ensemble d'un set (test, train ou valid) et redistribue les images de chaque ordre/famille
        par nom d'espèce
        Les fichiers reprennent le nom d'origine (nombre.jpg)
        Supprime les répertoires vides"""
        for dir in tqdm(os.listdir(set), "Parcours des dossiers pour le set " + set):
            old_path = os.path.join(set, dir)
            for file in os.listdir(old_path):
                birdName = file.split("_")[0]
                bird_path = os.path.join(set, birdName)
                if birdName not in self.df['init_name'].values:
                    continue
                old_file_path = os.path.join(old_path, file)
                new_file_path = os.path.join(bird_path, file.split('_')[1])
                if not os.path.isdir(bird_path):
                    os.mkdir(bird_path)
                os.rename(old_file_path, new_file_path)
        self.clean_empty_dir(set)

    def clean_empty_dir(self, set):
        """
        Supprime les répertoires vides
        """
        for dir in os.listdir(set):
            current_dir = os.path.join(set, dir)
            if os.listdir(current_dir) == []:
                shutil.rmtree(current_dir)

    def sort_by_species(self):
        """
        Applique le tri par espèce sur les trois sets
        """
        for set in ['test', 'train', 'valid']:
            set_path = os.path.join(self.db_to_sort, set)
            self.sort_set_by_species(set_path)

    def sort_by_order_or_family(self, level_of_taxo):
        """
        Applique le tri par ordre ou famille sur les trois sets
        """
        for set in ['test', 'train', 'valid']:
            set_path = os.path.join(self.db_to_sort, set)
            self.sort_set_by_order_or_family(set_path, level_of_taxo)

    def check_number_of_images(self):
        count = 0
        for set in os.listdir(self.db_to_sort):
            path1 = os.path.join(self.db_to_sort, set)
            for bird in os.listdir(path1):
                path2 = os.path.join(path1, bird)
                count += len(os.listdir(path2))
        print(count)

if __name__ == "__main__":
    const_prod.config_prod.set_root_dir()

    bG = birdsGrouping(const_prod.DATASET_TEST_PATH)
    bG.read_csv()
    bG.check_number_of_images()
    bG.sort_by_order_or_family("family") 
    #Indiquer "order" si on veut grouper les espèces par ordre ou "family" si on veut grouper par famille
    #Toujours passer par le groupement par espèce ci-dessous avant de faire un groupement par ordre/famille
    #On ne peut pas faire order => family ou family => order
    # bG.sort_by_species()
    bG.check_number_of_images()