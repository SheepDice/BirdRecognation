import os
import shutil
import const_prod
from PIL import Image, ImageEnhance, ImageOps
import random

import matplotlib.pyplot as plt

class OverSamplerImages:
    def __init__(self, root_dir):
        self.root_dir = root_dir #chemin du set train
        self.max_size = self.get_max_size() #nombre maximum d'image de train pour une classe
        
    def get_max_size(self): #On cherche la classe aec le plus d'image pour aligner les autres dessus
        classes = os.listdir(self.root_dir)
        max_size_class = 0
        for classe in classes:
            size_class = len(os.listdir(os.path.join(self.root_dir, classe)))
            if size_class > max_size_class:
                max_size_class = size_class
        return max_size_class
    
    def random_rot(self, img): #rotation de l'image aléatoire entre -90 et 90 degrés
        rot = random.randint(-90, 90) 
        img = img.rotate(rot, expand=True)
        img = img.resize((224,224))
        return img

    def random_flip(self, img): #flip horizontal éventuel
        if random.randint(0,1) == 1: #flip a coin
            img = ImageOps.mirror(img)
        return img

    def random_luminosity(self, img): #Changement aléatoire de la luminosité
        lum_factor = random.randint(1, 30) * 0.1 #lum_factore entre 0.1 et 3
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(lum_factor)
        return img

    def random_contrast(self, img): #Changement aléatoire de la valeur de contraste
        contrast_factor = random.randint(1, 30) * 0.1 #contrast_factor entre 0.1 et 5
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        return img
        
    def duplicate_random(self, classe_path, new_images): #duplique une image existante dans un dossier new
        files = os.listdir(classe_path) #La liste des images existantes
        random_file = "new"
        while random_file == "new":
            random_file = files[random.randint(0, len(files)-1)] #On en choisi une au hasard
        random_file_path = os.path.join(classe_path, random_file)
        new_file_name = random_file
        i = 0
        all_files = os.listdir(classe_path) + os.listdir(new_images)
        while new_file_name in all_files: #On cherche un nouveau nom qui n'existe pas déjà pour la nouvelle  image
            new_file_name = new_file_name.split('.')[0] + str(i) + ".jpg"
            new_file_name_path = os.path.join(new_images, new_file_name)
            i +=1
        shutil.copy(random_file_path, new_file_name_path) #On duplique l'image dans le dossier new
        return new_file_name_path

    def generate_random_image(self, classe_path, new_images):
        new_image_path = self.duplicate_random(classe_path, new_images) #Duplication d'une image aléatoire
        #Ouverture et modifications aléatoires sur l'image
        img = Image.open(new_image_path)
        img = self.random_rot(img)
        img = self.random_flip(img)
        img = self.random_contrast(img)
        img = self.random_luminosity(img)
        img.save(new_image_path) #Sauvegarde de l'image

    def over_sample(self):
        for classe in os.listdir(self.root_dir):
            classe_path = os.path.join(self.root_dir, classe)
            new_images_dir = os.path.join(classe_path, "new")
            if not os.path.isdir(new_images_dir):
                #Creation d'un dossier new pour mettre les images dupliquée, afin de ne pas risquer de redupliquer
                #une image déjà dupliquée et modifiée
                os.mkdir(new_images_dir)
            while len(os.listdir(classe_path)) + len(os.listdir(new_images_dir)) < self.max_size + 1: #+1 car il y a le "\new"
                #tant qu'il y a moins d'image que pour la classe majoritaire, on créé des images
                self.generate_random_image(classe_path, new_images_dir)
            for img in os.listdir(new_images_dir):
                #Une fois toute les images créées, on les sort du dossier new
                os.rename(os.path.join(new_images_dir, img), os.path.join(classe_path, img))
            if os.path.isdir(new_images_dir):
                #On supprime le dossier new qui est vide
                shutil.rmtree(new_images_dir)

    def check_distribution(self):
        #Affiche le nombre d'image par classe
        print("Images pour la classe majoritaire : ", self.max_size)
        classes = os.listdir(self.root_dir)
        nb_image_by_classe = list()
        for classe in classes:
            size_classe = len(os.listdir(os.path.join(self.root_dir, classe)))
            if size_classe != self.max_size:
                nb_image_by_classe.append(size_classe)
        print("Il y a ", len(nb_image_by_classe), " classe(s) à augmenter")
        print("Il faut créer ", sum(nb_image_by_classe), " image(s)")

if __name__ == "__main__":
    root_dir = os.path.join(const_prod.DATASET_TEST_PATH, "train")
    overSampler = OverSamplerImages(root_dir)
    overSampler.check_distribution()
    overSampler.over_sample()
    overSampler.check_distribution()


