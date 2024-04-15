from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
# import const_prod
import os
# from tkinter import filedialog
# from ModelBuilder import ModelBuilder
from prod_script.ModelBuilder import ModelBuilder

class GradCam:
    def __init__(self, dataset, last_conv_layer_name, size = (224,224), B7 = False):
        self.size = size
        self.image_array = None
        self.model = ModelBuilder.load_B0_523_DataAugment_Equilibre_Categorical(dataset)
        self.last_conv_layer_name = last_conv_layer_name

    def get_img_array(self, image_path):
        """
        Charge une image et retourne un tableau numpy qui la représente
        """
        img = image.load_img(image_path, target_size=self.size)
        array = image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        self.image_array =  array

    def make_gradcam_heatmap(self, image_path, pred_index=None):
        """
        Prend un un model et calcul la matrice représentant l'importance de chaque pixel
        pour la prédiction
        Retourne un tableau numpy représentant sa heatmap
        """
        self.get_img_array(image_path)
        grad_model = Model(
            [self.model.inputs], [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(self.image_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def display_gradcam(self, image_path, alpha=0.4):
        """
        Affiche l'image avec sa heatmap
        """
        heatmap = self.make_gradcam_heatmap(image_path)
        img = image.load_img(image_path, target_size=self.size, color_mode="grayscale")
        img = image.img_to_array(img)
    
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
    
        jet_heatmap = image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = image.img_to_array(jet_heatmap)
    
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = image.array_to_img(superimposed_img)
        fig = plt.figure()

        plt.imshow(superimposed_img)
        plt.show()
        return fig

# if __name__ == "__main__":
#     const_prod.config_prod.set_root_dir()
#     img_path = filedialog.askopenfilename()
#     # img_path = \
#     # "D:\Code\Datascience\\reco_oiseau_jan24bds\data\\dataset_birds_wo_background\\test\\NORTHERN FLICKER\\86.jpg"

#     gc = GradCam(const_prod.DATASET_CLEAN_WO_BACKGROUND_PATH, "top_conv")
#     gc.display_gradcam(img_path)