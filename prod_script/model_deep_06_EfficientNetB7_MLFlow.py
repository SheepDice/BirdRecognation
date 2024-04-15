from mlflow import MlflowClient
import mlflow
import argparse
import os
import const_prod

# installer tensorflow 2.10.0 et exactement cette version pour avoir le support du GPU

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

# les chemins vers le dataset sur lequel s'entraîner

path_train = os.path.join(const_prod.HALF_DATASET_CLEAN_WO_BACKGROUND_PART2, 'train')
path_valid = os.path.join(const_prod.HALF_DATASET_CLEAN_WO_BACKGROUND_PART2, 'valid')

# on se connecte au serveur de tracking MLFlow
# normalement, si il n'est pas lancé, cela fonctionnera tout de même mais sans suivi

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

birds_experiment = mlflow.set_experiment("Birds_Experiment")
artifact_path = "efficientnet_v2_artifacts"
mlflow.tensorflow.autolog()


# les arguments que l'on peut passer si nécessaire au lancement du script

parser = argparse.ArgumentParser()
parser.add_argument('--epochs_1', type = int, default = 15)
parser.add_argument('--epochs_2', type = int, default = 15)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--learning_rate', type = float, default = 1e-4)
args = parser.parse_args()

epochs_1 = args.epochs_1
epochs_2 = args.epochs_2
batch_size = args.batch_size
learning_rate = args.learning_rate

# on créer les générateurs et on augmente les données
# à priori celui de test ne sert à rien ici, je le retirerai sûrement

train_data_generator = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True
    )

validation_data_generator = ImageDataGenerator(
    preprocessing_function = preprocess_input
)

# on s'assure aussi que les images soient bien toutes en 224x224

train_generator = train_data_generator.flow_from_directory(directory=path_train,
                                                           class_mode ="sparse",
                                                          target_size = (224 , 224), 
                                                          batch_size = batch_size)

validation_generator = validation_data_generator.flow_from_directory(directory = path_valid,  
                                                        class_mode = "sparse",
                                                        target_size = (224, 224),
                                                        batch_size = batch_size)

# On définit le nombre de classes du dataset chargé

n_class = train_generator.num_classes

# on se base sur le modèle EfficientNetB7

base_model = EfficientNetB7(weights='imagenet', include_top=False) 

# on commence l'entraînement en ne touchant pas aux poids du modèle de base

for layer in base_model.layers: 
    layer.trainable = False

# on créer notre modèle et on le compile

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D()) 
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_class, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# on arrête l'entraînement si il n'y a pas d'évolution de validation_accuracy
# il y a deux early stopping car étonnament la métrique n'a pas le même nom avant et après le defreeze

early_stopping_1 = EarlyStopping(
                                patience=  2, 
                                min_delta = 0.005, 
                                verbose = 1, 
                                mode = 'max',
                                monitor = 'val_acc')

early_stopping_2 = EarlyStopping(
                                patience=  2, 
                                min_delta = 0.005, 
                                verbose = 1, 
                                mode = 'max',
                                monitor = 'val_accuracy')

# on compte le nombre d'images dans train, test et valid

nb_img_train = train_generator.samples
nb_img_valid = validation_generator.samples

# on lance l'entraînement et on le track avec MLFlow

with mlflow.start_run(run_name = "efficientent_b7_freezed") as run:
    history = model.fit_generator(train_generator, 
                                    epochs = epochs_1,
                                    steps_per_epoch = nb_img_train//batch_size,
                                    validation_data = validation_generator,
                                    validation_steps = nb_img_valid//batch_size,
                                    callbacks = [early_stopping_1],
                                    verbose = True
                                    )
    
    # on sauvegarde LES POIDS (pas le modèle entier, cela ne marche pas ici)

    model_path = "models/model_freezed_5.h5"
    model.save_weights(model_path)
    # on indique à MLFlow de suivre où se situe ce fichier
    mlflow.log_artifact(model_path, "model_freezed")

# on rend entrainable les 20 dernières couches d'EfficientNetB7
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run(run_name = "efficientent_b7_final") as run:
    history = model.fit_generator(generator=train_generator, 
                                    epochs = epochs_2,
                                    steps_per_epoch = nb_img_train//batch_size,
                                    validation_data = validation_generator,
                                    validation_steps = nb_img_valid//batch_size,
                                    callbacks = [early_stopping_2],
                                    verbose = True
                                    )
    
    # on sauvegarde LES POIDS (pas le modèle entier, cela ne marche pas ici)
    model_path = "models/model_final_5.h5"
    model.save_weights(model_path)
    # on indique à MLFlow de suivre où se situe ce fichier
    mlflow.log_artifact(model_path, "model_final")