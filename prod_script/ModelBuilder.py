import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobile
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_efficient
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import Model
from tkinter.filedialog import askopenfilename


class ModelBuilder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def import_model(dataset_path, model = "EfficientNetB0", sparse = False):

        if "test" not in dataset_path:

            dataset_test_path = os.path.join(dataset_path, 'test')
            
        if model == "MobileNetV2":
            # on créer le générateur pour test
            test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input_mobile)
        elif model == "FromScratch":
            # on créer le générateur pour test
            test_data_generator = ImageDataGenerator(rescale=1./255)
        else:
            # on créer le générateur pour test
            test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input_efficient)

        
        if sparse == False:

            # on charge les images et on les redimensionne
            test_generator = test_data_generator.flow_from_directory(
                directory=dataset_test_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )

        else:
            # on charge les images et on les redimensionne
            test_generator = test_data_generator.flow_from_directory(
                directory=dataset_test_path,
                target_size=(224, 224),
                batch_size=32,
                class_mode='sparse',
                shuffle=False
            )

        
        if model != "FromScratch":

            # on définit le nombre de classes du dataset chargé
            n_class = test_generator.num_classes

            # on définit l'architecture du modèle utilisé pour l'entrainement
            if model == "EfficientNetB7":
                base_model = EfficientNetB7(weights='imagenet', include_top=False)
            elif model == "EfficientNetB0":
                base_model = EfficientNetB0(weights='imagenet', include_top=False)
            elif model == "MobileNetV2":
                base_model = MobileNetV2(weights='imagenet', include_top=False)

            for layer in base_model.layers:
                layer.trainable = False

            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(rate=0.2)(x)
            predictions = Dense(n_class, activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            # enlever ces lignes pour tester sur les modèles "freezed"

            if model == "EfficientNetB7":
                for layer in base_model.layers[-16:]:
                    layer.trainable = True
            elif model == "EfficientNetB0":
                for layer in base_model.layers[-10:]:
                    layer.trainable = True
            elif model == "MobileNetV2":
                for layer in base_model.layers[-5:]:
                    layer.trainable = True

        
        return test_generator, model
        

    def load_B0_523_DataAugment_Equilibre_Categorical(dataset_path, 
                                                      model_name="EfficientNetB0_523_DataAugment_Equilibre_Categorical.h5"):
        test_generator, model = ModelBuilder.import_model(dataset_path, model = "EfficientNetB0", sparse = False)
        # test_generator, model = ModelBuilder.import_model(dataset_test_path, model = "EfficientNetB0", sparse = False, use_test_generator = True)
        try:
            model.load_weights(model_name)
        except:
            model_path = askopenfilename()
            model.load_weights(model_path)
        return model