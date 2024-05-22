import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch





def anotador():

    tokenizer = BertTokenizer.from_pretrained('./tokenizer_anotator')
    model = BertForSequenceClassification.from_pretrained('./model_anotator', num_labels=3)

    # Mapeo de etiquetas de texto a numéricas
    label_to_id = {"Premise": 0, "Claim": 1, "MajorClaim": 2}
    id_to_label = {0: "Premise", 1: "Claim", 2: "MajorClaim"}

    file_validate = './data_translated/validate.csv'
    csv = pd.read_csv(file_validate, index_col=False)
    df_val = pd.DataFrame(csv, columns=['tipo', 'text'])

    file_train = './data_translated/train.csv'
    csv2 = pd.read_csv(file_train, index_col=False)
    df_train = pd.DataFrame(csv2, columns=['tipo', 'text'])

    val_text = df_val['text']
    val_labels = df_val['tipo'].map(label_to_id)

    train_text = df_train['text']
    train_labels = df_train['tipo'].map(label_to_id)

    val_encodings = tokenizer(list(val_text), truncation=True, padding=True, max_length=512)
    train_encodings = tokenizer(list(train_text), truncation=True, padding=True, max_length=512)

    class AnotatorDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    val_dataset = AnotatorDataset(val_encodings, list(val_labels))
    train_dataset = AnotatorDataset(train_encodings, list(train_labels))
    #
    # Definir los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir='./results',  # Directorio para los resultados
        num_train_epochs=3,  # Número de épocas de entrenamiento
        per_device_train_batch_size=8,  # Tamaño del batch por dispositivo durante el entrenamiento
        per_device_eval_batch_size=8,  # Tamaño del batch por dispositivo durante la evaluación
        warmup_steps=500,  # Número de pasos de calentamiento para el optimizador
        weight_decay=0.01,  # Tasa de decaimiento de peso
        logging_dir='./logs',  # Directorio para los logs
        logging_steps=10,
    )

    # Inicializar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # # Entrenar el modelo
    # trainer.train()

    # Guardar el modelo y el tokenizador
    model.save_pretrained('./model_anotator')
    tokenizer.save_pretrained('./tokenizer_anotator')

    corpus = get_corpus()

    # Tokenizar el nuevo corpus
    new_encodings = tokenizer(corpus, truncation=True, padding=True, max_length=128)

    # Crear el dataset para el nuevo corpus
    new_dataset = AnotatorDataset(new_encodings, [0] * len(corpus))

    # Realizar la inferencia
    predictions = trainer.predict(new_dataset)

    # Obtener las etiquetas predichas
    predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)

    # Convertir etiquetas numéricas a texto
    predicted_labels_text = [id_to_label[label.item()] for label in predicted_labels]

    # Imprimir las etiquetas predichas para cada oración del nuevo corpus
    for i, label in enumerate(predicted_labels_text):
        print(f"Oración {i + 1}: Etiqueta predicha = {label}")






def get_corpus():
    textos = []
    with open('./CorpusSinAnotaciones/dev/abstracts/0211-699501013956.txt', 'r', encoding='utf-8') as fann:
        lines = fann.readlines()
        for i in range(lines.count('\n')):
            lines.remove('\n')
        for line in lines:
            line = line.replace('\n', '')
            textos.append(line)
        return textos







if __name__ == '__main__':

    anotador()