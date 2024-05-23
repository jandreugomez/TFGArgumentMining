import argparse
import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch





def anotador(
        source_file,
        corpus_dir,
        output_path,
        model_type,
        do_train: bool = False
):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if model_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    elif model_type == 'saved':
        tokenizer = BertTokenizer.from_pretrained('./tokenizer_anotator')
        model = BertForSequenceClassification.from_pretrained('./model_anotator', num_labels=3)
    else:
        raise Exception('Tipo de model no valido debe ser bert-base-uncased o saved')

    # Mapeo de etiquetas de texto a numéricas
    label_to_id = {"Premise": 0, "Claim": 1, "MajorClaim": 2}
    id_to_label = {0: "Premise", 1: "Claim", 2: "MajorClaim"}

    file_validate = source_file + '/validate.csv'
    csv = pd.read_csv(file_validate, index_col=False)
    df_val = pd.DataFrame(csv, columns=['tipo', 'text'])

    file_train = source_file + '/train.csv'
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

    if do_train:
        # Entrenar el modelo
        trainer.train()

    # Guardar el modelo y el tokenizador
    model.save_pretrained('./model_anotator')
    tokenizer.save_pretrained('./tokenizer_anotator')

    for f in tqdm(os.listdir(corpus_dir), total=len(os.listdir(corpus_dir))):
        if '.txt' in f:
            f_name = f[:-4]

        corpus = get_corpus(corpus_dir, f_name)

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

        with open(output_path + f_name + '.ann', 'a', encoding='utf-8') as fout:
            for i, label in enumerate(predicted_labels_text):
                fout.write('%s' % ('T' + str(i + 1)))
                fout.write('\t')
                fout.write('%s' % label)
                fout.write('\t')
                fout.write('%s' % corpus[i])
                fout.write('\n')






def get_corpus(corpus_dir, file_name):
    textos = []
    # with open('./CorpusSinAnotaciones/dev/abstracts/0211-699501013956.txt', 'r', encoding='utf-8') as fann:
    with open(corpus_dir + file_name + '.txt', 'r', encoding='utf-8') as fann:
        lines = fann.readlines()
        for i in range(lines.count('\n')):
            lines.remove('\n')
        for line in lines:
            line = line.replace('\n', '')
            textos.append(line)
        return textos







if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     required=True,
    #     help="Path al .csv que se usa para inferir anotaciones",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     required=True,
    #     help="Path donde se va a guardar el nuevo corpus anotado",
    # )
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     required=True,
    #     help='model: bert-base-uncased or saved'
    # )
    # parser.add_argument(
    #     '--do_train',
    #     action="store_true",
    #     help='Indica si realizar el entrenamiento. Se usa en caso de que el modelo sea el guardado'
    # )
    #
    # args = parser.parse_args()
    # anotador(
    #     source_file=args.data_path,
    #     output_path=args.output_dir,
    #     model_type=args.model
    # )
    anotador(
        source_file='./data_translated',
        corpus_dir='./CorpusSinAnotaciones/dev/abstracts/',
        output_path='./data_anotated/dev/abstracts/',
        model_type='saved'
    )