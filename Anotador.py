import argparse
import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split


# random_state=42
# random_state=15
# random_state=67

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


def anotador(
        source_file,
        corpus_dir,
        output_path,
        model_type,
        do_train: bool = False,
        do_test: bool = False
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

    file_test = source_file + '/test.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test = pd.DataFrame(csv3, columns=['tipo', 'text'])

    df_total = df_test

    if 'neoplasm' in source_file:
        file_validate = source_file + '/validate.csv'
        csv = pd.read_csv(file_validate, index_col=False)
        df_val = pd.DataFrame(csv, columns=['tipo', 'text'])

        file_train = source_file + '/train.csv'
        csv2 = pd.read_csv(file_train, index_col=False)
        df_train = pd.DataFrame(csv2, columns=['tipo', 'text'])

        df_total = pd.concat([df_val, df_train, df_test])


    df_total['tipo'] = df_total['tipo'].map(label_to_id)

    train_texts, val_texts, train_labels, val_labels = train_test_split(df_total['text'], df_total['tipo'], test_size=0.2,
                                                                        random_state=67)

    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)

    val_dataset = AnotatorDataset(val_encodings, list(val_labels))
    train_dataset = AnotatorDataset(train_encodings, list(train_labels))
    #
    # Definir los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
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

        # Guardar el modelo y el tokenizador entrenado
        model.save_pretrained('./model_anotator')
        tokenizer.save_pretrained('./tokenizer_anotator')

    if do_test:
        claims = 0
        mclaims = 0
        other = 0
        premisas = 0
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

            softmax = torch.nn.Softmax(dim=1)
            probabilities = softmax(torch.tensor(predictions.predictions))
            threshold = 0.9

            # Obtener las etiquetas predichas
            predicted_labels = torch.argmax(probabilities, dim=1)
            predicted_confidences = torch.max(probabilities, dim=1).values

            # Asignar labels segun el umbral de confianza
            predicted_labels_text = []
            for label, confidence in zip(predicted_labels, predicted_confidences):
                if confidence < threshold:
                    predicted_labels_text.append("Other")
                else:
                    predicted_labels_text.append(id_to_label[label.item()])

            premisas = premisas + predicted_labels_text.count('Premise')
            claims = claims + predicted_labels_text.count('Claim')
            mclaims = mclaims + predicted_labels_text.count('MajorClaim')
            other = other + predicted_labels_text.count('Other')

            with open(output_path + f_name + '.ann', 'a', encoding='utf-8') as fout:
                for i, label in enumerate(predicted_labels_text):
                    if label != 'Other':
                        fout.write('%s' % ('T' + str(i + 1)))
                        fout.write('\t')
                        fout.write('%s' % label)
                        fout.write('\t')
                        fout.write('%s' % corpus[i])
                        fout.write('\n')
        print('Premisas:' + str(premisas))
        print('Claims:' + str(claims))
        print('MajorClaims:' + str(mclaims))
        print('Other:' + str(other))


def anotador_relaciones(
        source_file,
        corpus_dir,
        output_path,
        model_type,
        do_train: bool = False,
        do_test: bool = False
):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if model_type == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    elif model_type == 'saved':
        tokenizer = BertTokenizer.from_pretrained('./tokenizer_relation')
        model = BertForSequenceClassification.from_pretrained('./model_relation', num_labels=3)
    else:
        raise Exception('Tipo de model no valido debe ser bert-base-uncased o saved')

    # Mapeo de etiquetas de texto a numéricas
    label_to_id = {"Support": 0, "Attack": 1, "Partial-Attack": 2}
    id_to_label = {0: "Support", 1: "Attack", 2: "Partial-Attack"}

    file_test = source_file + '/test_relation.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test = pd.DataFrame(csv3, columns=['tipo', 'text1', 'text2'])

    df_total = df_test

    if 'neoplasm' in source_file:
        file_validate = source_file + '/validate_relation.csv'
        csv = pd.read_csv(file_validate, index_col=False)
        df_val = pd.DataFrame(csv, columns=['tipo', 'text1', 'text2'])

        file_train = source_file + '/train_relation.csv'
        csv2 = pd.read_csv(file_train, index_col=False)
        df_train = pd.DataFrame(csv2, columns=['tipo', 'text1', 'text2'])

        df_total = pd.concat([df_val, df_train, df_test])

    df_total['tipo'] = df_total['tipo'].map(label_to_id)

    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(df_total['text1'], df_total['text2'], df_total['tipo'],
                                                                        test_size=0.2,
                                                                        random_state=42)

    val_encodings = tokenizer(list(val_texts1), list(val_texts2), truncation=True, padding=True, max_length=512)
    train_encodings = tokenizer(list(train_texts1), list(train_texts2), truncation=True, padding=True, max_length=512)

    val_dataset = AnotatorDataset(val_encodings, list(val_labels))
    train_dataset = AnotatorDataset(train_encodings, list(train_labels))

    # Definir los argumentos de entrenamiento
    relation_training_args = TrainingArguments(
        output_dir='./results_relation',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_relation',
        logging_steps=10,
    )

    # Inicializar el Trainer
    relation_trainer = Trainer(
        model=model,
        args=relation_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if do_train:
        # Entrenar el modelo
        relation_trainer.train()

        model.save_pretrained('./model_relation')
        tokenizer.save_pretrained('./tokenizer_relation')

    if do_test:
        # for f in tqdm(os.listdir(corpus_dir), total=len(os.listdir(corpus_dir))):
        #     if '.txt' in f:
        #         f_name = f[:-4]

        corpus, id_sentence = get_corpus_with_anotations(corpus_dir, "f_name")

        # Crear pares de frases
        pares = []
        for i in range(len(corpus)):
            for j in range(i + 1, len(corpus)):
                pares.append((corpus[i], corpus[j]))

        pair_texts1, pair_texts2 = zip(*pares)
        pair_encodings = tokenizer(list(pair_texts1), list(pair_texts2), truncation=True, padding=True,
                                          max_length=128)

        relation_inference_dataset = AnotatorDataset(pair_encodings, [0]*len(pares))

        # Realizar la inferencia
        predictions = relation_trainer.predict(relation_inference_dataset)

        softmax = torch.nn.Softmax(dim=1)
        relation_probabilities = softmax(torch.tensor(predictions.predictions))
        threshold = 0.7

        # Obtener las etiquetas predichas
        relation_predicted_labels = torch.argmax(relation_probabilities, dim=1)
        relation_predicted_confidences = torch.max(relation_probabilities, dim=1).values

        # Asignar etiquetas con umbral de confianza
        relation_predicted_labels_text = []
        for label, confidence in zip(relation_predicted_labels, relation_predicted_confidences):
            if confidence < threshold:
                relation_predicted_labels_text.append("Other")
            else:
                relation_predicted_labels_text.append(id_to_label[label.item()])

        # Imprimir las relaciones predichas para cada par de oraciones del nuevo corpus
        for i, (label, pair) in enumerate(zip(relation_predicted_labels_text, pares)):
            # print(f"Par de oraciones {i + 1}: {pair[0]} <--> {pair[1]}: Relación predicha = {label}")
            print(f"R{i + 1}\t{label}\tArg1:{id_sentence[corpus.index(pair[0])]}\tArg2:{id_sentence[corpus.index(pair[1])]}")

            # with open(output_path + '0211-699501013956' + '.ann', 'a', encoding='utf-8') as fout:
            #     for i, (label, pair) in enumerate(zip(relation_predicted_labels_text, pares)):
            #         fout.write('%s' % ('R' + str(i + 1)))
            #         fout.write('\t')
            #         fout.write('%s' % label)
            #         fout.write('\t')
            #         fout.write('%s' % corpus[i])
            #         fout.write('\t')
            #         fout.write('%s' % corpus[i])
            #         corpus.index()
            #         fout.write('\n')


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


def get_corpus_with_anotations(corpus_dir, file_name):
    textos = []
    id_sentence = []
    with open('./data_anotated/dev/abstracts/0211-699501013956.ann', 'r', encoding='utf-8') as fann:
    # with open(corpus_dir + file_name + '.ann', 'r', encoding='utf-8') as fann:
        lines = fann.readlines()
        for line in lines:
            line = line.split('\t')
            id_sentence.append(line[0])
            textos.append(line[2].replace('\n', ''))
        return textos, id_sentence




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
        source_file='./data_translated/mixed',
        corpus_dir='./CorpusSinAnotaciones/dev/abstracts/',
        output_path='data_anotated/dev/abstracts/',
        model_type='saved',
        do_test=True
    )
    # anotador_relaciones(
    #     source_file='./data_translated/neoplasm',
    #     corpus_dir='./CorpusSinAnotaciones/dev/abstracts/',
    #     output_path='./data_anotated/dev/abstracts/',
    #     model_type='saved',
    #     do_test=True
    # )