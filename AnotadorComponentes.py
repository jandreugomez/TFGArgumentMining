import argparse
import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import nltk
nltk.download('punkt')
import re

from nltk.tokenize import sent_tokenize


# seed=42
# seed=15
# seed=67
# seed=54
# seed=99

# Mapeo de etiquetas de texto a numéricas
label_to_id = {"Premise": 0, "Claim": 1, "MajorClaim": 2, "O": 3}
id_to_label = {0: "Premise", 1: "Claim", 2: "MajorClaim", 3: "O"}



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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
    accuracy = accuracy_score(labels, preds)

    # Calcular métricas promedio ponderadas también
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, preds,
                                                                                          average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    for i, label in enumerate(label_to_id.keys()):
        metrics[f'precision_{label}'] = precision[i]
        metrics[f'recall_{label}'] = recall[i]
        metrics[f'f1_{label}'] = f1[i]

    return metrics

def train_component_anotator(trainer, save_name):

    # Entrenar el modelo
    trainer.train()

    if args.save_model:
        # Guardar el modelo y el tokenizador entrenado
        model.save_pretrained('./model_anotator_'+save_name)
        tokenizer.save_pretrained('./tokenizer_anotator_'+save_name)

    if args.do_eval:
        eval_results = trainer.evaluate()
        print(eval_results)



def inference_component_anotator(trainer):

    claims = 0
    mclaims = 0
    other = 0
    premisas = 0
    for f in tqdm(os.listdir(args.corpus_dir), total=len(os.listdir(args.corpus_dir))):
        if '.txt' in f:
            f_name = f[:-4]

        corpus = get_corpus(args.corpus_dir, f_name)

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

        premisas = premisas + predicted_labels_text.count('Premise')
        claims = claims + predicted_labels_text.count('Claim')
        mclaims = mclaims + predicted_labels_text.count('MajorClaim')
        other = other + predicted_labels_text.count('O')

        with open(args.output_path + f_name + '.ann', 'a', encoding='utf-8') as fout:
            for i, label in enumerate(predicted_labels_text):
                if label != 'O':
                    fout.write('%s' % ('T' + str(i + 1)))
                    fout.write('\t')
                    fout.write('%s' % label)
                    fout.write('\t')
                    fout.write('%s' % corpus[i])
                    fout.write('\n')
    print('Premisas:' + str(premisas))
    print('Claims:' + str(claims))
    print('MajorClaims:' + str(mclaims))
    print('O:' + str(other))



def get_corpus(corpus_dir, file_name):
    with open(corpus_dir + file_name + '.txt', 'r', encoding='utf-8') as ftxt:
        texto_preprocesado = re.sub(r'(?<=[a-zA-Z])\n', '. ', ftxt.read())
        texto_preprocesado = re.sub(r'\n', ' ', texto_preprocesado)
        frases = sent_tokenize(texto_preprocesado)

        return frases


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_file",
        type=str,
        required=True,
        help="Path al .csv que se usa para el entrenamiento.",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path donde se coge el corpus sin anotar.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path donde se va a guardar el nuevo corpus anotado.",
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        help='model: dccuchile/bert-base-spanish-wwm-uncased or bert-base-multilingual-uncased'
    )
    parser.add_argument(
        '--do_train',
        action="store_true",
        help='Indica si se realiza el entrenamiento.'
    )
    parser.add_argument(
        '--do_eval',
        action="store_true",
        help='Indica si se realiza la evaluacion del entrenamiento.'
    )
    parser.add_argument(
        '--do_test',
        action="store_true",
        help='Indica si se realiza la inferencia de anotaciones'
    )
    parser.add_argument(
        '--save_model',
        action="store_true",
        help='Si se usa el modelo guardado'
    )
    parser.add_argument(
        '--use_saved_model',
        action="store_true",
        help='Si se usa el modelo guardado'
    )
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Numero total de epoch de entrenamiento.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="El learning rate para Adam.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--seed", default=42, type=int, help="Indica la semilla para la reproducibilidad del mezclado de datos de entrenamiento."
    )

    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.use_saved_model:
        if args.model_type == 'dccuchile/bert-base-spanish-wwm-uncased':
            save_name = 'BETO'
        if args.model_type == 'bert-base-multilingual-uncased':
            save_name = 'bert-base-multilingual-uncased'
        tokenizer = BertTokenizer.from_pretrained('./tokenizer_anotator_' + save_name)
        model = BertForSequenceClassification.from_pretrained('./model_anotator_' + save_name, num_labels=4)
    elif args.model_type == 'bert-base-multilingual-uncased' or args.model_type == 'dccuchile/bert-base-spanish-wwm-uncased':
        if args.model_type == 'bert-base-multilingual-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=4)
            save_name = 'bert-base-multilingual-uncased'
        elif args.model_type == 'dccuchile/bert-base-spanish-wwm-uncased':
            tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
            model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=4)
            save_name = 'BETO'
    else:
        raise Exception('Tipo de modelo no valido debe ser dccuchile/bert-base-spanish-wwm-uncased o bert-base-multilingual-uncased')

    file_test = args.source_file + '/test.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test = pd.DataFrame(csv3, columns=['tipo', 'text'])

    df_total = df_test

    if 'neoplasm' in args.source_file:
        file_validate = args.source_file + '/validate.csv'
        csv = pd.read_csv(file_validate, index_col=False)
        df_val = pd.DataFrame(csv, columns=['tipo', 'text'])

        file_train = args.source_file + '/train.csv'
        csv2 = pd.read_csv(file_train, index_col=False)
        df_train = pd.DataFrame(csv2, columns=['tipo', 'text'])

        df_total = pd.concat([df_val, df_train, df_test])

    df_total['tipo'] = df_total['tipo'].map(label_to_id)

    train_texts, val_texts, train_labels, val_labels = train_test_split(df_total['text'], df_total['tipo'],
                                                                        test_size=0.2,
                                                                        random_state=args.seed)

    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)

    val_dataset = AnotatorDataset(val_encodings, list(val_labels))
    train_dataset = AnotatorDataset(train_encodings, list(train_labels))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataset)
    )

    # Definir los argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10
    )

    # Inicializar el Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler)
    )

    if args.do_train:
        train_component_anotator(trainer, save_name)

    if args.do_test:
        inference_component_anotator(trainer)

