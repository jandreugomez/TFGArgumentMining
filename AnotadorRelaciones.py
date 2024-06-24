
import argparse
import os
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Mapeo de relaciones de texto a numéricas
rel_to_id = {"Support": 0, "Attack": 1, "Partial-Attack": 2, "no_rel": 3}
id_to_rel = {0: "Support", 1: "Attack", 2: "Partial-Attack", 3: "no_rel"}

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

    for i, label in enumerate(rel_to_id.keys()):
        metrics[f'precision_{label}'] = precision[i]
        metrics[f'recall_{label}'] = recall[i]
        metrics[f'f1_{label}'] = f1[i]

    return metrics

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

def inferenceRelationAnotator(relation_trainer):
        Support = 0
        Attack = 0
        Partial_Attack = 0
        no_rel_label = 0
        for f in tqdm(os.listdir(args.output_path), total=len(os.listdir(args.corpus_dir))):
            if '.ann' in f:
                f_name = f[:-4]

            corpus, id_sentence = get_corpus_with_anotations(args.output_path, f_name)

            # Crear pares de frases
            pares = []
            for i in range(len(corpus)):
                for j in range(i + 1, len(corpus)):
                    pares.append((corpus[i], corpus[j]))

            try:
                pair_texts1, pair_texts2 = zip(*pares)
            except:
                continue
            pair_encodings = tokenizer(list(pair_texts1), list(pair_texts2), truncation=True, padding=True,
                                              max_length=128)

            relation_inference_dataset = AnotatorDataset(pair_encodings, [0]*len(pares))

            # Realizar la inferencia
            predictions = relation_trainer.predict(relation_inference_dataset)

            # Obtener las etiquetas predichas
            predicted_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)

            # Convertir etiquetas numéricas a texto
            relation_predicted_labels_text = [id_to_rel[label.item()] for label in predicted_labels]

            Support = Support + relation_predicted_labels_text.count('Support')
            Attack = Attack + relation_predicted_labels_text.count('Attack')
            Partial_Attack = Partial_Attack + relation_predicted_labels_text.count('Partial-Attack')
            no_rel_label = no_rel_label + relation_predicted_labels_text.count('no_rel')

            with open(args.output_path + f_name + '.ann', 'a', encoding='utf-8') as fout:
                j = 0
                for i, (label, pair) in enumerate(zip(relation_predicted_labels_text, pares)):
                    if label != 'no_rel':
                        j = j + 1
                        fout.write('%s' % ('R' + str(j)))
                        fout.write('\t')
                        fout.write('%s' % label)
                        fout.write('\t')
                        fout.write('%s' % f'Arg1:{id_sentence[corpus.index(pair[0])]}')
                        fout.write('\t')
                        fout.write('%s' % f'Arg2:{id_sentence[corpus.index(pair[1])]}')
                        fout.write('\n')

        if not os.path.exists('./label_results/' + args.output_path):
            os.makedirs('./label_results/' + args.output_path)

        with open('./label_results/' + args.output_path + 'rel_labels.txt', 'w', encoding='utf-8') as fout:
            fout.write('Support:' + str(Support))
            fout.write('\n')
            fout.write('Attack:' + str(Attack))
            fout.write('\n')
            fout.write('Partial-Attack:' + str(Partial_Attack))
            fout.write('\n')
            fout.write('rel_label:' + str(no_rel_label))


def get_corpus_with_anotations(corpus_dir, file_name):
    textos = []
    id_sentence = []
    with open(corpus_dir + file_name + '.ann', 'r', encoding='utf-8') as fann:
        lines = fann.readlines()
        for line in lines:
            line = line.split('\t')
            id_sentence.append(line[0])
            textos.append(line[2].replace('\n', ''))
        return textos, id_sentence

def proccess_dataset(args, source_file):
    file_test = source_file + 'neoplasm/test_relation.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test_neo = pd.DataFrame(csv3, columns=['tipo', 'text1', 'text2'])

    file_validate = source_file + 'neoplasm/validate_relation.csv'
    csv = pd.read_csv(file_validate, index_col=False)
    df_val_neo = pd.DataFrame(csv, columns=['tipo', 'text1', 'text2'])

    file_train = source_file + 'neoplasm/train_relation.csv'
    csv2 = pd.read_csv(file_train, index_col=False)
    df_train_neo = pd.DataFrame(csv2, columns=['tipo', 'text1', 'text2'])

    file_test = source_file + 'glaucoma/test_relation.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test_glau = pd.DataFrame(csv3, columns=['tipo', 'text1', 'text2'])

    file_test = source_file + 'mixed/test_relation.csv'
    csv3 = pd.read_csv(file_test, index_col=False)
    df_test_mixed = pd.DataFrame(csv3, columns=['tipo', 'text1', 'text2'])

    df_total_neo = pd.concat([df_val_neo, df_train_neo, df_test_neo])

    df_total_neo['tipo'] = df_total_neo['tipo'].map(rel_to_id)
    df_test_glau['tipo'] = df_test_glau['tipo'].map(rel_to_id)
    df_test_mixed['tipo'] = df_test_mixed['tipo'].map(rel_to_id)

    train_texts1_glau, val_texts1_glau, train_texts2_glau, val_texts2_glau, train_labels_glau, val_labels_glau = train_test_split(df_test_glau['text1'],
                                                                                            df_test_glau['text2']
                                                                                            , df_test_glau['tipo'],
                                                                                            test_size=0.2,
                                                                                            random_state=args.seed)

    train_texts1_mixed, val_texts1_mixed, train_texts2_mixed, val_texts2_mixed, train_labels_mixed, val_labels_mixed = train_test_split(df_test_mixed['text1'],
                                                                                                df_test_mixed['text2'],
                                                                                                df_test_mixed['tipo'],
                                                                                                test_size=0.2,
                                                                                                random_state=args.seed)

    train_texts1_neo, val_texts1_neo, train_texts2_neo, val_texts2_neo, train_labels_neo, val_labels_neo = train_test_split(df_total_neo['text1'],
                                                                                        df_total_neo['text2'],
                                                                                        df_total_neo['tipo'],
                                                                                        test_size=0.2,
                                                                                        random_state=args.seed)

    train_texts1 = pd.concat([train_texts1_glau, train_texts1_mixed, train_texts1_neo])
    val_texts1 = pd.concat([val_texts1_glau, val_texts1_mixed, val_texts1_neo])
    train_texts2 = pd.concat([train_texts2_glau, train_texts2_mixed, train_texts2_neo])
    val_texts2 = pd.concat([val_texts2_glau, val_texts2_mixed, val_texts2_neo])
    train_labels = pd.concat([train_labels_glau, train_labels_mixed, train_labels_neo])
    val_labels = pd.concat([val_labels_glau, val_labels_mixed, val_labels_neo])

    return train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels


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
        help='Si se guarda el modelo'
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
        "--seed", default=42, type=int,
        help="Indica la semilla para la reproducibilidad del mezclado de datos de entrenamiento."
    )

    args = parser.parse_args()


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.use_saved_model:
        if args.model_type == 'dccuchile/bert-base-spanish-wwm-uncased':
            save_name = 'BETO'
        if args.model_type == 'bert-base-multilingual-uncased':
            save_name = 'bert-base-multilingual-uncased'
        if args.model_type == 'PlanTL-GOB-ES/roberta-base-bne':
            save_name = 'ROBERTA-BNE'
            tokenizer = RobertaTokenizer.from_pretrained('./tokenizer_relation_' + save_name)
            model = RobertaForSequenceClassification.from_pretrained('./model_relation_' + save_name, num_labels=4)
        else:
            tokenizer = BertTokenizer.from_pretrained('./tokenizer_relation_' + save_name)
            model = BertForSequenceClassification.from_pretrained('./model_relation_' + save_name, num_labels=4)
    else:
        if args.model_type == 'bert-base-multilingual-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=4)
            save_name = 'bert-base-multilingual-uncased'
        elif args.model_type == 'dccuchile/bert-base-spanish-wwm-uncased':
            tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
            model = BertForSequenceClassification.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', num_labels=4)
            save_name = 'BETO'
        elif args.model_type == 'PlanTL-GOB-ES/roberta-base-bne':
            save_name = 'ROBERTA-BNE'
            tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
            model = RobertaForSequenceClassification.from_pretrained('PlanTL-GOB-ES/roberta-base-bne', num_labels=4)
        else:
            raise Exception(
                'Tipo de modelo no valido debe ser dccuchile/bert-base-spanish-wwm-uncased o bert-base-multilingual-uncased o PlanTL-GOB-ES/roberta-base-bne')

    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = proccess_dataset(args, args.source_file)

    val_encodings = tokenizer(list(val_texts1), list(val_texts2), truncation=True, padding=True, max_length=512)
    train_encodings = tokenizer(list(train_texts1), list(train_texts2), truncation=True, padding=True, max_length=512)

    val_dataset = AnotatorDataset(val_encodings, list(val_labels))
    train_dataset = AnotatorDataset(train_encodings, list(train_labels))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataset)
    )

    # Definir los argumentos de entrenamiento
    relation_training_args = TrainingArguments(
        output_dir='./results_relation',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_relation',
        logging_steps=10,
    )

    if args.do_test:
        metricas = None
    else:
        metricas = compute_metrics

    # Inicializar el Trainer
    relation_trainer = Trainer(
        model=model,
        args=relation_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=metricas
    )

    if args.do_train:
        # Entrenar el modelo
        relation_trainer.train()

        if args.save_model:
            model.save_pretrained('./model_relation_' + save_name)
            tokenizer.save_pretrained('./tokenizer_relation_' + save_name)

    if args.do_eval:
        eval_results = relation_trainer.evaluate()
        if not os.path.exists('./eval_results_rel/'):
            os.makedirs('./eval_results_rel/')
        with open('eval_results/eval_results_relations_' + save_name + '.txt', "w") as writer:
            for key in sorted(eval_results.keys()):
                writer.write("%s = %s\n" % (key, str(eval_results[key])))

    if args.do_test:
        inferenceRelationAnotator(relation_trainer)