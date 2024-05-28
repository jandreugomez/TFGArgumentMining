import argparse
import os

import pandas as pd

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def create_csv(input_dir, output_file):
    with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s' % ('id_fichero', 'id_frase', 'tipo', 'text'))
        fout.write('\n')
    for f in os.listdir(input_dir):
        if '.ann' in f:
            f_name = f[:-4]
            line_gold = {}
            list_text = []

            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.readlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        line_gold['id_fichero'] = f_name
                        line_gold['id_frase'] = line[0]
                        line_gold['tipo'] = line[1].split()[0]
                        line_gold['text'] = line[2].replace('\n', '')
                        list_text.append(line_gold['text'])

                        with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
                            for k, v in line_gold.items():
                                if k == 'text':
                                    fout.write('"%s",' % (v.replace('"', '""')))
                                else:
                                    fout.write('%s,' % (v))
                            fout.write('\n')
                with open(input_dir + f_name + '.txt', 'r', encoding='utf-8') as ftxt:
                    frases = sent_tokenize(ftxt.read())
                    for i, frase in enumerate(frases):
                        aux = True
                        for text in list_text:
                            if text in frase:
                                aux = False
                                break
                        if frase.strip() not in list_text and aux:
                            with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
                                fout.write('"%s",' % f_name)
                                fout.write('"%s",' % 'Niguno')
                                fout.write('"%s",' % 'O')
                                fout.write('"%s",' % frase.strip())
                                fout.write('\n')


def create_relations_csv(input_dir, output_file):
    with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s' % ('id_fichero', 'tipo', 'text1', 'text2'))
        fout.write('\n')
    for f in os.listdir(input_dir):
        if '.ann' in f:
            f_name = f[:-4]
            line_gold = {}

            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.readlines()
                for line in lines:
                    line = line.split('\t')
                    if 'R' in line[0]:
                        line_gold['id_fichero'] = f_name
                        line_gold['tipo'] = line[1].split()[0]
                        line_gold['text1'] = line[1].split(':')[1].split()[0]
                        line_gold['text2'] = line[1].split(':')[2]

                        with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
                            for k, v in line_gold.items():
                                fout.write('%s,' % (v))
                            fout.write('\n')


def relations_csv_with_text(input_csv, text_csv, output_data):
    df_rel = pd.read_csv(input_csv, index_col=False)
    df_text = pd.read_csv(text_csv, index_col=False)

    for i, linerel in df_rel.iterrows():
        for j, linetext in df_text.iterrows():

            if linetext['id_fichero'] == linerel['id_fichero'] and linetext['id_frase'] == linerel['text1']:
                    df_rel.loc[i, 'text1'] = str(linetext['text'])

            if linetext['id_fichero'] == linerel['id_fichero'] and linetext['id_frase'] == linerel['text2']:
                    df_rel.loc[i, 'text2'] = str(linetext['text'])

    df_rel.to_csv(output_data, index=False)
    add_pairs_without_relations_to_csv(text_csv, output_data)

def add_pairs_without_relations_to_csv(text_csv, output_data):
    df_out = pd.read_csv(output_data, index_col=False)
    df_text = pd.read_csv(text_csv, index_col=False)
    textos_rel = []
    for i, linerel in df_out.iterrows():
        textos_rel.append((linerel[2], linerel[3]))

    textos = df_text['text']
    id_fichero = df_text['id_fichero']
    labels = df_text['tipo']
    pares = []
    for i in range(len(textos)):
        for j in range(i + 1, len(textos)):
            if labels[i] != 'O' and labels[j] != 'O':
                if id_fichero[i] == id_fichero[j]:
                    pares.append((textos[i], textos[j]))

    df_nuevo = df_out
    id_list = ['id'] * len(pares)
    tipo_list = ['no_rel'] * len(pares)
    df_pares = pd.DataFrame([(id_list[x], tipo_list[x], par[0], par[1]) for x, par in enumerate(pares)], columns=['id_fichero', 'tipo', 'text1', 'text2'])
    df_pares['exists'] = df_pares.apply(lambda row: ((df_nuevo['text1'] == row['text1']) & (df_nuevo['text2'] == row['text2'])).any(), axis=1)
    nuevos_pares = df_pares[df_pares['exists'] == False].drop(columns='exists')

    df_nuevo = pd.concat([df_nuevo, nuevos_pares])
    df_nuevo.to_csv(output_data, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path donde se encuentra el corpus anotado",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path donde se va a guardar el csv con las anotaciones o relaciones",
    )
    parser.add_argument(
        "--text_translated",
        type=str,
        help="Path donde se en cuentran los textos traducidos para las relaciones",
    )
    parser.add_argument(
        "--type_data",
        type=str,
        required=True,
        help="Tipo de datos que se van a procesar. Anotaciones o Relaciones",
    )

    args = parser.parse_args()

    if args.type_data == 'Anotaciones':
        create_csv(
            input_dir=args.input_file,
            output_file=args.output_dir
        )

    if args.type_data == 'Relaciones':
        create_relations_csv(
            input_dir=args.input_file,
            output_file=args.output_dir
        )
        relations_csv_with_text(
            input_csv=args.input_file,
            text_csv=args.text_translated,
            output_data=args.output_dir
        )