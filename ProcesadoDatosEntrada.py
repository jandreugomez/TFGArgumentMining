import os
import shutil

import pandas as pd


def create_csv(input_dir, output_file):
    with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s' % ('id_fichero', 'id_frase', 'tipo', 'text'))
        fout.write('\n')
    for f in os.listdir(input_dir):
        if '.ann' in f:
            f_name = f[:-4]
            line_gold = {}

            with open(input_dir + f_name + '.ann', 'r', encoding='utf-8') as fann:
                lines = fann.readlines()
                for line in lines:
                    line = line.split('\t')
                    if 'T' in line[0]:
                        line_gold['id_fichero'] = f_name
                        line_gold['id_frase'] = line[0]
                        line_gold['tipo'] = line[1].split()[0]
                        line_gold['text'] = line[2].replace('\n', '')

                        with open(output_file + '.csv', 'a', encoding='utf-8') as fout:
                            for k, v in line_gold.items():
                                if k == 'text':
                                    fout.write('"%s",' % (v.replace('"', '""')))
                                else:
                                    fout.write('%s,' % (v))
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


if __name__ == '__main__':

    # file_dir = './data/glaucoma/validate'
    # path = "old_data/dev/neoplasm_dev/"
    # create_csv(path, file_dir)
    # file_dir = './data/glaucoma/train'
    # path = "old_data/train/glaucoma_train/"
    # create_csv(path, file_dir)
    # file_dir = 'data/glaucoma/test'
    # path = "old_data/test/glaucoma_test/"
    # create_csv(path, file_dir)

    # file_dir = './data/neoplasm/validate_relation'
    # path = "old_data/dev/neoplasm_dev/"
    # create_relations_csv(path, file_dir)
    # file_dir = './data/neoplasm/train_relation'
    # path = "old_data/train/neoplasm_train/"
    # create_relations_csv(path, file_dir)
    # file_dir = 'data/neoplasm/test_relation'
    # path = "old_data/test/neoplasm_test/"
    # create_relations_csv(path, file_dir)

    # input_csv = './data/neoplasm/validate_relation.csv'
    # text_csv = './data_translated/neoplasm/validate.csv'
    # output_data = './data_translated/neoplasm/validate_relation.csv'
    # relations_csv_with_text(input_csv, text_csv, output_data)
    #
    # input_csv = './data/neoplasm/train_relation.csv'
    # text_csv = './data_translated/neoplasm/train.csv'
    # output_data = './data_translated/neoplasm/train_relation.csv'
    # relations_csv_with_text(input_csv, text_csv, output_data)

    input_csv = './data/mixed/test_relation.csv'
    text_csv = './data_translated/mixed/test.csv'
    output_data = './data_translated/mixed/test_relation.csv'
    relations_csv_with_text(input_csv, text_csv, output_data)