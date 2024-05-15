from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
from tqdm import tqdm

def traductor(df, output_dir):
    model_name_mt = 'Helsinki-NLP/opus-mt-en-es'
    tokenizer_mt = MarianTokenizer.from_pretrained(model_name_mt)
    model_mt = MarianMTModel.from_pretrained(model_name_mt)

    with open(output_dir + '.csv', 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s' % ('id_fichero', 'id_frase', 'tipo', 'text'))
        fout.write('\n')
    for _, line in tqdm(df.iterrows(), total=len(df)):

        traducido = model_mt.generate(**tokenizer_mt(line['text'], return_tensors='pt'))
        texto_traducido = [tokenizer_mt.decode(t, skip_special_tokens=True) for t in traducido]

        with open(output_dir + '.csv', 'a', encoding='utf-8') as fout:
            for k, v in line.items():
                if k == 'text':
                    fout.write('"%s",' % texto_traducido[0])
                else:
                    fout.write('%s,' % v)
            fout.write('\n')





if __name__ == '__main__':

    file_validate = './data/validate.csv'
    file_dir = 'data_translated/validate'
    df_validate = pd.read_csv(file_validate, index_col=False)
    traductor(df_validate, file_dir)

    file_train = './data/train.csv'
    file_dir = 'data_translated/train'
    df_train = pd.read_csv(file_train, index_col=False)
    traductor(df_train, file_dir)

    file_test = 'data/test.csv'
    file_dir = 'data_translated/test'
    df_test = pd.read_csv(file_test, index_col=False)
    traductor(df_test, file_dir)

