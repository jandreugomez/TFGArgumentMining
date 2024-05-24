import argparse
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm

def traductor(input_file, output_dir):
    model_name_mt = 'Helsinki-NLP/opus-mt-en-es'
    tokenizer_mt = MarianTokenizer.from_pretrained(model_name_mt)
    model_mt = MarianMTModel.from_pretrained(model_name_mt)

    df = pd.read_csv(input_file, index_col=False)

    with open(output_dir, 'a', encoding='utf-8') as fout:
        fout.write('%s,%s,%s,%s' % ('id_fichero', 'id_frase', 'tipo', 'text'))
        fout.write('\n')
    for _, line in tqdm(df.iterrows(), total=len(df)):

        traducido = model_mt.generate(**tokenizer_mt(line['text'], return_tensors='pt'))
        texto_traducido = [tokenizer_mt.decode(t, skip_special_tokens=True) for t in traducido]

        with open(output_dir, 'a', encoding='utf-8') as fout:
            for k, v in line.items():
                if k == 'text':
                    fout.write('"%s",' % texto_traducido[0])
                else:
                    fout.write('%s,' % v)
            fout.write('\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path del fichero csv que se va a traducir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path donde se va a guardar el csv con los textos traducidos traducido",
    )

    args = parser.parse_args()

    traductor(
        input_file=args.input_file,
        output_dir=args.output_dir
    )

