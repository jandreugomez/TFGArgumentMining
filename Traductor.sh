export INPUT_PATH='./data/neoplasm/validate.csv'
export OUTPUT_TRANSLATION_PATH='./data_translated/neoplasm/validate.csv'

python Traductor.py \
--input_file=$INPUT_PATH \
--output_dir=$OUTPUT_TRANSLATION_PATH