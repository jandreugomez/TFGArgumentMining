export INPUT_PATH='./old_data/dev/neoplasm_dev/'
export TEXT_TRANSLATED='./data_translated/mixed/test.csv'
export OUTPUT_PATH='./data/neoplasm/validate'
export TYPE_DATA='Anotaciones'

python ProcesadoDatosEntrada.py \
--input_file=$INPUT_PATH \
--text_translated=$TEXT_TRANSLATED \
--output_dir=$OUTPUT_PATH \
--type_data=$TYPE_DATA