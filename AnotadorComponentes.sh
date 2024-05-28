export SOURCE_FILE='./data_translated/glaucoma'
export CORPUS_DIR='./CorpusSinAnotaciones/dev/eudract/'
export OUTPUT_PATH='data_annotated_mBERT/dev/eudract/'
export MODEL_TYPE='bert-base-multilingual-uncased'
#export MODEL_TYPE='dccuchile/bert-base-spanish-wwm-uncased'

#Añadir lo siguiente para hacer la inferencia
# --do_test


python AnotadorComponentes.py \
--source_file=$SOURCE_FILE \
--corpus_dir=$CORPUS_DIR \
--output_path=$OUTPUT_PATH \
--model_type=$MODEL_TYPE \
--save_model \
--use_saved_model \
--seed=15 \
--learning_rate=2e-5 \
--do_test
