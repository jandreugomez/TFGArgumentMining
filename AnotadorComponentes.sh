export SOURCE_FILE='./data_translated/'
export CORPUS_DIR='./CorpusSinAnotaciones/train/eudract/'
export OUTPUT_PATH='data_annotated_ROBERTA_BNE/train/eudract/'

#export MODEL_TYPE='bert-base-multilingual-uncased'
#export MODEL_TYPE='dccuchile/bert-base-spanish-wwm-uncased'
export MODEL_TYPE='PlanTL-GOB-ES/roberta-base-bne'

#Añadir lo siguiente para hacer la inferencia
# --do_test


python AnotadorComponentes.py \
--source_file=$SOURCE_FILE \
--corpus_dir=$CORPUS_DIR \
--output_path=$OUTPUT_PATH \
--model_type=$MODEL_TYPE \
--use_saved_model \
--learning_rate=2e-5 \
--do_train \
--do_eval
