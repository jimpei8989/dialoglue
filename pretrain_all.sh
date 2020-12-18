
# MODEL_NAME_OR_PATH="bert-base-uncased"
# NAME_PREFIX="bert"

MODEL_NAME_OR_PATH="berts/convbert"
NAME_PREFIX="convbert"

for d in banking clinc dstc8_sgd hwu multiwoz restaurant8k top; do
    python3 pretrain_dg.py \
        --model_name_or_path $MODEL_NAME_OR_PATH
        --mlm_data_txt data_utils/dialoglue/mlm_$d.txt
        --output_dir berts/$NAME_PREFIX-$d
        --num_epochs 20
done
