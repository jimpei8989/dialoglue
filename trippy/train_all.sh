BERT_PREFIX=$1

if [ ${BERT_PREFIX} -e "BERT" ]; then
    ./DO_PREFIX
else
    ./DO_PREFIX ${BERT_PREFIX}
fi

for bert_variant in dg banking clinc dstc8_sgd hwu multiwoz restaurant8k top; do
    echo $bert_variant
    ./DO.fewshot.sh ${BERT_PREFIX}-${bert_variant}
done
