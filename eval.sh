#!/bin/bash

# change this line
export CUDA_VISIBLE_DEVICES="0"

# Define pairs of VAL_NAME and VAL_META
declare -a val_pairs=("ValSetAll datasets/val.txt" "ValSetMountains datasets/val_mountains.txt" "ValSetHills datasets/val_hills.txt" "ValSetPlains datasets/val_plains.txt")

declare -a method_pairs=(
    "options/real/MDN1-default.yml experiments/MDN1-default/models/net_g_latest.pth"
    # "options/real/MDN1-t2.yml experiments/MDN1-default/models/net_g_latest.pth"
)


for method in "${method_pairs[@]}"; do
    
    read YML_FILE WEIGHT_FILE <<< "$method"
    echo "Using $WEIGHT_FILE"
    export TMP_TXT=$WEIGHT_FILE.txt

    # Loop through each pair
    for pair in "${val_pairs[@]}"; do
        # Split the pair into VAL_NAME and VAL_META
        read VAL_NAME VAL_META <<< "$pair"

        # Execute the script with the current VAL_NAME and VAL_META
        echo "python basicsr/train.py -opt $YML_FILE --val --force_yml datasets:val:name=$VAL_NAME datasets:val:meta_info=$VAL_META path:pretrain_network_g=$WEIGHT_FILE &> $TMP_TXT"
        python basicsr/train.py -opt $YML_FILE --val --force_yml datasets:val:name=$VAL_NAME datasets:val:meta_info=$VAL_META path:pretrain_network_g=$WEIGHT_FILE &> $TMP_TXT

        # Grep the specific line
        grep "Validation $VAL_NAME" $TMP_TXT

        # Optional: Clear tmp.txt for the next iteration
        > $TMP_TXT
    done

    # Optional: Remove tmp.txt after completion
    rm $TMP_TXT
done