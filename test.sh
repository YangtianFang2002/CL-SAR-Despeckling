#!/bin/bash

# change this line
export CUDA_VISIBLE_DEVICES="0"

export TMP_TXT=$WEIGHT_FILE.txt

# clear screen and print WEIGHT_FILE
clear

declare -a test_dir=(
                     "datasets/SARdata/multitest/Capella-X" \
                     "datasets/SARdata/multitest/TerraSAR&Tandem-X"
                     )

declare -a method_pairs=(
    "options/real/MDN1-default.yml experiments/MDN1-default/models/net_g_latest.pth"
    # "options/real/MDN1-t2.yml experiments/MDN1-t2/models/net_g_latest.pth"
)

for method in "${method_pairs[@]}"; do
    
    read YML_FILE WEIGHT_FILE <<< "$method"
    echo "Using $WEIGHT_FILE"
    export TMP_TXT=$WEIGHT_FILE.txt

    # Loop through each pair
    for pair in "${test_dir[@]}"; do
        # Split the pair into VAL_NAME and VAL_META
        read TEST_DIR <<< "$pair"

        # Execute the script with the current TEST_DIR
        echo "python basicsr/predict.py -i $TEST_DIR -opt $YML_FILE --force_yml path:pretrain_network_g=$WEIGHT_FILE &> $TMP_TXT"
        python basicsr/predict.py -i $TEST_DIR -opt $YML_FILE --force_yml path:pretrain_network_g=$WEIGHT_FILE &> $TMP_TXT

        # Grep the specific line
        grep "Tested" $TMP_TXT
        grep "Final Metric" $TMP_TXT

        # echo TMP TXT
        # cat $TMP_TXT

        # Optional: Clear tmp.txt for the next iteration
        > $TMP_TXT
    done
    rm $TMP_TXT
done
