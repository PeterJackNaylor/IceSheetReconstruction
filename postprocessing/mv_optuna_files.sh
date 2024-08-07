#!/bin/bash

optuna="./optuna"
multiple="./multiple"
csv="./INR_mini_Model_MFN_Coh_false_Swa_true_Dem_true_PDEc_false__trial_scores.csv"
destination="$1"


if [ -f "$optuna" ]; then
    cp -r "$optuna" "$destination"
else
    echo "The file does not exist, no action taken."
fi

if [ -f "$multiple" ]; then
    cp -r "$multiple" "$destination"
else
    echo "The file does not exist, no action taken."
fi

if [ -f "$csv" ]; then
    cp "$csv" "$destination"
else
    echo "The file does not exist, no action taken."
fi
