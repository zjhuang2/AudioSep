#!/bin/bash
# Script to run the complete noise sensitivity feasibility study

echo "AudioSep Noise Sensitivity Feasibility Study"
echo "==========================================="

# Step 1: Prepare audio samples (if not already done)
if [ ! -f "audioset_samples/trigger_samples.json" ]; then
    echo "Step 1: Preparing AudioSet samples..."
    python prepare_audioset_samples.py \
        --audioset-csv evaluation/metadata/audioset_eval_strong.csv \
        --output-dir audioset_samples
else
    echo "Step 1: Audio samples already prepared."
fi

# Step 2: Run the evaluation
echo -e "\nStep 2: Running noise sensitivity evaluation..."
python noise_sensitivity_evaluation.py \
    --model-config config/audiosep_base.yaml \
    --checkpoint checkpoint/audiosep_base_4M_steps.ckpt \
    --audio-samples-json audioset_samples/trigger_samples.json \
    --noise-samples-json audioset_samples/noise_samples.json \
    --output-dir noise_sensitivity_results

# Step 3: Generate visualizations (optional)
echo -e "\nStep 3: Results saved to noise_sensitivity_results/"
echo "Check the following files:"
echo "  - evaluation_results_*.json: Detailed results for each test"
echo "  - evaluation_summary_*.csv: Summary statistics"
echo "  - mixed_audio/: Mixed audio samples used for testing"
echo "  - separated_audio/: AudioSep output (separated and eliminated sounds)"

echo -e "\nEvaluation complete!"