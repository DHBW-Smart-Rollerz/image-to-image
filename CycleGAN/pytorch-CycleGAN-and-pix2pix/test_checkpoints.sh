#!/bin/bash

# Script zum schnellen Testen mehrerer Checkpoints

CHECKPOINT_NAME="sim2real_run1"
TEST_DATA="./datasets/sim2real"
NUM_TEST=50
EPOCHS=(100 200 300 400 500 600 700)

echo "=== CycleGAN Checkpoint Tester ==="
echo "Testing checkpoints from: $CHECKPOINT_NAME"
echo "Number of test images: $NUM_TEST"
echo ""

# Check if checkpoint directory exists
if [ ! -d "checkpoints/$CHECKPOINT_NAME" ]; then
    echo "Error: checkpoints/$CHECKPOINT_NAME not found!"
    exit 1
fi

# Loop through epochs
for epoch in "${EPOCHS[@]}"; do
    echo "----------------------------------------"
    echo "Testing Epoch $epoch..."
    
    # Check if checkpoint exists
    if [ ! -f "checkpoints/$CHECKPOINT_NAME/${epoch}_net_G_A.pth" ]; then
        echo "⚠️  Checkpoint $epoch not found, skipping..."
        continue
    fi
    
    # Run test
    python test.py \
        --dataroot "$TEST_DATA" \
        --name "$CHECKPOINT_NAME" \
        --model cycle_gan \
        --epoch "$epoch" \
        --num_test "$NUM_TEST" \
        --no_dropout
    
    if [ $? -eq 0 ]; then
        echo "✓ Epoch $epoch completed"
    else
        echo "✗ Epoch $epoch failed"
    fi
    echo ""
done

echo "========================================="
echo "Testing completed!"
echo ""
echo "Results are in: results/$CHECKPOINT_NAME/test_*/"
echo ""
echo "To view results, open the index.html files:"
for epoch in "${EPOCHS[@]}"; do
    html_file="results/$CHECKPOINT_NAME/test_${epoch}/index.html"
    if [ -f "$html_file" ]; then
        echo "  - Epoch $epoch: $html_file"
    fi
done
