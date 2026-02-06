#!/bin/bash

# Script zum Anzeigen aller Trainingsläufe

CHECKPOINT_DIR="./checkpoints"
BASE_NAME="sim2real"

echo "=== Alle CycleGAN Trainingsläufe ==="
echo ""

# Zähle alle Runs
TOTAL=0

# Liste alle sim2real Checkpoints
for dir in "$CHECKPOINT_DIR"/${BASE_NAME}*; do
    if [ -d "$dir" ]; then
        TOTAL=$((TOTAL + 1))
        run_name=$(basename "$dir")
        
        # Zähle Checkpoints
        num_checkpoints=$(ls "$dir"/*_net_G_A.pth 2>/dev/null | grep -o "[0-9]*_net_G_A.pth" | wc -l)
        
        # Größe
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        
        # Letztes Training
        if [ -f "$dir/loss_log.txt" ]; then
            last_epoch=$(grep "End of epoch" "$dir/loss_log.txt" 2>/dev/null | tail -1 | grep -oP "epoch \K[0-9]+")
            last_date=$(stat -c %y "$dir/loss_log.txt" | cut -d' ' -f1)
        else
            last_epoch="?"
            last_date="unbekannt"
        fi
        
        echo "[$TOTAL] $run_name"
        echo "    Checkpoints: $num_checkpoints"
        echo "    Letzte Epoche: $last_epoch"
        echo "    Größe: $size"
        echo "    Zuletzt: $last_date"
        echo ""
    fi
done

if [ $TOTAL -eq 0 ]; then
    echo "Keine Trainingsläufe gefunden in $CHECKPOINT_DIR"
else
    echo "Gesamt: $TOTAL Trainingsläufe"
fi
