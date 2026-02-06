#!/bin/bash

# Script zum Organisieren von CycleGAN Checkpoints in verschiedene Durchgänge

CHECKPOINT_DIR="./checkpoints"
SOURCE_NAME="sim2real_docker"
BASE_NAME="sim2real"

# Farben für Output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== CycleGAN Checkpoint Organizer ==="
echo ""

# Prüfen ob Source existiert
if [ ! -d "$CHECKPOINT_DIR/$SOURCE_NAME" ]; then
    echo "Error: $CHECKPOINT_DIR/$SOURCE_NAME nicht gefunden!"
    exit 1
fi

# Finde die nächste freie Run-Nummer
RUN_NUM=1
while [ -d "$CHECKPOINT_DIR/${BASE_NAME}_run${RUN_NUM}" ]; do
    RUN_NUM=$((RUN_NUM + 1))
done

NEW_NAME="${BASE_NAME}_run${RUN_NUM}"

echo -e "${YELLOW}Aktueller Checkpoint:${NC} $SOURCE_NAME"
echo -e "${GREEN}Wird umbenannt zu:${NC} $NEW_NAME"
echo ""

# Frage zur Bestätigung
read -p "Fortfahren? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Abgebrochen."
    exit 0
fi

# Umbenennen
mv "$CHECKPOINT_DIR/$SOURCE_NAME" "$CHECKPOINT_DIR/$NEW_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Erfolgreich!${NC}"
    echo ""
    echo "Checkpoint wurde verschoben nach:"
    echo "  $CHECKPOINT_DIR/$NEW_NAME"
    echo ""
    echo "Für den nächsten Trainingslauf kannst du wieder --name sim2real_docker verwenden"
    echo "oder --name ${BASE_NAME}_run$((RUN_NUM + 1)) für einen neuen Run."
else
    echo "Fehler beim Umbenennen!"
    exit 1
fi

# Zeige alle vorhandenen Runs
echo ""
echo "=== Vorhandene Trainingsläufe ==="
ls -d "$CHECKPOINT_DIR"/${BASE_NAME}_run* 2>/dev/null | while read dir; do
    run_name=$(basename "$dir")
    num_checkpoints=$(ls "$dir"/*_net_G_A.pth 2>/dev/null | wc -l)
    size=$(du -sh "$dir" | cut -f1)
    echo "  $run_name: $num_checkpoints Checkpoints ($size)"
done
