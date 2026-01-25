
# Anleitung: I2I-Translation (Dataset → LoRA → Inferenz)

Kurze Übersicht der drei Hauptschritte, um eine Image-to-Image (I2I) Translation mit diesem Repository durchzuführen:

- Schritt 1: Datensatz vorbereiten mit `scripts/generate_captions.py`
- Schritt 2: LoRA trainieren mit `scripts/train_network.py`
- Schritt 3: Inferenz ausführen mit `infer.py`

## 1) Datensatz vorbereiten

Ziel: Generiere Captions/Metadaten für die Eingangsimages und lege die Dateien in einem für das Training geeigneten Format ab.

Docker-Beispiel:

```bash
# Image bauen 
docker build -t i2i-controlnet ./ControlNet

# Script zum Generieren der Captions ausführen
docker run --d --gpus all \
	-v /path/to/repo:/workspace \
	-w /workspace \
	i2i-controlnet \
	bash -c "python3 ControlNet/scripts/generate_captions.py --input ControlNet/scripts/data/ --output ControlNet/scripts/data/"
```

Tipps:
- Passe `--input` an den Ordner mit deinen Rohbildern an.
- Prüfe `ControlNet/outputs/captions.jsonl` nach der Ausführung.

## 2) LoRA trainieren

Ziel: Trainiere eine LoRA (Low-Rank Adaptation) für die gewünschte I2I-Translation mit `scripts/train_network.py`.

Kurz: `train_network.py` liest den vorbereiteten Datensatz, führt das Training durch und speichert Checkpoints/Weights (z. B. in `output_lora/`).

Beispiel-Docker-Befehl:

```bash
docker exec -it controlnet_dev /bin/bash -c "accelerate launch --num_processes 1 --mixed_precision fp16 /workspace/ControlNet/sd-scripts-main/sd-scripts-main/train_network.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --train_data_dir /workspace/ControlNet/sd-scripts-main/sd-scripts-main/data/train --resolution 512 --enable_bucket --max_data_loader_n_workers 0 --network_module networks.lora --network_dim 8 --network_alpha 8 --train_batch_size 2 --gradient_accumulation_steps 4 --learning_rate 1e-4 --text_encoder_lr 5e-5 --max_train_steps 4000 --lr_scheduler cosine --output_dir /workspace/ControlNet/output_lora --output_name sim2real_dashcam"
```

Erläuterungen zu Flags (Beispiel):
- `--data_dir`: Pfad zu deinen Trainingsdaten / Bildern.
- `--output_dir`: Zielordner für die LoRA-Weights und Logs.
- `--epochs`, `--batch_size`, `--learning_rate`: Musterwerte — anpassen nach GPU-Ressourcen und Dataset-Größe.

Tipps:
- Überwache GPU- und Speicherverbrauch; bei OOM kleinere `--batch_size` wählen.
- Nach erfolgreichem Training findest du Dateien in `ControlNet/output_lora` (z. B. `.safetensors` oder `.ckpt`).

## 3) Inferenz (I2I-Translation)

Ziel: Produziere die übersetzten/überschriebenen Images mit `infer.py` unter Verwendung der trainierten LoRA-Weights.

Kurz: `infer.py` lädt das Basismodell + LoRA-Weights und führt die Inferenz auf Eingangsimages aus, schreibt Resultate in einen Output-Ordner.

Beispiel-Docker-Befehl:

```bash
docker exec -it controlnet_dev /bin/bash -c "python3 infer.py"
```



