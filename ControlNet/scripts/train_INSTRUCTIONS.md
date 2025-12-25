# Minimaler Train-Workflow (ControlNet + LoRA Hinweise)

Dieses Dokument enthält ein kleines, robustes Vorgehen, um ein ControlNet-/LoRA-Feintuning zu starten. Es nutzt die offiziellen `diffusers`-Beispiele (empfohlen) und zeigt die minimal notwendigen Schritte, um auf CUDA (GPU) zu trainieren.

1) Vorbereitung

- Erstelle das vorbereitete Dataset (Control-Maps + Resize):

```bash
python ControlNet/scripts/prepare_dataset.py --sim_dir ControlNet/data/sim --out_dir ControlNet/data/prepared --val_ratio 0.1
```

Das erzeugt `ControlNet/data/prepared/sim_images/` und `ControlNet/data/prepared/sim_control/` sowie die Dateien `train_pairs.txt` und `val_pairs.txt`.

2) Trainingsskript verwenden

Für ein echtes, getestetes Training empfehle ich das Diffusers-Example-Repository (ControlNet fine-tune). Klone es lokal:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/examples/controlnet
```

3) Beispiel `accelerate`-Startbefehl (LoRA / Adapter)

Ersetze `--train_data_dir`/`--output_dir` nach Bedarf. Dieses Kommando nutzt ein externes Beispielskript aus dem `diffusers`-Repo (dessen Parameter sich je nach Version leicht unterscheiden können).

```bash
python -m pip install -r ControlNet/requirements.txt
python -m pip install accelerate transformers peft safetensors

accelerate launch --num_processes 1 --num_machines 1 run_train_controlnet.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir ControlNet/data/prepared \
  --resolution 512 \
  --output_dir ControlNet/checkpoints/lora \
  --learning_rate 2e-4 \
  --max_train_steps 1000 \
  --train_batch_size 1 \
  --mixed_precision fp16 \
  --controlnet_model_name_or_path lllyasviel/sd-controlnet-canny \
  --use_lora True
```

Hinweis: `run_train_controlnet.py` ist Platzhalter für das konkrete Trainingsskript im `diffusers/examples/controlnet` Ordner. Wenn du das Beispiel-Skript anders nennst, passe den Pfad an.

4) Nach dem Training

- Das Training speichert Adapter-/LoRA-Gewichte in `ControlNet/checkpoints/lora`. Diese lädst du anschließend in `StableDiffusionControlNetImg2ImgPipeline` oder in dein Inferenz-Skript, je nachdem wie das Beispiel die Ausgabe formatiert.

5) Schnelltest der neuen Gewichte

- Modifiziere `ControlNet/scripts/infer.py`, um die Adapter/Checkpoint zu laden (falls nötig). Dann:

```bash
python ControlNet/scripts/infer.py --input ControlNet/inputs --output ControlNet/outputs --device cuda
```


