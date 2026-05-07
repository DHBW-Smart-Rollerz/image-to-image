
# Anleitung: I2I-Translation (Dataset → LoRA → Inferenz)

Kurze Übersicht der drei Hauptschritte, um eine Image-to-Image (I2I) Translation mit diesem Repository durchzuführen:

- Schritt 1: Datensatz vorbereiten
- Schritt 2: LoRA trainieren 
- Schritt 3: Inferenz ausführen 

# Grundlagen 
Kurz:
Die I2I-Translation ermöglicht es, Bilder von einer Domäne (z. B. Simulation) in eine andere Domäne (z. B. Real) zu übersetzen. Dafür sind verschiedene Bausteine notwendig, die in diesem Repository bereitgestellt werden: 

1. **Datensatz**: Für den Datensatz werden Realaufnahmen benötigt. Dafür werden die Aufnahmen von Smarty verwendet, die beispielsweise auch bei der Lane Detection und Object Detection verwendet werden. Es sollten 500-700 Bilder sein, um ein gutes Training zu ermöglichen. Simulationsbilder sind nicht notwendig (nur für die Inferenz).
2. **Stable Diffusion**: Stable Diffusion ist ein vortrainiertes Text-zu-Bild-Modell, das als Basis für die I2I-Translation dient. Es wurde auf einer großen Menge von Bildern und Texten trainiert und kann daher eine Vielzahl von visuellen Konzepten verstehen und generieren.
3. **LoRA (Low-Rank Adaptation)**: Eine LoRA ist eine effiziente Methode, um ein bereits vortrainiertes Modell (hier: Stable Diffusion) an eine neue Aufgabe oder Domäne anzupassen, ohne das gesamte Modell neu trainieren zu müssen. In diesem Fall wird eine LoRA trainiert, um die I2I-Translation von Simulationsbildern zu Realbildern zu ermöglichen.
4. **ControlNet**: ControlNet ist eine Erweiterung von Stable Diffusion, die es ermöglicht, zusätzliche Steuerinformationen (z. B. Kanten, Tiefeninformationen) in den Generierungsprozess einzubeziehen. In diesem Repository wird ControlNet verwendet, um die Struktur zu konditionieren - das bedeutet, dass in den generierten Bildern an denselben Stellen Straßen, Objekte, etc. sein sollten wie in den Simulationsbildern. 
In diesem Repository werden zwei ControlNets verwendet: 
    - **Canny ControlNet**: Konditioniert die Generierung auf den Kanten der Eingabebilder. Dadurch wird sichergestellt, dass die generierten Bilder die gleiche Struktur wie die Eingabebilder haben (z. B. gleiche Anordnung von Straßen, Autos, etc.).
    - **Soft Edge ControlNet (HED)**: Konditioniert die Generierung auf weicheren Kanten, um eine realistischere Übersetzung von Schriftzügen und Straßenschildern zu ermöglichen.

Für unseren Verwendungszweck muss lediglich der LoRA-Teil trainiert werden, da die ControlNets bereits vortrainiert sind und direkt verwendet werden können.

# Anleitung für die I2I-Translation mit diesem Repository
## 1) Datensatz vorbereiten

Ziel: Generiere Captions/Metadaten für die Eingangsimages und lege die Dateien in einem für das Training geeigneten Format ab.

Weil Stable Diffusion hauptsächlich mit Text-zu-Bild trainiert wurde, ist es wichtig, dass die Bilder mit passenden Captions/Metadaten versehen werden, damit das Modell die gewünschten Features lernen kann. 

Der SLURM-Job 'generate_captions_pyxis.slurm' ist dafür zuständig, die Captions für die Bilder zu generieren und diese umzubenennen. Dafür sind folgende Schritte notwendig: 

1. Die Bilder des Datensatzes müssen in den Ordner `ControlNet/scripts/data/1_dashcam` kopiert werden. Es sollten 500-700 Bilder aus Realaufnahmen sein, um ein gutes Training zu ermöglichen. Simulationsbilder sind nicht notwending. 
2. Den SLURM-Job `generate_captions_pyxis.slurm` ausführen: 
    ```bash 
    sbatch generate_captions_pyxis.slurm
    ```

    oder direkt im Docker-Container: 

    ```bash
    docker exec -it controlnet_dev /bin/bash -c "python3 scripts/generate_captions.py --dir scripts/data/1_dashcam --sequential"
    ```

    Nach erfolgreicher Ausführung sollten die generierten Captions in `ControlNet/scripts/data/1_dashcam` liegen, zusammen mit den Bildern, die aufsteigend nummeriert benannt sind (z. B. `00001.jpg`, `00002.jpg`, ...).

Weil das Modell sich beim Training stark auf das Auto im Vordergrund konzentriert, wird das Auto auf den Eingabebildern maskiert. Zusätzlich werden 40% der Bilder auf die Straßen gecropt. Dazu werden die folgenden Skripte verwendet:

1. Maskieren der Autos mit `scripts/mask.py`:
    ```bash
    sbatch mask_pyxis.slurm
    ```
    bzw. direkt im Docker-Container: 
    ```bash
    docker exec -it controlnet_dev /bin/bash -c "python3 scripts/mask.py"
    ```
2. Croppen von 40% der Bilder auf die Straße mit `scripts/crop.py`:
    ```bash
    sbatch crop_pyxis.slurm
    ```
    bzw. direkt im Docker-Container: 
    ```bash
    docker exec -it controlnet_dev /bin/bash -c "python3 scripts/crop_upper_half_40pct.py --dir scripts/data/1_dashcam --fraction 0.4 --seed 42"
    ```

Der Datensatz aus vorverarbeiteten Bildern und Captions sollte nun in `ControlNet/scripts/data/1_dashcam` liegen.

## 2) LoRA trainieren

Ziel: Trainiere eine LoRA (Low-Rank Adaptation) für die gewünschte I2I-Translation mit `scripts/train_network.py`.

Kurz: `train_network.py` liest den vorbereiteten Datensatz, führt das Training durch und speichert Checkpoints/Weights (z. B. in `output_lora/`).

> Hinweis: Falls in output_lora bereits eine Datei mit dem Namen `sim2real_dashcam.safetensors` existiert, muss diese gelöscht oder umbenannt werden, damit das Training erfolgreich durchgeführt werden kann.

Um die LoRA-Weights zu trainieren, kann der SLURM-Job `train.slurm` verwendet werden oder direkt im Docker-Container ausgeführt werden:

```bash 
sbatch train.slurm
```

oder 
```bash
docker exec -it controlnet_dev /bin/bash -c "python3 -m accelerate.commands.launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    /workspace/ControlNet/scripts/train_network.py \
      --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
      --train_data_dir /workspace/ControlNet/scripts/data/ \
      --resolution 512 \
      --enable_bucket \
      --max_data_loader_n_workers 0 \
      --network_module networks.lora \
      --network_dim 16 \
      --network_alpha 16 \
      --train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --text_encoder_lr 5e-5 \
      --max_train_steps 8000 \
      --lr_scheduler cosine \
      --output_dir /workspace/ControlNet/output_lora \
      --output_name sim2real_dashcam"
```
Ist das Training erfolgreich, sollten die LoRA-Weights in `ControlNet/output_lora/sim2real_dashcam.safetensors` liegen.

## 3) Inferenz (I2I-Translation)

Um Simulationsbilder in Realbilder zu übersetzen, können die trainierten LoRA-Weights mit `scripts/infer.py` verwendet werden.

> Hinweise: 
> - Für die Inferenz müssen die trainierten LoRA-Weights in `ControlNet/output_lora/sim2real_dashcam.safetensors` liegen.
> - Falls bereits Bilder in 'ControlNet/outputs' liegen, müssen diese gelöscht oder umbenannt werden, damit die Inferenz erfolgreich durchgeführt werden kann.

Dabei sind folgende Schritte notwendig:

1. Die Simulationsbilder, die übersetzt werden sollen, müssen in den Ordner `ControlNet/inputs` kopiert werden. 
2. Den SLURM-Job `infer.slurm` ausführen:
    ```bash
    sbatch infer_canny_pyxis.slurm
    ```
    oder 
    ```bash
    docker exec -it controlnet_dev /bin/bash -c "python3 /workspace/ControlNet/scripts/infer.py" 
    ```
3. Nach erfolgreicher Ausführung sollten die übersetzten Bilder in `ControlNet/outputs` liegen. Der Dateiname der übersetzten Bilder entspricht dem der Eingabebilder. 
