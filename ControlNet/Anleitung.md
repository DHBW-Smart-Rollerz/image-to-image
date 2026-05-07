
# Anleitung: I2I-Translation (Dataset → LoRA → Inferenz)

Kurze Übersicht der drei Hauptschritte, um eine Image-to-Image (I2I) Translation mit diesem Repository durchzuführen:

- Schritt 1: Datensatz vorbereiten
- Schritt 2: LoRA trainieren 
- Schritt 3: Inferenz ausführen 

# Grundlagen 
Kurz:
Die I2I-Translation ermöglicht es, Bilder von einer Domäne (z. B. Simulation) in eine andere Domäne (z. B. Real) zu übersetzen. Dafür sind verschiedene Bausteine notwendig, die in diesem Repository bereitgestellt werden: 

1. **Datensatz**: Für den Datensatz werden Realaufnahmen benötigt. Dafür werden die Aufnahmen von Smarty verwendet, die beispielsweise auch bei der Lane Detection und Object Detection verwendet werden. Es sollten 700-800 Bilder sein, um ein gutes Training zu ermöglichen. Simulationsbilder sind nicht notwendig (nur für die Inferenz).
2. **Stable Diffusion**: Stable Diffusion ist ein vortrainiertes Text-zu-Bild-Modell, das als Basis für die I2I-Translation dient. Es wurde auf einer großen Menge von Bildern und Texten trainiert und kann daher eine Vielzahl von visuellen Konzepten verstehen und generieren.
3. **LoRA (Low-Rank Adaptation)**: Eine LoRA ist eine effiziente Methode, um ein bereits vortrainiertes Modell (hier: Stable Diffusion) an eine neue Aufgabe oder Domäne anzupassen, ohne das gesamte Modell neu trainieren zu müssen. In diesem Fall wird eine LoRA trainiert, um die I2I-Translation von Simulationsbildern zu Realbildern zu ermöglichen.
4. **ControlNet**: ControlNet ist eine Erweiterung von Stable Diffusion, die es ermöglicht, zusätzliche Steuerinformationen (z. B. Kanten, Tiefeninformationen) in den Generierungsprozess einzubeziehen. In diesem Repository wird ControlNet verwendet, um die Struktur zu konditionieren - das bedeutet, dass in den generierten Bildern an denselben Stellen Straßen, Objekte, etc. sein sollten wie in den Simulationsbildern. Dies erfolgt über die Canny-Edge-Maps der Simulationsbilder, die als zusätzliche Steuerinformationen in den Generierungsprozess einbezogen werden.

Für unseren Verwendungszweck muss lediglich der LoRA-Teil trainiert werden, da die ControlNets bereits vortrainiert sind und direkt verwendet werden können.

Während des Trainingsprozesses werden **zwei separate LoRAs** trainiert: eine für die **Straßenumgebung** (dashcam) und eine für das **Auto**. Dies wird auch beim Inferenz-Prozess beibehalten, weshalb die Inferenz aus zwei Schritten besteht:

1. **Schritt 1 - Straßenumgebung generieren**: Das Simulationsbild wird mit der `sim2real_dashcam.safetensors` LoRA verarbeitet. Dabei wird die ControlNet-Konditionierung verwendet, um sicherzustellen, dass die Struktur und das Layout der Straße korrekt bleiben. Das Modell konzentriert sich auf die realistischen Details der Straße, Markierungen, Umgebung, etc.

2. **Schritt 2 - Auto generieren**: Das zuvor generierte Bild wird dann mit der `sim2real_car.safetensors` LoRA verarbeitet, um das Auto realistisch zu gestalten. Da das Auto separat trainiert wurde, kann sich die LoRA speziell auf die Details des Fahrzeugs konzentrieren (Textur, Reflektionen, Details, etc.).

# Projektaufbau

Das Repository hat folgende Verzeichnisstruktur mit den entsprechenden Zwecken:

## Hauptordner

- **ControlNet/**: Hauptverzeichnis für die I2I-Translation mit LoRA und ControlNet
  - **scripts/**: Python-Skripte für alle Verarbeitungsschritte
    - **data/**: Verzeichnis für Datensätze
      - `1_dashcam/`: Datensatz für Straßenumgebung (Realaufnahmen + Captions)
      - `1_car/`: Datensatz für Auto-Generierung (Realaufnahmen + Captions)
    - **library/**: Bibliotheken und Utility-Funktionen für Training und Inferenz
      - Enthält Module für verschiedene Modell-Architekturen (Stable Diffusion, ControlNet, Flux, etc.)
      - Utility-Funktionen für Training, Modellmanagement und Datenverarbeitung
    - **networks/**: Netzwerk-Architecturen und LoRA-Implementierungen
      - LoRA-Module für verschiedene Modelle
      - Konvertierungs- und Merge-Skripte für LoRA-Weights
  - **inputs/**: Eingabebilder für die Inferenz (Simulationsbilder zum Übersetzen)
  - **outputs/**: Ausgabebilder nach der Inferenz (übersetzte Realbilder)
  - **output_lora/**: Trainierte LoRA-Weights nach dem Training
    - `sim2real_dashcam.safetensors`: LoRA für Straßenumgebung
    - `sim2real_car.safetensors`: LoRA für Auto-Generierung

## SLURM-Jobs

Im `ControlNet/`-Verzeichnis befinden sich SLURM-Job-Dateien für die Ausführung auf GPU-Clustern:
- `generate_captions_road.slurm`: Generiert Captions für Straßen-Datensatz
- `generate_captions_car.slurm`: Generiert Captions für Auto-Datensatz
- `mask_road.slurm`: Maskiert Autos im Straßen-Datensatz
- `mask_car.slurm`: Skaliert Auto-Datensatz auf 512x512
- `crop_pyxis.slurm`: Croppt Bilder auf Straßenteile
- `train.slurm`: Trainiert LoRA für Straßenumgebung
- `train_car.slurm`: Trainiert LoRA für Auto-Generierung
- `infer.slurm`: Führt Inferenz aus (übersetzt Simulationsbilder)


# Anleitung für die I2I-Translation mit diesem Repository

> Hinweis: Datensätze und die trainierten LoRA-Weights stehen auf dem NAS zur Verfügung. Wenn diese genutzt werden sollen, können die Schritte 1 und 2 übersprungen werden.



## 1) Datensätze vorbereiten

Ziel: Generiere Captions/Metadaten für die Eingangsimages und lege die Dateien in einem für das Training geeigneten Format ab.

Weil Stable Diffusion hauptsächlich mit Text-zu-Bild trainiert wurde, ist es wichtig, dass die Bilder mit passenden Captions/Metadaten versehen werden, damit das Modell die gewünschten Features lernen kann. 

Wir bereiten zwei Datensätze, da wir später zwei LoRAs trainieren wollen: eine für die Generierung der Straßenumgebung und eine für die Generierung des Autos. Beide Datensätze basieren auf denselben Bildern, aber mit unterschiedlichen Captions/Metadaten und Masken, damit die LoRAs lernen, sich auf unterschiedliche Features zu konzentrieren.

### 1.1) Datensatz für die Generierung der Straßenumgebung vorbereiten

Der SLURM-Job `generate_captions_road.slurm` ist dafür zuständig, die Captions für die Bilder zu generieren und diese umzubenennen. Dafür sind folgende Schritte notwendig: 

1. Die Bilder des Datensatzes müssen in den Ordner `ControlNet/scripts/data/1_dashcam` kopiert werden. Es sollten 700-800 Bilder aus Realaufnahmen sein, um ein gutes Training zu ermöglichen. Simulationsbilder sind nicht notwendig. 

2. Den SLURM-Job `generate_captions_road.slurm` ausführen: 
    ```bash 
    sbatch generate_captions_road.slurm
    ```
    Nach erfolgreicher Ausführung sollten die generierten Captions in `ControlNet/scripts/data/1_dashcam` liegen, zusammen mit den Bildern, die aufsteigend nummeriert benannt sind (z. B. `00001.jpg`, `00002.jpg`, ...).

Weil das Modell sich beim Training stark auf das Auto im Vordergrund konzentriert, wird das Auto auf den Eingabebildern maskiert. Zusätzlich werden 40% der Bilder auf die Straßen gecropt. Dazu werden die folgenden Skripte verwendet:

1. Maskieren der Autos mit `scripts/mask.py`:
    ```bash
    sbatch mask_road.slurm
    ```
2. Croppen von 40% der Bilder auf die Straße mit `scripts/crop.py`:
    ```bash
    sbatch crop_road.slurm
    ```

Der Datensatz aus vorverarbeiteten Bildern und Captions sollte nun in `ControlNet/scripts/data/1_dashcam` liegen.

### 1.2) Datensatz für die Generierung des Autos vorbereiten

Dasselbe wird für den Datensatz für die Generierung des Autos gemacht, allerdings mit abgewandelten Captions. In diesem Schritt muss auch nicht auf die Straße gecropt werden, da das Auto im Vordergrund liegt. Dafür sind folgende Schritte notwendig:

1. Die Bilder des Datensatzes müssen in den Ordner `ControlNet/scripts/data/1_car` kopiert werden. Es sollten 700-800 Bilder aus Realaufnahmen sein, um ein gutes Training zu ermöglichen. Simulationsbilder sind nicht notwendig.
2. Den SLURM-Job `generate_captions_car.slurm` ausführen: 
    ```bash 
    sbatch generate_captions_car.slurm
    ```
    Nach erfolgreicher Ausführung sollten die generierten Captions in `ControlNet/scripts/data/1_car` liegen, zusammen mit den Bildern, die aufsteigend nummeriert benannt sind (z. B. `00001.jpg`, `00002.jpg`, ...).
3. Den SLURM-Job `mask_car.slurm` ausführen. Tatsächlich werden hier keine der Bilder maskiert, aber auf 512x512 skaliert. Da dabei dasselbe Skript wie beim Maskieren der Straße verwendet wird, muss dieser Schritt trotzdem durchgeführt werden, damit die Bilder die richtige Größe haben.
    ```bash
    sbatch mask_car.slurm
    ```
Der Datensatz aus vorverarbeiteten Bildern und Captions sollte nun in `ControlNet/scripts/data/1_car` liegen.

## 2) LoRAs trainieren

Ziel: Trainiere eine LoRA (Low-Rank Adaptation) für die gewünschte I2I-Translation mit `scripts/train_network.py`.

Kurz: `train_network.py` liest den vorbereiteten Datensatz, führt das Training durch und speichert Checkpoints/Weights (in `output_lora/`).

Um die LoRA-Weights zu trainieren, kann der SLURM-Job `train.slurm` (für das LoRA für die Straßenumgebung) bzw. `train_car.slurm` (für das LoRA für das Auto) verwendet werden:

```bash 
sbatch train.slurm
```
und 

```bash
sbatch train_car.slurm
```
Das Training dauert ungefähr eine Stunde und 45 Minuten für einen LoRA. Es müssen sowohl `train.slurm` als auch `train_car.slurm` ausgeführt werden, damit beide LoRAs trainiert werden.

Ist das Training erfolgreich, sollten die LoRA-Weights in `ControlNet/output_lora/sim2real_dashcam.safetensors` und `ControlNet/output_lora/sim2real_car.safetensors` liegen.

## 3) Inferenz (I2I-Translation)

Um Simulationsbilder in Realbilder zu übersetzen, können die trainierten LoRA-Weights mit `infer.py` verwendet werden.

> Hinweise: 
> - Für die Inferenz müssen die trainierten LoRA-Weights in `ControlNet/output_lora/sim2real_dashcam.safetensors` und `ControlNet/output_lora/sim2real_car.safetensors` liegen.

Dabei sind folgende Schritte notwendig:

1. Die Simulationsbilder, die übersetzt werden sollen, müssen in den Ordner `ControlNet/inputs` kopiert werden. 
2. Den SLURM-Job `infer.slurm` ausführen:
    ```bash
    sbatch infer.slurm
    ```
3. Nach erfolgreicher Ausführung sollten die übersetzten Bilder in `ControlNet/outputs` liegen. Der Dateiname der übersetzten Bilder entspricht dem der Eingabebilder. 
