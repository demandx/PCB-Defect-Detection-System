I built this because I wanted a computer vision project that actually maps to something real solder defect inspection is a genuine problem on electronics assembly lines and a good fit for CNNs.
The whole thing runs as a single script. No dataset to download, no API keys, nothing to configure. Just run it and it handles everything from generating training data to producing inspection reports.
bashpython pipeline.py

**The problem it solves**
On a PCB assembly line, boards go through soldering and then need to be inspected before they move to the next stage. Doing this manually is slow and inconsistent. The four defects this system catches:

**Good solder** : clean joint, passes inspection
**Solder bridge** : excess solder shorting two adjacent pads together. Hard fault, board is dead.
**Missing component** : IC chip never placed. Pads are there, chip isn't.
**Cold joint** : bad soldering. Looks soldered but the joint is weak and will fail under heat or vibration.


I built this because I wanted a computer vision project that actually maps to something real  solder defect inspection is a genuine problem on electronics assembly lines and a good fit for CNNs.
The whole thing runs as a single script. **No dataset to download, no API keys, nothing to configure.** Just run it and it handles everything from generating training data to producing inspection reports.
bashpython pipeline.py

**The problem it solves
On a PCB assembly line, boards go through soldering and then need to be inspected before they move to the next stage. Doing this manually is slow and inconsistent.**
The four defects this system catches:

good solder : clean joint, passes inspection
solder bridge : excess solder shorting two adjacent pads together. Hard fault, board is dead.
missing component : IC chip never placed. Pads are there, chip isn't.
cold joint : bad soldering. Looks soldered but the joint is weak and will fail under heat or vibration.


**How it works**
Data generation
**I didn't use a real PCB dataset. Instead I wrote OpenCV code that draws synthetic 64×64 images** each class has a distinct visual signature baked into the generator:

gen_good() — dark IC body, bright gold pads, white highlight dot at each pad centre
gen_bridge() — starts from a good board, adds a large golden polygon bridging two pads
gen_missing() — no IC body at all, just bare pads and a ghost outline of where the chip should be
cold_joint() — IC present but pads are dark and grainy, no shiny highlight

Each image gets light augmentation before saving , small rotation, brightness jitter, 50% horizontal flip. Nothing aggressive enough to wipe out the class-defining features.
The CNN
Three conv blocks, progressively deeper (16 → 32 → 64 filters), BatchNorm after every conv, GlobalAveragePooling instead of Flatten to keep parameter count low. Small Dense head (64 neurons) with dropout before the 4-class softmax output.
Trained with Adam at 3e-4, label smoothing of 0.05, EarlyStopping watching val_accuracy. Balanced 80/20 split per class so the val set always has equal representation.
Bridge detection fallback
The CNN handles classification, but I also wrote a rule-based bridge detector using contour analysis as a safety net. After thresholding, it looks for blobs in the 40–600px² area range with aspect ratio > 1.8 (elongated = bridge-shaped). If the CNN says clean but contours find a bridge, the contour result wins. Belt and suspenders.


Installation
bashpip install opencv-python numpy matplotlib scikit-learn tensorflow
On Ubuntu 24.04 you'll need to add --break-system-packages because of PEP 668:
bashpip install opencv-python numpy matplotlib scikit-learn tensorflow --break-system-packages
Also if python gives a "not found" error, either install the alias:
bashsudo apt install python-is-python3
or just use python3 directly.

Changing things
Everything tunable is at the top of pipeline.py:
pythonIMG_SIZE = 64     # resolution fed to CNN
SAMPLES  = 300    # images per class
BATCH    = 32     
EPOCHS   = 30     # EarlyStopping usually kicks in before this

If you want to use real PCB images
Swap the data/ folder for a real labelled dataset (the Kaggle PCB defect dataset works). Keep the same subfolder structure matching CLASS_NAMES. Skip the generate_dataset() call in main and build the manifest from your folder instead.
For better accuracy on real data, bump IMG_SIZE to 224 and switch build_model() to use MobileNetV2 with ImageNet weights — the architecture change is already stubbed in the code. On a real dataset this gets you to 90%+ without much effort.

Stack
Python 3.12 · OpenCV · NumPy · TensorFlow/Keras · Matplotlib · scikit-learn
Data generation
I didn't use a real PCB dataset. Instead I wrote OpenCV code that draws synthetic 64×64 images — each class has a distinct visual signature baked into the generator:

gen_good() — dark IC body, bright gold pads, white highlight dot at each pad centre
gen_bridge() — starts from a good board, adds a large golden polygon bridging two pads
gen_missing() — no IC body at all, just bare pads and a ghost outline of where the chip should be
cold_joint() — IC present but pads are dark and grainy, no shiny highlight

Each image gets light augmentation before saving  small rotation, brightness jitter, 50% horizontal flip. Nothing aggressive enough to wipe out the class-defining features.
The CNN
Three conv blocks, progressively deeper (16 → 32 → 64 filters), BatchNorm after every conv, GlobalAveragePooling instead of Flatten to keep parameter count low. Small Dense head (64 neurons) with dropout before the 4-class softmax output.
Trained with Adam at 3e-4, label smoothing of 0.05, EarlyStopping watching val_accuracy. Balanced 80/20 split per class so the val set always has equal representation.
Bridge detection fallback
The CNN handles classification, but I also wrote a rule-based bridge detector using contour analysis as a safety net. After thresholding, it looks for blobs in the 40–600px² area range with aspect ratio > 1.8 (elongated = bridge-shaped). If the CNN says clean but contours find a bridge, the contour result wins. Belt and suspenders.
Reports
After training and running inference on 80 test boards, the pipeline saves:
reports/
├── training_history.png      # accuracy + loss curves
├── dataset_samples.png       # 4 samples per class in a grid
├── sample_predictions.png    # GT vs predicted with confidence scores
├── confusion_matrix.png      
├── batch_summary.png         # pass/fail pie, defect breakdown, latency histogram
└── batch_summary.csv         # one row per inspected board

Project layout
PCB Defect Detection System/
├── pipeline.py        ← everything is in here
├── data/              ← created on first run
│   ├── good_solder/
│   ├── solder_bridge/
│   ├── missing_component/
│   ├── cold_joint/
│   └── manifest.json
├── models/
│   ├── pcb_classifier.keras
│   └── training_history.json
└── reports/
    └── (all the PNGs + CSV)

Installation
bashpip install opencv-python numpy matplotlib scikit-learn tensorflow
On Ubuntu 24.04 you'll need to add --break-system-packages because of PEP 668:
bashpip install opencv-python numpy matplotlib scikit-learn tensorflow --break-system-packages
Also if python gives a "not found" error, either install the alias:
bashsudo apt install python-is-python3
or just use python3 directly.

Changing things
Everything tunable is at the top of pipeline.py:
pythonIMG_SIZE = 64     # resolution fed to CNN
SAMPLES  = 300    # images per class
BATCH    = 32     
EPOCHS   = 30     # EarlyStopping usually kicks in before this

If you want to use real PCB images
Swap the data/ folder for a real labelled dataset (the Kaggle PCB defect dataset works). Keep the same subfolder structure matching CLASS_NAMES. Skip the generate_dataset() call in main and build the manifest from your folder instead.
For better accuracy on real data, bump IMG_SIZE to 224 and switch build_model() to use MobileNetV2 with ImageNet weights — the architecture change is already stubbed in the code. On a real dataset this gets you to 90%+ without much effort.

Stack
Python 3.12 · OpenCV · NumPy · TensorFlow/Keras · Matplotlib · scikit-learn
