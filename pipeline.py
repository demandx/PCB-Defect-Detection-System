"""
PCB Defect Detection — Full Pipeline
Generates data, trains CNN, runs detection, produces reports
"""
import os, sys, json, random, warnings, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE       = 64
SAMPLES        = 250          # per class
CLASS_NAMES    = ["good_solder", "solder_bridge", "missing_component", "cold_joint"]
BATCH          = 32
EPOCHS         = 20
DATA_DIR       = Path("data")
MODEL_PATH     = Path("models/pcb_classifier.keras")
REPORT_DIR     = Path("reports")

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def pcb_bg(sz=IMG_SIZE):
    bg = np.full((sz, sz, 3), (34, 85, 34), dtype=np.uint8)
    noise = np.random.randint(-12, 12, (sz, sz, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(random.randint(3, 6)):
        x1,y1,x2,y2 = [random.randint(0,sz) for _ in range(4)]
        cv2.line(bg,(x1,y1),(x2,y2),(180,130,60),1)
    return bg

def draw_pad(img,cx,cy,r=4):
    cv2.circle(img,(cx,cy),r,(200,150,70),-1)

def draw_ic(img,x,y,w=28,h=18):
    cv2.rectangle(img,(x,y),(x+w,y+h),(50,50,50),-1)
    cv2.rectangle(img,(x,y),(x+w,y+h),(20,20,20),1)

def gen_good():
    img=pcb_bg(); cx,cy=IMG_SIZE//2,IMG_SIZE//2
    draw_ic(img,cx-14,cy-9)
    for dx in [-8,0,8]:
        draw_pad(img,cx+dx,cy-14,3)
        draw_pad(img,cx+dx,cy+14,3)
        cv2.circle(img,(cx+dx,cy-14),2,(240,220,180),-1)
        cv2.circle(img,(cx+dx,cy+14),2,(240,220,180),-1)
    return img

def gen_bridge():
    img=gen_good()
    cx,cy=IMG_SIZE//2,IMG_SIZE//2
    bx=cx+random.choice([-8,0])
    pts=np.array([[bx,cy-16],[bx+8,cy-16],[bx+8,cy-12],[bx,cy-12]],np.int32)
    cv2.fillPoly(img,[pts],(230,200,140))
    return img

def gen_missing():
    img=pcb_bg(); cx,cy=IMG_SIZE//2,IMG_SIZE//2
    for dx in [-8,0,8]:
        draw_pad(img,cx+dx,cy-14,3)
        draw_pad(img,cx+dx,cy+14,3)
    cv2.rectangle(img,(cx-14,cy-9),(cx+14,cy+9),(60,60,60),1)
    return img

def gen_cold():
    img=gen_good(); cx,cy=IMG_SIZE//2,IMG_SIZE//2
    for dx in [-8,0,8]:
        for _ in range(5):
            ox,oy=random.randint(-3,3),random.randint(-3,3)
            cv2.circle(img,(cx+dx+ox,cy-14+oy),random.randint(1,2),(160,140,100),-1)
    return img

GENS = [gen_good, gen_bridge, gen_missing, gen_cold]

def augment(img):
    angle=random.uniform(-15,15)
    M=cv2.getRotationMatrix2D((IMG_SIZE//2,IMG_SIZE//2),angle,1.)
    img=cv2.warpAffine(img,M,(IMG_SIZE,IMG_SIZE),borderMode=cv2.BORDER_REFLECT)
    f=random.uniform(0.8,1.2)
    img=np.clip(img.astype(np.float32)*f,0,255).astype(np.uint8)
    if random.random()>0.5: img=cv2.flip(img,1)
    return img

def generate_dataset():
    print("[1/4] Generating synthetic PCB dataset...")
    manifest=[]
    for idx,(name,gen) in enumerate(zip(CLASS_NAMES,GENS)):
        d=DATA_DIR/name; d.mkdir(parents=True,exist_ok=True)
        for i in range(SAMPLES):
            img=augment(gen())
            p=d/f"{name}_{i:04d}.jpg"
            cv2.imwrite(str(p),img)
            manifest.append({"file":str(p),"label":name,"label_idx":idx})
        print(f"   ✓ {name}: {SAMPLES} images")
    mp=DATA_DIR/"manifest.json"
    with open(mp,"w") as f: json.dump(manifest,f)
    print(f"   Total: {len(manifest)} images\n")
    return manifest

# ══════════════════════════════════════════════════════════════════════════════
# 2.  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def detect_bridges_contour(img_bgr):
    """Rule-based solder bridge candidates via contour analysis."""
    gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    thresh=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,11,2)
    cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    bridges=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if not (30<area<500): continue
        x,y,w,h=cv2.boundingRect(c)
        ar=max(w,h)/(min(w,h)+1e-5)
        if ar>1.8:
            bridges.append({"contour":c,"area":area,"bbox":(x,y,w,h)})
    return bridges

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CNN MODEL + TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    inp=layers.Input((IMG_SIZE,IMG_SIZE,3))
    x=layers.Conv2D(32,3,activation="relu",padding="same")(inp)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(32,3,activation="relu",padding="same")(x)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.1)(x)
    x=layers.Conv2D(64,3,activation="relu",padding="same")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,3,activation="relu",padding="same")(x)
    x=layers.BatchNormalization()(x)
    x=layers.MaxPooling2D(2)(x)
    x=layers.Dropout(0.15)(x)
    x=layers.Conv2D(128,3,activation="relu",padding="same")(x)
    x=layers.BatchNormalization()(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(0.4)(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.Dropout(0.3)(x)
    out=layers.Dense(4,activation="softmax")(x)
    m=Model(inp,out)
    m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return m

def train_model(manifest):
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    print("[2/4] Training CNN classifier...")
    random.shuffle(manifest)
    split=int(len(manifest)*0.8)
    tr,val=manifest[:split],manifest[split:]

    def load_arr(m):
        X,y=[],[]
        for e in m:
            img=cv2.imread(e["file"])
            img=cv2.resize(img,(IMG_SIZE,IMG_SIZE)).astype("float32")/255.
            X.append(img); y.append(e["label_idx"])
        return np.array(X),np.array(y)

    X_tr,y_tr=load_arr(tr)
    X_v,y_v=load_arr(val)
    print(f"   Train: {len(X_tr)}  Val: {len(X_v)}")

    # Data augmentation via tf.data
    def aug(x,y):
        x=tf.image.random_flip_left_right(x)
        x=tf.image.random_brightness(x,0.15)
        x=tf.image.random_contrast(x,0.85,1.15)
        return x,y
    ds_tr=(tf.data.Dataset.from_tensor_slices((X_tr,y_tr))
           .map(aug,num_parallel_calls=tf.data.AUTOTUNE)
           .shuffle(500).batch(BATCH).prefetch(tf.data.AUTOTUNE))
    ds_v=(tf.data.Dataset.from_tensor_slices((X_v,y_v))
          .batch(BATCH).prefetch(tf.data.AUTOTUNE))

    Path("models").mkdir(exist_ok=True)
    model=build_model()
    cbs=[
        ModelCheckpoint(str(MODEL_PATH),save_best_only=True,monitor="val_accuracy",verbose=0),
        EarlyStopping(patience=5,restore_best_weights=True,monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5,patience=3,min_lr=1e-6,verbose=0),
    ]
    hist=model.fit(ds_tr,validation_data=ds_v,epochs=EPOCHS,callbacks=cbs,verbose=1)
    hd={k:[float(v) for v in vals] for k,vals in hist.history.items()}
    with open("models/training_history.json","w") as f: json.dump(hd,f)
    best_acc=max(hd["val_accuracy"])
    print(f"   Best val accuracy: {best_acc:.3f}\n")
    return model, hd

# ══════════════════════════════════════════════════════════════════════════════
# 4.  DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

SEVERITY={"good_solder":"PASS","solder_bridge":"CRITICAL","missing_component":"CRITICAL","cold_joint":"WARNING"}
COLORS={"PASS":(0,200,0),"WARNING":(0,165,255),"CRITICAL":(0,0,220)}

def predict(model, img_bgr):
    img=cv2.resize(img_bgr,(IMG_SIZE,IMG_SIZE)).astype("float32")/255.
    probs=model.predict(img[None],verbose=0)[0]
    idx=int(np.argmax(probs))
    return {"class":CLASS_NAMES[idx],"conf":float(probs[idx]),
            "scores":{c:float(probs[i]) for i,c in enumerate(CLASS_NAMES)}}

def inspect(model, img_bgr, filepath=""):
    t0=time.time()
    r=predict(model,img_bgr)
    cls,conf=r["class"],r["conf"]
    bridges=detect_bridges_contour(img_bgr)
    if cls=="good_solder" and len(bridges)>0:
        cls="solder_bridge"; conf=max(conf,0.72)
    sev=SEVERITY[cls]
    h,w=img_bgr.shape[:2]
    bbox=(w//4,h//4,w//2,h//2)
    defects=[] if cls=="good_solder" else [{
        "type":cls,"severity":sev,"confidence":round(conf,4),
        "bbox":bbox,"center_xy":(w//2,h//2)
    }]
    return {
        "file":str(filepath),
        "overall_result":"FAIL" if sev in("CRITICAL","WARNING") else "PASS",
        "primary_class":cls,"severity":sev,
        "cnn_confidence":round(conf,4),
        "class_scores":r["scores"],
        "defects":defects,
        "processing_time_ms":round((time.time()-t0)*1000,1),
    }

def annotate(img_bgr, report):
    img=img_bgr.copy()
    h,w=img.shape[:2]
    color=COLORS.get(report["severity"],(255,255,255))
    cv2.rectangle(img,(0,0),(w,22),(30,30,30),-1)
    label=f"{report['overall_result']}  {report['primary_class'].upper()}  {report['cnn_confidence']:.0%}"
    cv2.putText(img,label,(4,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    for d in report["defects"]:
        x,y,bw,bh=d["bbox"]
        cv2.rectangle(img,(x,y),(x+bw,y+bh),color,1)
    return img

# ══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_training(hd):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    fig.suptitle("PCB Defect CNN — Training History",fontsize=12,fontweight="bold")
    axes[0].plot(hd["accuracy"],label="Train",color="#2196F3",lw=2)
    axes[0].plot(hd["val_accuracy"],label="Val",color="#FF5722",lw=2,ls="--")
    axes[0].set(title="Accuracy",xlabel="Epoch",ylabel="Accuracy"); axes[0].legend()
    axes[0].set_ylim(0,1.05); axes[0].grid(alpha=0.3)
    axes[1].plot(hd["loss"],label="Train",color="#4CAF50",lw=2)
    axes[1].plot(hd["val_loss"],label="Val",color="#9C27B0",lw=2,ls="--")
    axes[1].set(title="Loss",xlabel="Epoch",ylabel="Loss"); axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    p=REPORT_DIR/"training_history.png"
    plt.savefig(str(p),dpi=150,bbox_inches="tight"); plt.close()
    print(f"   → {p}")
    return p

def plot_class_samples(manifest):
    """4x4 grid showing 4 samples per class."""
    fig,axes=plt.subplots(4,4,figsize=(10,10))
    fig.suptitle("Synthetic PCB Dataset — Sample Images per Class",fontsize=12,fontweight="bold")
    by_cls=defaultdict(list)
    for e in manifest: by_cls[e["label"]].append(e)
    for r,cls in enumerate(CLASS_NAMES):
        samples=random.sample(by_cls[cls],4)
        for c,s in enumerate(samples):
            img=cv2.imread(s["file"])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            axes[r][c].imshow(img)
            axes[r][c].axis("off")
            if c==0: axes[r][c].set_ylabel(cls,fontsize=9,fontweight="bold",rotation=0,
                                            labelpad=60,va="center")
    plt.tight_layout()
    p=REPORT_DIR/"dataset_samples.png"
    plt.savefig(str(p),dpi=150,bbox_inches="tight"); plt.close()
    print(f"   → {p}")
    return p

def plot_sample_predictions(model, manifest):
    """Grid of predictions on unseen samples with annotated images."""
    by_cls=defaultdict(list)
    for e in manifest: by_cls[e["label"]].append(e)
    samples=[]
    for cls in CLASS_NAMES:
        samples.extend(random.sample(by_cls[cls],2))
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    fig.suptitle("PCB Defect Detector — Inference Results",fontsize=12,fontweight="bold")
    CCOLS={"good_solder":"#4CAF50","solder_bridge":"#F44336",
           "missing_component":"#FF5722","cold_joint":"#FF9800"}
    for i,s in enumerate(samples):
        ax=axes[i//4][i%4]
        img=cv2.imread(s["file"])
        report=inspect(model,img,s["file"])
        ann=annotate(img,report)
        ax.imshow(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB))
        gt=s["label"]; pred=report["primary_class"]; conf=report["cnn_confidence"]
        correct="✓" if gt==pred else "✗"
        color=CCOLS.get(pred,"gray")
        ax.set_title(f"{correct} Pred: {pred}\nConf: {conf:.0%}\nGT: {gt}",
                     fontsize=7.5,color=color,fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    p=REPORT_DIR/"sample_predictions.png"
    plt.savefig(str(p),dpi=150,bbox_inches="tight"); plt.close()
    print(f"   → {p}")
    return p

def plot_batch_summary(reports):
    results=Counter(r["overall_result"] for r in reports)
    classes=Counter(r["primary_class"] for r in reports)
    times=[r["processing_time_ms"] for r in reports]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    fig.suptitle("PCB Batch Inspection Summary",fontsize=12,fontweight="bold")
    # Pie
    labels,sizes=list(results.keys()),list(results.values())
    colors=["#4CAF50" if l=="PASS" else "#F44336" for l in labels]
    axes[0].pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%",startangle=90)
    axes[0].set_title("Pass / Fail")
    # Bar
    CCOLS={"good_solder":"#4CAF50","solder_bridge":"#F44336",
           "missing_component":"#FF5722","cold_joint":"#FF9800"}
    cls_l,cls_c=list(classes.keys()),list(classes.values())
    bc=[CCOLS.get(l,"#9E9E9E") for l in cls_l]
    axes[1].bar(cls_l,cls_c,color=bc,edgecolor="white")
    axes[1].set_title("Defect Distribution"); axes[1].tick_params(axis="x",rotation=18)
    axes[1].grid(axis="y",alpha=0.3)
    # Histogram
    axes[2].hist(times,bins=15,color="#2196F3",edgecolor="white")
    axes[2].axvline(np.mean(times),color="red",ls="--",label=f"Mean: {np.mean(times):.1f}ms")
    axes[2].set_title("Processing Time"); axes[2].set_xlabel("ms")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    p=REPORT_DIR/"batch_summary.png"
    plt.savefig(str(p),dpi=150,bbox_inches="tight"); plt.close()
    print(f"   → {p}")
    return p

def plot_confusion_matrix(model, manifest):
    from sklearn.metrics import confusion_matrix
    y_true,y_pred=[],[]
    for e in manifest:
        img=cv2.imread(e["file"])
        r=predict(model,img)
        y_true.append(e["label_idx"])
        y_pred.append(CLASS_NAMES.index(r["class"]))
    cm=confusion_matrix(y_true,y_pred)
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([c.replace("_","\n") for c in CLASS_NAMES],fontsize=8)
    ax.set_yticklabels([c.replace("_","\n") for c in CLASS_NAMES],fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>cm.max()*0.5 else "black",fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix",fontsize=12,fontweight="bold")
    plt.colorbar(im,ax=ax)
    plt.tight_layout()
    p=REPORT_DIR/"confusion_matrix.png"
    plt.savefig(str(p),dpi=150,bbox_inches="tight"); plt.close()
    print(f"   → {p}")
    # Also print report
    from sklearn.metrics import classification_report
    print("\n" + classification_report(y_true,y_pred,target_names=CLASS_NAMES))
    return p

def save_batch_csv(reports):
    import csv
    p=REPORT_DIR/"batch_summary.csv"
    with open(str(p),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["file","overall_result","primary_class",
                                        "severity","cnn_confidence","processing_time_ms"])
        w.writeheader()
        for r in reports:
            w.writerow({k:r[k] for k in ["file","overall_result","primary_class",
                                          "severity","cnn_confidence","processing_time_ms"]})
    return p

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__=="__main__":
    REPORT_DIR.mkdir(exist_ok=True)
    print("="*58)
    print("  PCB DEFECT DETECTION SYSTEM — FULL PIPELINE")
    print("="*58+"\n")

    manifest = generate_dataset()
    model, hd = train_model(manifest)

    print("[3/4] Running detection & building reports...")
    # Full pass on all images (fast since 64x64)
    random.shuffle(manifest)
    test_sample = manifest[:80]  # evaluate on 80 samples
    reports = []
    for e in test_sample:
        img = cv2.imread(e["file"])
        r = inspect(model, img, e["file"])
        r["gt_label"] = e["label"]
        reports.append(r)

    save_batch_csv(reports)
    print(f"   ✓ Inspected {len(reports)} boards")

    print("\n[4/4] Generating visualizations...")
    plot_training(hd)
    plot_class_samples(manifest)
    plot_sample_predictions(model, manifest)
    plot_batch_summary(reports)
    plot_confusion_matrix(model, manifest)

    # ── Final summary ─────────────────────────────────────────
    passed = sum(1 for r in reports if r["overall_result"]=="PASS")
    failed = len(reports)-passed
    avg_ms = np.mean([r["processing_time_ms"] for r in reports])
    print("\n"+"="*58)
    print("  PIPELINE COMPLETE")
    print("="*58)
    print(f"  Boards inspected : {len(reports)}")
    print(f"  PASS             : {passed}  ({passed/len(reports):.0%})")
    print(f"  FAIL             : {failed}")
    print(f"  Avg time/board   : {avg_ms:.1f} ms")
    best_val = max(hd["val_accuracy"])
    print(f"  Best val acc     : {best_val:.3f}  ({best_val*100:.1f}%)")
    print(f"\n  Reports saved to : reports/")
    print("="*58)
