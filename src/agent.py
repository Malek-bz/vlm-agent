"""
Voice-Driven Object Detection Learning Agent
=============================================

LIVE (webcam) mode — default:
    python src/agent.py

DATASET mode — no camera needed:
    python src/agent.py --dataset 


"""

import argparse
import os
import sys
import json
import base64
import re
import io
import tempfile
import textwrap
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] openai not   found")
    sys.exit(1)

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import pyttsx3
    TTS_ENGINE = pyttsx3.init()
    TTS_ENGINE.setProperty("rate", 160)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY env var not set.")
    sys.exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_ROOT      = Path("agent_data")
MODEL_ROOT     = Path("agent_models")
YOLO_JSON      = Path(__file__).parent.parent / "yolo_classes.json"
IMG_SIZE       = 224
TARGET_ACC     = 0.90
TARGET_SAMPLES = 300
SAMPLE_RATE    = 16000

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

with open(YOLO_JSON) as f:
    _yolo_raw = json.load(f)["class"]
YOLO_CLASSES = {v.lower(): int(k) for k, v in _yolo_raw.items()}
YOLO_IDX_MAP = {int(k): v.lower() for k, v in _yolo_raw.items()}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def speak(text: str):
    print(f"[agent] {text}")
    if TTS_AVAILABLE:
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()

#   ---------------------------------------------------------------------------
# STT 
# ---------------------------------------------------------------------------

def record_audio(seconds: float = 5.0) -> bytes:
    print(f"[mic] Recording {seconds}s ... ", end="", flush=True)
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="int16")
    sd.wait()
    print("done.")
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def listen(prompt_seconds: float = 5.0) -> str:
    if not AUDIO_AVAILABLE:
        return input("you> ").strip()
    speak("Listening...")
    wav_bytes = record_audio(prompt_seconds)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text")
        text = transcript.strip()
        print(f'[STT] "{text}"')
        return text
    except Exception as e:
        print(f"[STT error] {e}")
        return ""
    finally:
        os.unlink(tmp_path)

# ---------------------------------------------------------------------------
#  parsing
# ---------------------------------------------------------------------------

INTENT_SYSTEM = textwrap.dedent("""
    You are a command parser for a robot vision agent.
    Extract the intent and target object from the user's message.

    Intents:
    - "look_for" : detect/find something. e.g. "look for bolts", "find cars"
    - "run"      : use a trained model. e.g. "run the screw model"
    - "exit"     : quit. e.g. "exit", "quit", "goodbye"
    - "unknown"  : none of the above

    Normalize object: lowercase, singular, no articles.
    "the bolts" → "bolt", "prescription bottles" → "prescription bottle"

    Respond ONLY with valid JSON: {"intent": "look_for", "object": "bolt"}
""")


def parse_intent(text: str) -> tuple[str, str | None]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": INTENT_SYSTEM},
                      {"role": "user",   "content": text}],
            response_format={"type": "json_object"},
            max_tokens=80, temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        intent = data.get("intent", "unknown")
        obj = (data.get("object") or "").strip().lower()
        return intent, obj or None
    except Exception as e:
        print(f"[intent error] {e}")
        return "unknown", None

# ---------------------------------------------------------------------------
# available  class lookup
# ---------------------------------------------------------------------------

def find_in_yolo(name: str) -> int | None:
    n = name.lower().strip()
    if n in YOLO_CLASSES:
        return YOLO_CLASSES[n]
    for cls, idx in YOLO_CLASSES.items():
        if n in cls or cls in n:
            return idx
    return None

# ---------------------------------------------------------------------------
#  Vision helpers
# ---------------------------------------------------------------------------

def frame_to_b64(frame_or_path) -> str:
    if isinstance(frame_or_path, (str, Path)):
        with open(frame_or_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    _, buf = cv2.imencode(".jpg", frame_or_path, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode()


def openai_classify(frame_or_path, object_name: str) -> bool:
    """Returns True if object_name is clearly visible."""
    b64 = frame_to_b64(frame_or_path)
    prompt = (f"Is there a '{object_name}' clearly visible in this image? "
              "Answer exactly: YES or NO.")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=5, temperature=0,
        )
        return resp.choices[0].message.content.strip().upper().startswith("Y")
    except Exception as e:
        print(f"[vision error] {e}")
        return False


def openai_describe(frame_or_path, object_name: str) -> dict:
    """Returns structured JSON description of the object in the image."""
    b64 = frame_to_b64(frame_or_path)
    prompt = (
        f"Identify and classify the '{object_name}' object(s) visible. "
        "Return ONLY JSON with fields: "
        "detected (bool), count (int), type (string), "
        "condition (string: new/used/rusty/etc), confidence (0-1 float)."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                {"type": "text", "text": prompt},
            ]}],
            max_tokens=200, temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
#  utilities
# ---------------------------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def check_dataset_structure(dataset_dir: Path) -> str:
    """
    Returns:
      "split"  - has pos/ and neg/ sub-folders
      "flat"   - flat folder of images (needs Vision sorting)
      "empty"  - no images found
    """
    pos = dataset_dir / "pos"
    neg = dataset_dir / "neg"
    if pos.is_dir() and list_images(pos):
        return "split"
    flat = list_images(dataset_dir)
    if flat:
        return "flat"
    return "empty"


def sort_flat_dataset(dataset_dir: Path, object_name: str):
    """
    If the dataset is a flat folder of images, run Vision API on each
    and sort them into pos/ and neg/ sub-folders.
    """
    images = list_images(dataset_dir)
    if not images:
        speak("No images found in the dataset folder.")
        return

    pos_dir = dataset_dir / "pos"
    neg_dir = dataset_dir / "neg"
    pos_dir.mkdir(exist_ok=True)
    neg_dir.mkdir(exist_ok=True)

    speak(f"Sorting {len(images)} images using OpenAI Vision. This may take a moment.")
    pos_count = neg_count = 0

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        detected = openai_classify(img_path, object_name)
        dest = pos_dir if detected else neg_dir
        shutil.copy2(img_path, dest / img_path.name)
        label = "pos" if detected else "neg"
        print(label)
        if detected:
            pos_count += 1
        else:
            neg_count += 1

    speak(f"Sorted: {pos_count} positives, {neg_count} negatives.")
    print(f"[dataset] pos={pos_count}, neg={neg_count}")

# ---------------------------------------------------------------------------
# CNN  model
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    def __init__(self, obj_dir: Path, transform=TRANSFORM):
        self.transform = transform
        self.samples = []
        for label, sub in enumerate(["neg", "pos"]):
            folder = obj_dir / sub
            if not folder.is_dir():
                continue
            for f in list_images(folder):
                self.samples.append((str(f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_classifier(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model.to(device)


def train_and_evaluate(obj_dir: Path, device: str, epochs: int = 8):
    dataset = CropDataset(obj_dir)
    if len(dataset) < 20:
        return None, 0.0

    val_size   = max(4, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=16, num_workers=0)

    model = build_classifier(device)
    opt   = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)

    return model, (correct / total if total else 0.0)

# ------------------------------------------------------------------------- detection 
# ---------------------------------------------------------------------------

def yolo_detection_mode(object_name: str, yolo_idx: int, dataset_dir: Path | None = None):
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError:
        speak("YOLO not installed. Run: pip install ultralytics")
        return

    yolo = UltralyticsYOLO("yolov8n.pt")

    if dataset_dir:
        # ── Dataset mode: score every image ──────────────────────────────
        images = (list_images(dataset_dir / "pos") +
                  list_images(dataset_dir / "neg") +
                  list_images(dataset_dir))
        images = list(dict.fromkeys(images))  
        if not images:
            speak("No images found in the dataset folder.")
            return

        speak(f"Running YOLO on {len(images)} dataset images for {object_name}.")
        detected_count = 0
        results_log = []

        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            results = yolo(frame, classes=[yolo_idx], verbose=False)
            n_det   = len(results[0].boxes)
            detected_count += int(n_det > 0)
            results_log.append({"file": img_path.name, "detections": n_det})

            annotated = results[0].plot()
            label = f"YOLO {object_name}: {n_det} found — {img_path.name}"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if n_det else (0, 60, 200), 2)
            cv2.imshow(f"YOLO dataset scan (any key = next, Q = stop)", annotated)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        speak(f"Done. {object_name} detected in {detected_count} of {len(images)} images.")
        print(f"\n[YOLO results] {json.dumps(results_log, indent=2)}\n")

    else:
        #    cam  
        speak(f"Running YOLO for {object_name} on webcam. Press Q to stop.")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results   = yolo(frame, classes=[yolo_idx], verbose=False)
            annotated = results[0].plot()
            detected  = len(results[0].boxes) > 0
            color     = (0, 255, 0) if detected else (0, 60, 200)
            cv2.putText(annotated,
                        f"{object_name} {'DETECTED' if detected else 'not found'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(f"YOLO - {object_name} (Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
#  learn mode
# ---------------------------------------------------------------------------

def teach_mode(object_name: str, device: str, dataset_dir: Path | None = None):
    safe_name = object_name.replace(" ", "_")
    obj_dir   = DATA_ROOT / safe_name
    pos_dir   = obj_dir / "pos"
    neg_dir   = obj_dir / "neg"

    if dataset_dir:
        _teach_from_dataset(object_name, obj_dir, pos_dir, neg_dir,
                            dataset_dir, device)
    else:
        _teach_from_webcam(object_name, obj_dir, pos_dir, neg_dir, device)


#   Dataset 

def _teach_from_dataset(object_name, obj_dir, pos_dir, neg_dir,
                        dataset_dir: Path, device: str):

    structure = check_dataset_structure(dataset_dir)

    if structure == "empty":
        speak(f"The dataset folder {dataset_dir} has no images. Aborting.")
        return

    if structure == "flat":
        speak(
            f"Found a flat image folder. I'll use OpenAI Vision to sort "
            f"them into positive and negative samples automatically."
        )
        sort_flat_dataset(dataset_dir, object_name)

    src_pos = dataset_dir / "pos"
    src_neg = dataset_dir / "neg"

    if src_pos.is_dir():
        pos_dir.mkdir(parents=True, exist_ok=True)
        for img in list_images(src_pos):
            dest = pos_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    if src_neg.is_dir():
        neg_dir.mkdir(parents=True, exist_ok=True)
        for img in list_images(src_neg):
            dest = neg_dir / img.name
            if not dest.exists():
                shutil.copy2(img, dest)

    pos_count = len(list_images(pos_dir)) if pos_dir.exists() else 0
    neg_count = len(list_images(neg_dir)) if neg_dir.exists() else 0
    speak(f"Dataset loaded: {pos_count} positives, {neg_count} negatives.")

    sample_images = list_images(pos_dir) if pos_dir.exists() else []
    if sample_images:
        speak("Running initial classification on a sample image...")
        report = openai_describe(sample_images[0], object_name)
        speak(
            f"Sample analysis — type: {report.get('type','?')}, "
            f"condition: {report.get('condition','?')}, "
            f"confidence: {int(float(report.get('confidence', 0)) * 100)}%."
        )
        print(f"\n[Vision Report]\n{json.dumps(report, indent=2)}\n")

    #  Warn if below count 
    if pos_count < TARGET_SAMPLES:
        speak(
            f"You have {pos_count} positive samples. "
            f"Ideally {TARGET_SAMPLES} are needed for best accuracy, "
            f"but I'll train on what we have."
        )

    #  Train 
    while True:
        speak(f"Training CNN on {pos_count + neg_count} images. Please wait...")
        model, acc = train_and_evaluate(obj_dir, device)

        if model is None:
            speak("Not enough data to train. Need at least 20 images.")
            return

        speak(f"Validation accuracy: {int(acc * 100)} percent.")

        if acc >= TARGET_ACC:
            MODEL_ROOT.mkdir(exist_ok=True)
            save_path = MODEL_ROOT / f"{object_name.replace(' ', '_')}.pt"
            torch.save(model.state_dict(), str(save_path))
            speak(
                f"I know {object_name} now! Model saved. "
                f"Say 'run {object_name}' to score new images with it."
            )
            print(f"[agent] Model saved: {save_path}")
            return
        else:
            speak(
                f"Accuracy is only {int(acc * 100)}%. "
                f"Please add more images to {dataset_dir}/pos/ "
                f"and re-run, or say 'train anyway' to continue."
            )
            cmd = listen(8.0) if AUDIO_AVAILABLE else input("you> ")
            if re.search(r"train anyway|continue|skip", cmd, re.I):
                MODEL_ROOT.mkdir(exist_ok=True)
                save_path = MODEL_ROOT / f"{object_name.replace(' ', '_')}.pt"
                torch.save(model.state_dict(), str(save_path))
                speak(f"Model saved at {int(acc * 100)}% accuracy.")
                return
            speak("Add more images to the dataset folder and try again.")
            return


#  cam 

def _teach_from_webcam(object_name, obj_dir, pos_dir, neg_dir, device):
    speak(
        f"I don't know {object_name}. "
        "Place some in front of the camera and say 'look at them'."
    )
    while True:
        cmd = listen(6.0)
        if re.search(r"look at (them|it)|ready|go|start|ok", cmd, re.I):
            break
        speak("Say 'look at them' when ready.")

    cap = cv2.VideoCapture(0)

    ret, snapshot = cap.read()
    if ret:
        speak("Analysing snapshot...")
        report = openai_describe(snapshot, object_name)
        speak(
            f"I see: type={report.get('type','?')}, "
            f"condition={report.get('condition','?')}, "
            f"confidence={int(float(report.get('confidence',0))*100)}%."
        )
        print(f"\n[Vision Report]\n{json.dumps(report, indent=2)}\n")

    existing   = len(list_images(pos_dir)) if pos_dir.exists() else 0
    sample_id  = existing
    needed     = max(0, TARGET_SAMPLES - existing)
    speak(f"Collecting {needed} frames. Keep {object_name} in view.")

    frame_counter = 0
    last_detected = False

    while sample_id < TARGET_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1

        if frame_counter % 10 == 0:
            last_detected = openai_classify(frame, object_name)

        color  = (0, 255, 0) if last_detected else (0, 60, 200)
        cv2.putText(frame, f"Collecting {sample_id}/{TARGET_SAMPLES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"{'[VISIBLE]' if last_detected else '[not seen]'} {object_name}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow(f"Teaching: {object_name} (Q to stop early)", frame)

        if last_detected:
            pos_dir.mkdir(parents=True, exist_ok=True)
            neg_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pos_dir / f"{sample_id:05d}.jpg"), frame)
            h, w = frame.shape[:2]
            s = min(h, w) // 5
            cv2.imwrite(str(neg_dir / f"{sample_id:05d}.jpg"), frame[0:s, 0:s])
            sample_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    while True:
        speak(f"Training CNN on {sample_id} samples...")
        model, acc = train_and_evaluate(obj_dir, device)
        if model is None:
            speak("Not enough data yet.")
            return
        speak(f"Validation accuracy: {int(acc * 100)} percent.")
        if acc >= TARGET_ACC:
            MODEL_ROOT.mkdir(exist_ok=True)
            sp = MODEL_ROOT / f"{object_name.replace(' ', '_')}.pt"
            torch.save(model.state_dict(), str(sp))
            speak(f"I know {object_name} now! Model saved.")
            return
        else:
            speak(f"Only {int(acc * 100)}%. Add more samples.")
            # collect 50 more
            cap = cv2.VideoCapture(0)
            added = 0
            fc = 0
            ld = False
            while added < 50:
                ret, frame = cap.read()
                if not ret:
                    break
                fc += 1
                if fc % 10 == 0:
                    ld = openai_classify(frame, object_name)
                if ld:
                    cv2.imwrite(str(pos_dir / f"{sample_id:05d}.jpg"), frame)
                    h, w = frame.shape[:2]
                    s = min(h, w) // 5
                    cv2.imwrite(str(neg_dir / f"{sample_id:05d}.jpg"), frame[0:s, 0:s])
                    sample_id += 1
                    added += 1
                cv2.imshow("Extra samples (Q stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
#  inference can be better
# ---------------------------------------------------------------------------

def cnn_detection_mode(object_name: str, device: str, dataset_dir: Path | None = None):
    model_path = MODEL_ROOT / f"{object_name.replace(' ', '_')}.pt"
    if not model_path.exists():
        speak(f"No trained model for {object_name}. Say 'look for {object_name}' to teach me first.")
        return

    model = build_classifier(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    if dataset_dir:
        # ── Score every image in dataset ──────────────────────────────────
        images = (list_images(dataset_dir / "pos") +
                  list_images(dataset_dir / "neg") +
                  list_images(dataset_dir))
        images = list(dict.fromkeys(images))
        if not images:
            speak("No images found in dataset folder.")
            return

        speak(f"Running CNN on {len(images)} images for {object_name}.")
        tp = tn = fp = fn = 0
        results_log = []

        for img_path in images:
            is_pos = img_path.parent.name == "pos"
            frame  = cv2.imread(str(img_path))
            if frame is None:
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), 1)[0][1].item()

            predicted = prob > 0.5
            results_log.append({
                "file":      img_path.name,
                "folder":    img_path.parent.name,
                "prob":      round(prob, 3),
                "predicted": "pos" if predicted else "neg",
            })
            if is_pos and predicted:     tp += 1
            elif is_pos and not predicted: fn += 1
            elif not is_pos and predicted: fp += 1
            else:                          tn += 1

            color = (0, 255, 0) if predicted else (0, 60, 200)
            cv2.putText(frame, f"{object_name}: {prob:.0%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, img_path.name, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow(f"CNN dataset — {object_name} (any key=next, Q=stop)", frame)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

        total   = tp + tn + fp + fn
        overall = (tp + tn) / total if total else 0
        speak(
            f"Done. Overall accuracy {int(overall * 100)}%. "
            f"True positives: {tp}, true negatives: {tn}, "
            f"false positives: {fp}, false negatives: {fn}."
        )
        print(f"\n[CNN Results]\n{json.dumps(results_log, indent=2)}\n")

    else:
        # ── Live webcam ───────────────────────────────────────────────────
        speak(f"Running CNN for {object_name} on webcam. Press Q to stop.")
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), 1)[0][1].item()
            color = (0, 255, 0) if prob > 0.5 else (0, 60, 200)
            cv2.putText(frame, f"{object_name}: {prob:.0%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow(f"CNN - {object_name} (Q quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Voice object detection agent")
    parser.add_argument(
        "--dataset", metavar="PATH", default=None,
        help=(
            "Path to a dataset folder instead of using the webcam. "
            "Expected layout: <PATH>/pos/*.jpg and <PATH>/neg/*.jpg  "
            "OR a flat folder of images (Vision API will sort them). "
            "Example: --dataset datasets/screws"
        ),
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset) if args.dataset else None

    if dataset_dir:
        if not dataset_dir.exists():
            print(f"[ERROR] Dataset folder not found: {dataset_dir}")
            sys.exit(1)
        print(f"[agent] Dataset mode: {dataset_dir.resolve()}")
    else:
        print("[agent] Live webcam mode")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[agent] Device: {device}")
    print(f"[agent] YOLO classes: {len(YOLO_CLASSES)}")
    print(f"[agent] Audio: {'yes' if AUDIO_AVAILABLE else 'text fallback'}")
    print(f"[agent] TTS:   {'yes' if TTS_AVAILABLE else 'print only'}\n")

    if dataset_dir:
        speak(
            "Dataset mode active. Say things like: "
            "'look for screws', 'run screws', or 'exit'."
        )
    else:
        speak(
            "Hello! Say 'look for bolts', 'look for cars', or 'exit'."
        )

    while True:
        text = listen(6.0)
        if not text:
            continue

        intent, obj = parse_intent(text)
        print(f"[parsed] intent={intent!r}, object={obj!r}")

        if intent == "exit" or re.search(r"\bexit\b|\bquit\b", text, re.I):
            speak("Goodbye!")
            break

        elif intent == "look_for" and obj:
            yolo_idx = find_in_yolo(obj)
            if yolo_idx is not None:
                speak(f"{obj} is YOLO class {yolo_idx}. Starting detection.")
                yolo_detection_mode(obj, yolo_idx, dataset_dir)
            else:
                model_path = MODEL_ROOT / f"{obj.replace(' ', '_')}.pt"
                if model_path.exists():
                    speak(f"I already know {obj}. Running my trained model.")
                    cnn_detection_mode(obj, device, dataset_dir)
                else:
                    speak(f"{obj} is not a standard YOLO class.")
                    teach_mode(obj, device, dataset_dir)

        elif intent == "run" and obj:
            cnn_detection_mode(obj, device, dataset_dir)

        else:
            speak("Try: 'look for screws', 'run screws', or 'exit'.")


if __name__ == "__main__":
    main()
