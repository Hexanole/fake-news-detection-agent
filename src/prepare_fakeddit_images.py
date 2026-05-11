import argparse
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image

from src.config import BASE_DIR


def safe_filename(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in value)
    return cleaned[:120]


def is_valid_url(url: str) -> bool:
    if not isinstance(url, str):
        return False

    url = url.strip()

    if not url or url.lower() in ["nan", "none", "null"]:
        return False

    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"] and bool(parsed.netloc)


def download_image(url: str, output_path: Path, timeout: int = 15) -> bool:
    try:
        headers = {
            "User-Agent": "DisinfoAgent/1.0"
        }

        response = requests.get(url, headers=headers, timeout=timeout, stream=True)

        if response.status_code != 200:
            return False

        content_type = response.headers.get("Content-Type", "").lower()

        if "image" not in content_type:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        # Vérifier que PIL peut ouvrir l'image
        with Image.open(output_path) as img:
            img.verify()

        return True

    except Exception:
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass
        return False


def prepare_images(
    input_tsv: Path,
    output_csv: Path,
    images_dir: Path,
    max_per_class: int,
    sleep_seconds: float,
) -> None:
    if not input_tsv.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_tsv}")

    df = pd.read_csv(input_tsv, sep="\t")

    required = {"id", "clean_title", "image_url", "6_way_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df[["id", "clean_title", "image_url", "6_way_label"]].copy()
    df = df.dropna(subset=["id", "image_url", "6_way_label"])

    df["clean_title"] = df["clean_title"].fillna("").astype(str)
    df["image_url"] = df["image_url"].astype(str)
    df["6_way_label"] = pd.to_numeric(df["6_way_label"], errors="coerce")
    df = df.dropna(subset=["6_way_label"])
    df["6_way_label"] = df["6_way_label"].astype(int)

    df = df[df["image_url"].apply(is_valid_url)].copy()

    df["label"] = df["6_way_label"].apply(lambda x: "real" if x == 0 else "fake")

    real_df = df[df["label"] == "real"].sample(
        n=min(max_per_class, len(df[df["label"] == "real"])),
        random_state=42,
    )

    fake_df = df[df["label"] == "fake"].sample(
        n=min(max_per_class, len(df[df["label"] == "fake"])),
        random_state=42,
    )

    work_df = pd.concat([real_df, fake_df])
    work_df = work_df.sample(frac=1, random_state=42).reset_index(drop=True)

    rows = []

    print(f"Images candidates : {len(work_df)}")
    print("Téléchargement en cours...")

    for index, row in work_df.iterrows():
        label = row["label"]
        image_id = safe_filename(str(row["id"]))
        image_path = images_dir / label / f"{image_id}.jpg"

        if image_path.exists():
            success = True
        else:
            success = download_image(row["image_url"], image_path)
            time.sleep(sleep_seconds)

        if success:
            rows.append({
                "image_path": str(image_path),
                "label": label,
                "text": row["clean_title"],
                "image_url": row["image_url"],
            })

        if (index + 1) % 100 == 0:
            print(f"{index + 1}/{len(work_df)} traitées | images valides : {len(rows)}")

    output = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8")

    print("\nDataset image préparé ✅")
    print(f"Sortie : {output_csv}")
    print(output["label"].value_counts() if not output.empty else "Aucune image téléchargée")


def main():
    parser = argparse.ArgumentParser(description="Préparer dataset image Fakeddit")

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--images-dir", type=str, required=True)

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Nombre maximum d'images par classe real/fake"
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Pause entre téléchargements"
    )

    args = parser.parse_args()

    prepare_images(
        input_tsv=Path(args.input),
        output_csv=Path(args.output),
        images_dir=Path(args.images_dir),
        max_per_class=args.max_per_class,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()