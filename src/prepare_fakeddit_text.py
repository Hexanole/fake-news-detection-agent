import argparse
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    return pd.read_csv(path, sep="\t")


def prepare_fakeddit(
    input_path: Path,
    output_path: Path,
    max_samples: int | None = None,
) -> None:
    df = read_table(input_path)

    print("Colonnes disponibles :")
    print(list(df.columns))

    required = {"clean_title", "6_way_label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    df = df[["clean_title", "6_way_label"]].copy()
    df = df.dropna(subset=["clean_title", "6_way_label"])

    df["clean_title"] = df["clean_title"].astype(str).str.strip()
    df["6_way_label"] = pd.to_numeric(df["6_way_label"], errors="coerce")
    df = df.dropna(subset=["6_way_label"])

    df["6_way_label"] = df["6_way_label"].astype(int)

    # Fakeddit 6-way :
    # 0 = true / real
    # 1,2,3,4,5 = fake / satire / misleading / manipulated / etc.
    df["label"] = df["6_way_label"].apply(
        lambda x: "real" if x == 0 else "fake"
    )

    output = df.rename(columns={"clean_title": "text"})[["text", "label"]]

    output = output[output["text"].str.len() > 3]
    output = output.drop_duplicates(subset=["text"])

    if max_samples is not None and max_samples > 0 and len(output) > max_samples:
        per_class = max_samples // 2

        real_df = output[output["label"] == "real"]
        fake_df = output[output["label"] == "fake"]

        real_sample = real_df.sample(
            n=min(len(real_df), per_class),
            random_state=42,
        )

        fake_sample = fake_df.sample(
            n=min(len(fake_df), per_class),
            random_state=42,
        )

        output = pd.concat([real_sample, fake_sample])
        output = output.sample(frac=1, random_state=42).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    print("\nDataset préparé avec succès ✅")
    print(f"Entrée : {input_path}")
    print(f"Sortie : {output_path}")
    print("\nDistribution :")
    print(output["label"].value_counts())
    print("\nExemples :")
    print(output.head())


def main():
    parser = argparse.ArgumentParser(
        description="Préparer Fakeddit pour entraînement texte fake/real"
    )

    parser.add_argument("--input", type=str, required=True)

    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "data" / "processed" / "fakeddit_text.csv"),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
    )

    args = parser.parse_args()

    prepare_fakeddit(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()