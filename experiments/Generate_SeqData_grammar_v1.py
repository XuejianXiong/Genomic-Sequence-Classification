import os
import random
import pandas as pd

random.seed(42)

# =========================
# DNA generator
# =========================

def random_dna(length: int) -> str:
    # slight GC bias (more realistic)
    return "".join(random.choices(
        population="ACGT",
        weights=[0.25, 0.25, 0.25, 0.25],
        k=length
    ))

# =========================
# insert WITHOUT overwriting
# =========================

def insert_motif(sequence: str, motif: str, pos: int) -> str:
    return sequence[:pos] + motif + sequence[pos:]


# =========================
# grammar generator (FIXED)
# =========================

def generate_grammar_sequence(
    seq_length: int,
    motif1: str,
    motif2: str,
    positive: bool
) -> str:

    seq = random_dna(seq_length)

    min_gap = 5
    max_gap_pos = 20
    min_gap_neg = 100
    max_gap_neg = 250

    if positive:
        distance = random.randint(min_gap, max_gap_pos)
    else:
        distance = random.randint(min_gap_neg, max_gap_neg)

    # ensure space for both motifs
    max_pos1 = seq_length - len(motif1) - len(motif2) - distance
    if max_pos1 <= 0:
        raise ValueError(
            f"Sequence too short: {seq_length}"
        )

    pos1 = random.randint(0, max_pos1)
    pos2 = pos1 + len(motif1) + distance

    # enforce strict boundary safety
    if pos2 + len(motif2) > seq_length:
        raise ValueError("Invalid placement (should not happen)")

    seq = insert_motif(seq, motif1, pos1)
    seq = insert_motif(seq, motif2, pos2)

    return seq


# =========================
# dataset generator
# =========================

def generate_dataset(n_samples: int = 2000):

    motif1 = "CACGTG"
    motif2 = "TGACTCA"

    sequences, labels = [], []

    for _ in range(n_samples // 2):
        length = random.randint(300, 500)
        sequences.append(
            generate_grammar_sequence(length, motif1, motif2, True)
        )
        labels.append("enhancer")

    for _ in range(n_samples // 2):
        length = random.randint(300, 500)
        sequences.append(
            generate_grammar_sequence(length, motif1, motif2, False)
        )
        labels.append("non-enhancer")

    df = pd.DataFrame({
        "sequence": sequences,
        "label": labels
    })

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================
# main
# =========================

def main():
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()

    outfile = "data/grammar_proper.csv"
    df.to_csv(outfile, index=False)

    print("Saved:", outfile)
    print(df["label"].value_counts())
    print(df.head())


if __name__ == "__main__":
    main()