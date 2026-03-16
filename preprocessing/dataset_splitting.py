from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def _build_video_key(frame_df: pd.DataFrame, video_col: str, category_col: str) -> pd.Series:
    return (
        frame_df[category_col].astype("string")
        + "::"
        + frame_df[video_col].astype("string")
    )


def make_video_level_split(
    csv_path: str,
    output_csv: str,
    video_col: str = "video_name",
    category_col: str = "category",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    df = pd.read_csv(csv_path, low_memory=False)

    if video_col not in df.columns:
        raise ValueError(f"Missing column: {video_col}")
    if category_col not in df.columns:
        raise ValueError(f"Missing column: {category_col}")

    # tạo binary label
    df["binary_label"] = df[category_col].apply(
        lambda x: "real" if str(x).lower() == "original" else "fake"
    )
    df["video_key"] = _build_video_key(df, video_col=video_col, category_col=category_col)

    # 1 dòng đại diện cho 1 video
    video_df = (
        df[["video_key", video_col, category_col, "binary_label"]]
        .drop_duplicates(subset=["video_key"])
        .reset_index(drop=True)
    )

    # split theo video, stratify theo category
    train_videos, temp_videos = train_test_split(
        video_df,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=video_df[category_col],
    )

    val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)

    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=(1 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_videos[category_col],
    )

    train_set = set(train_videos["video_key"])
    val_set = set(val_videos["video_key"])
    test_set = set(test_videos["video_key"])

    def assign_split(video_key: str):
        if video_key in train_set:
            return "train"
        if video_key in val_set:
            return "val"
        if video_key in test_set:
            return "test"
        raise RuntimeError(f"Video key {video_key} not assigned")

    df["split"] = df["video_key"].map(assign_split)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["video_key"]).to_csv(output_csv, index=False)

    print("Saved to:", output_csv)
    print("\nFrame count by split:")
    print(df["split"].value_counts())

    print("\nFrame count by split x category:")
    print(pd.crosstab(df["split"], df[category_col]))

    print("\nFrame count by split x binary_label:")
    print(pd.crosstab(df["split"], df["binary_label"]))

    print("\nUnique video count by split:")
    tmp = df[["video_key", category_col, video_col, "split"]].drop_duplicates()
    print(tmp["split"].value_counts())

    print("\nUnique video count by split x category:")
    print(pd.crosstab(tmp["split"], tmp[category_col]))


if __name__ == "__main__":
    make_video_level_split(
        csv_path="frame_data/frame_extraction_metadata.csv",
        output_csv="artifacts/split_metadata.csv",
        video_col="video_name",
        category_col="category",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )