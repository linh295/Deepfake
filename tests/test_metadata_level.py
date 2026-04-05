from __future__ import annotations

import unittest

from preprocessing import metadata_level as ml


class MetadataLevelSplitTest(unittest.TestCase):
    def test_assign_splits_balances_each_category_independently(self) -> None:
        records: list[dict[str, object]] = []
        for category in ("original", "Deepfakes"):
            for idx in range(10):
                records.append(
                    {
                        "category": category,
                        "_split_group": f"{category}-group-{idx}",
                    }
                )

        split_by_group = ml.assign_splits(records)

        for category in ("original", "Deepfakes"):
            counts = {"train": 0, "val": 0, "test": 0}
            for idx in range(10):
                split = split_by_group[f"{category}-group-{idx}"]
                counts[split] += 1
            self.assertEqual(counts, {"train": 8, "val": 1, "test": 1})

    def test_assign_splits_keeps_group_members_together(self) -> None:
        records: list[dict[str, object]] = []
        for idx in range(5):
            group_key = f"DeepFakeDetection-group-{idx}"
            records.append({"category": "DeepFakeDetection", "_split_group": group_key, "video_id": f"{idx}a"})
            records.append({"category": "DeepFakeDetection", "_split_group": group_key, "video_id": f"{idx}b"})

        split_by_group = ml.assign_splits(records)

        assigned = {split_by_group[f"DeepFakeDetection-group-{idx}"] for idx in range(5)}
        self.assertTrue(assigned.issubset({"train", "val", "test"}))
        self.assertEqual(len(split_by_group), 5)


if __name__ == "__main__":
    unittest.main()
