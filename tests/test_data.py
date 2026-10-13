from food_classifier.data import CLASSES, make_split


def test_split_matches_project_plan():
    names = [*CLASSES, "apple_pie"]
    official_train = {name: [f"{name}/train_{row:06d}" for row in range(750)] for name in names}
    official_test = {name: [f"{name}/test_{row:06d}" for row in range(250)] for name in names}

    split = make_split(official_train, official_test)

    assert {part: len(ids) for part, ids in split.items()} == {
        "train": 7200,
        "val": 1800,
        "test": 3000,
    }
    all_ids = [image_id for ids in split.values() for image_id in ids]
    assert len(set(all_ids)) == 12000
    assert not any(image_id.startswith("apple_pie/") for image_id in all_ids)
    assert make_split(official_train, official_test) == split
