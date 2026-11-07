from src.ingestion.raw_ingestion import read_source_data

def test_read_source_data_parses_fd001_text(tmp_path):
    train_path = tmp_path / "train_FD001.txt"
    train_path.write_text(
        "1  7  0.25  0.50  100.0  "
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21\n",
        encoding="utf-8",
    )

    test_path = tmp_path / "test_FD001.txt"
    test_path.write_text(
        "1  3  0.25  0.50  100.0  "
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21\n",
        encoding="utf-8",
    )

    RUL_path = tmp_path / "RUL_FD001.txt"
    RUL_path.write_text(
        "112\n98\n ",
        encoding="utf-8",
    )

    
    source_files = {
        "train_FD001.txt": train_path,
        "test_FD001.txt": test_path,
        "RUL_FD001.txt": RUL_path,
    }

    result = read_source_data(source_files)

    assert result["train_data"].loc[0, "time_in_cycles"] == 7
    assert result["test_data"].loc[0, "time_in_cycles"] == 3
    assert result["train_data"].loc[0,"sensor_measurement_21"] == 21.0
    assert result["RUL_data"]["unit_number"].tolist() == [1, 2]
    assert result["RUL_data"]["remaining_useful_life"].tolist() == [112, 98]
    assert result["train_data"].loc[0,"operational_setting_1"] == 0.25
    assert result["train_data"].shape == (1, 26)
    assert result["test_data"].shape == (1, 26)
    assert result["RUL_data"].shape == (2, 2)
