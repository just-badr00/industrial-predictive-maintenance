from pathlib import Path


def check_source_data(project_root):

    filename = ["train_FD001.txt" , "test_FD001.txt", "RUL_FD001.txt"]

    root = Path(project_root)
    check = {}
    for file in filename:

        # NEW: Check each file using its Path, without opening its contents.
        file_path = root / "CMAPSSData" / file

        if not file_path.is_file():
            raise FileNotFoundError(f"Required source data file '{file}' is missing. Check that the CMAPSSData folder contains the required files.")

        if file_path.stat().st_size == 0:
            raise ValueError(f"Source data file '{file}' is empty. Check that it contains valid data.")

        check[file] = str(file_path)

    # Return only after every required file has passed both checks.
    return check
    # END NEW: Source-file checks and returned paths.
