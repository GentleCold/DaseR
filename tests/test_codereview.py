import os

def test_codereview_dir_exists():
    assert os.path.exists("codereview"), "codereview directory does not exist"
    assert os.path.isdir("codereview"), "codereview is not a directory"
