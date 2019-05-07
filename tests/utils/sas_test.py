from scripts.utils.sas import sas

import scripts.utils._io as io

import os

path = os.path.abspath("tests/utils/data/")

def test_sas_simple():

    json_fname = os.path.join(path, "simple.json")

    data = io.load_json(json_fname)

    sub_fname = os.path.join(path, "simple.txt")

    content = io.load_file(sub_fname)

    changed = sas(content, data)