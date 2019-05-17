import json


def load_json(fname: str):

    with open(fname, "r") as f:

        data = json.load(f)

    return data


def load_file(fname: str) -> str:

    with open(fname, "r") as f:

        content = f.read()

    return content
