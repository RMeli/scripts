"""
Search And Substitute (SAS)
"""

import re

from typing import Dict


def sas(content: str, pattern_and_repl: Dict[str, str]) -> str:

    sub_content = content
    for pattern, repl in pattern_and_repl.items():
        sub_content = re.sub(pattern, repl, sub_content)

    return sub_content


if __name__ == "__main__":
    pass
