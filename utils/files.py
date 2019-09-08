from pathlib import Path
from typing import Union

def get_files(path: Union[str, Path], extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))
