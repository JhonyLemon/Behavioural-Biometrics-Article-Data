import json
from typing import List

import entities
import os


def load_users(path_to_dir: str) -> List[entities.User]:
    users = []

    path = os.path.join(os.getcwd(), path_to_dir)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            data = json.load(f)
            user = entities.User.from_dict(data, filename)
            users.append(user)
    return users