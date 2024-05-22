import json
from typing import List
from entities import User

import os


def load_users(path_to_dir: str) -> List[User]:
    users = []

    path = os.path.join(os.getcwd(), path_to_dir)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r') as f:
            data = json.load(f)
            user = User.from_dict(data, filename)
            users.append(user)
    users.sort(key=lambda x: x.identifier)
    return users
