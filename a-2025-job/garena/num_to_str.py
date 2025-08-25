from typing import Union, List, Dict

Node = Union[int, str, List['Node'], Dict['Node', 'Node']]


def convert_ints_to_str(obj: Node) -> Node:
    if isinstance(obj, int):
        return str(obj)
    elif isinstance(obj, list):
        return [convert_ints_to_str(item) for item in obj]
    elif isinstance(obj, dict):
        return {convert_ints_to_str(key): convert_ints_to_str(value) for key, value in obj.items()}
    else:
        return obj


if __name__ == "__main__":
    obj:Node = {1: [1, 2, 3]}
    res = convert_ints_to_str(obj)
    print(res)