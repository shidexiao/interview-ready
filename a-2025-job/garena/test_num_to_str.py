from num_to_str import convert_ints_to_str, Node


def test_single_int() -> None:
    obj: Node = 123
    expected: Node = "123"
    assert convert_ints_to_str(obj) == expected

def test_single_str() -> None:
    obj: Node = "hello"
    expected: Node = "hello"
    assert convert_ints_to_str(obj) == expected

def test_list_of_ints() -> None:
    obj: Node = [1, 2, 3]
    expected: Node = ["1", "2", "3"]
    assert convert_ints_to_str(obj) == expected

def test_dict_of_int_keys_and_list_values()-> None:
    obj: Node = {1: [1, 2], 3: 4}
    expected: Node = {"1": ["1", "2"], "3": "4"}
    assert convert_ints_to_str(obj) == expected


def test_dict_key_is_str_value_is_list_dict()-> None:
    obj: Node = {
        "a": [1, 2],
        "b": {"c": 3}
    }
    expected: Node = {
        "a": ["1", "2"],
        "b": {"c": "3"}
    }
    assert convert_ints_to_str(obj) == expected


def test_nested_list_and_dict()-> None:
    obj: Node = [
        1,
        "hello",
        {2: [3, 4], "nested": {5: 6}},
        ["deep", {7: "end"}]
    ]
    expected: Node = [
        "1",
        "hello",
        {"2": ["3", "4"], "nested": {"5": "6"}},
        ["deep", {"7": "end"}]
    ]
    assert convert_ints_to_str(obj) == expected
