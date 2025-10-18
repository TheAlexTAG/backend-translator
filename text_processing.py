def process(input: list[str], capitalize_first: bool = False) -> list[str]:
    """
    Process a list of strings:
      - If capitalize_first=False: make all lowercase
      - If capitalize_first=True: capitalize only the first letter of each string
    """
    result = []
    for s in input:
        if not isinstance(s, str):
            continue  # skip non-strings
        s = s.strip()
        if capitalize_first:
            result.append(s.capitalize())
        else:
            result.append(s.lower())
    return result
