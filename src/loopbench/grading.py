import re
def extract_number(text):
    matches=re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",text)
    if not matches: return None
    return float(matches[-1])
def grade(text, expected, tolerance=1e-6):
    actual=extract_number(text)
    if actual is None: return {"correct":False,"parse_status":"parse_failure","predicted":None}
    return {"correct":abs(actual-float(expected))<=tolerance,"parse_status":"parsed","predicted":actual}
