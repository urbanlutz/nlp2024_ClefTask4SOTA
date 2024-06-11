import re
import json
from src.dataset import UNANSWERABLE

def _find_sections(tex):
    p = re.compile("\\\\section\\{(.+?)\\}(.*?)(?=\\\\section|$)", re.DOTALL)
    try:
        r = [(match.group(1), match.group(2)) for match in re.finditer(p, tex)]
        return r
    except:
        return []

def _find_unclosed(tex, name):
    """matches tags of the pattern \<name>{<content>}"""
    p = re.compile(f"\\\\{name}\\{{(.+?)\\}}", re.DOTALL)
    try:
        r = [(name, match.group(1)) for match in re.finditer(p, tex)]
        return r
    except:
        return []

name = "table"
def _find_begin_end(tex, name):
    """matches tags of the pattern \begin{<name>}<content>\end{<name>}"""
    p = re.compile(f"\\\\begin\\{{{name}\\}}(.*?)(?=\\\\end\\{{{name}\\}}|$)", re.DOTALL)
    try:
        r = [(name, match.group(1)) for match in re.finditer(p, tex)]
        return r
    except:
        return []

def all_sections(tex):
    return [
        *_find_unclosed(tex, "title"),
        *_find_begin_end(tex, "abstract"),
        *_find_begin_end(tex, "table"),
        *_find_sections(tex)
    ]

def naive_doctaet(tex):
    elements = [
        *_find_unclosed(tex, "title"),
        *_find_begin_end(tex, "abstract"),
        *_find_begin_end(tex, "table"),
        *[(k, v) for k, v in _find_sections(tex) if "experiment" in k.lower() or "result" in k.lower()]
    ]
    return '\n'.join([f"{name}\n{text}" for name, text in elements])



def _convert_tdms_to_tuple(model_output_parsed):
    tuples = []
    for item in model_output_parsed:
        try:
            t = ((item["Task"], item["Dataset"],item["Metric"],item["Score"]))
            tuples.append(t)
        except:
            # parse error, ignore instance
            pass
    return tuples

def _format_tdms(tuples):
    """make unique, format as string"""
    unique = set(tuples)
    dicts = [{"LEADERBOARD": {
        "Task": t,
        "Dataset":d,
        "Metric":m,
        "Score":s
    }} for t,d,m,s in unique]
    if dicts:
        return str(dicts)
    else:
        return UNANSWERABLE


def parse_response(response):
    try:
        response = response.replace("\\", "")
        response = "[" + response.split("[", 1)[-1].rsplit("]", 1)[0] + "]"
        response = json.loads(response)
        response = _convert_tdms_to_tuple(response)
        return _format_tdms(response)
    except Exception as ex:
        print(ex)
        print(str(response))
        return str(response)