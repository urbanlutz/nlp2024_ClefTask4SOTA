from src.dataset import UNANSWERABLE
import re
import json

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

def _information(text):
    return len(re.sub(r'\{|\}|\[|\]|"|\'|:|,| |\n|leaderboard|task|dataset|metric|score|null|none', "", text.lower()))

def _is_empty(text):
    return _information(text) < 10

def empty_to_unanswerable(text):
    if len(text) < 10 or _is_empty(text):
        return UNANSWERABLE
    else:
        return text

def fish_json(text):
    return "[" + text.split("[", 1)[-1].rsplit("]", 1)[0] + "]"

def remove_newline_tab(text):
    text = re.sub("\n|\t", "", text)
    text = re.sub(" +", " ", text)
    return text

def replace_quotes(text):
    return text.replace('"', "'")


def _add_LB_regex(text):
    if '"' in text:
        lb = '{"LEADERBOARD":{'
    else:
        lb = "{'LEADERBOARD':{"
    text = re.sub(r"\{", lb, text)
    text = re.sub(r"\}", "}}", text)
    return text

def _remove_LB_regex(text):
    text = re.sub(r'\{[ |\n|\t]*["|\']LEADERBOARD["|\']:[ |\n|\t]*{', "{", text)
    text = re.sub(r"\}[ |\n|\t]*\}", "}", text)
    return text

def _add_LB_json(text):
    parsed = json.loads(text)
    parsed = [{"LEADERBOARD": obj} for obj in parsed]
    return json.dumps(parsed)

def _remove_LB_json(text):
    parsed = json.loads(text)
    parsed = [d.get("LEADERBOARD", d) for d in parsed]
    return json.dumps(parsed)

def add_LEADERBOARD(text):
    try:
        text = _remove_LB_json(text)
        text = _add_LB_json(text)
    except:
        text = _remove_LB_regex(text)
        text = _add_LB_regex(text)
    finally:
        return text
    

def format(text):
    text = empty_to_unanswerable(text)
    if text == UNANSWERABLE or text == UNANSWERABLE.rstrip("\n"):
        return text
    text = fish_json(text)

    text = add_LEADERBOARD(text)
    # try to parse it, otherwise make a pseudo correct structure
    try:
        text = str(json.loads(text))
    except Exception as e:
        text = remove_newline_tab(text)
        text = replace_quotes(text)
    return text

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