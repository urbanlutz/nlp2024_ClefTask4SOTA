import json


def get_file_content_as_string(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except IOError:
        return "Error reading file."


def model_name_existing(data, name_to_check):
    # Iterate through the models in the dictionary
    for model in data.get('models', []):
        # Check if the name matches the one we are looking for
        if model.get('name') == name_to_check:
            return True
    return False
