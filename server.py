# Imports various Python modules and custom modules used in a Gradio-based chatbot application.
# Sets up the logging configuration, disables Gradio's analytics tracking and welcome message, and ignores specific warnings.
# Additionally, replaces the `requests.get` method with a custom function to prevent Gradio from phoning home when it gets imported.
# Finally, it defines several functions related to text generation, loading and unloading AI models, applying extensions, and wrapping chat in HTML.
import logging
import os
import requests
import warnings
# A custom module for colored logging
import modules.logging_colors

# Disable Gradio's analytics tracking
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
# Disable welcome message
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# Sett up basic configuration for logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# This is a hack to prevent Gradio from phoning home when it gets imported
def my_get(url, **kwargs):
    # Log that Gradio's request was redirected
    logging.info('Gradio HTTP request redirected to localhost :)')
    # Ensure that redirects are allowed
    kwargs.setdefault('allow_redirects', True)
    # Redirect the request to localhost
    return requests.api.request('get', 'http://127.0.0.1/', **kwargs)

# Save the original 'get' method of requests
original_get = requests.get
# Replace 'get' method with the new function
requests.get = my_get
# Importing Gradio after the 'get' method has been replaced
import gradio as gr
# Restoring the original 'get' method
requests.get = original_get

import matplotlib
matplotlib.use('Agg')  # This fixes LaTeX rendering on some systems

import importlib # Provides the implementation of the import statement
import io # Tools for working with streams (Input/Output data)
import json # Used to work with JSON data
import math # Provides mathematical functions
import os # Provides functions for interacting with the operating system
import re # Provides regular expression matching operations
import sys # Provides access to some variables used or maintained by the Python interpreter
import time # Provides various time-related functions
import traceback # Used for printing stack traces of Python programs
import zipfile # Used to work with ZIP archives
from datetime import datetime # Used to work with dates and times
from functools import partial # Used for partial function application which fixes some function arguments
from pathlib import Path # Object-oriented filesystem paths

import psutil # Cross-platform library for process and system monitoring
import torch # PyTorch, a deep learning framework
import yaml # Used to work with YAML data
from PIL import Image # Used to open, manipulate, and save different image file formats

import modules.extensions as extensions_module # Custom module for extensions
from modules import chat, shared, training, ui, utils # Importing other custom modules
from modules.extensions import apply_extensions # Function to apply extensions
from modules.html_generator import chat_html_wrapper # Function to wrap chat in HTML
from modules.LoRA import add_lora_to_model # Function to add LoRA (an AI model) to another model
from modules.models import load_model, load_soft_prompt, unload_model # Functions to load and unload AI models
from modules.text_generation import generate_reply_wrapper, get_encoded_length, stop_everything_event # Functions related to text generation
from typing import Generator


# This code defines a function called `load_model_wrapper` that takes two arguments: `selected_model` and `autoload`.
# The `selected_model` argument is a string that represents the name of the model to be loaded. The `autoload`
# argument is a boolean that indicates whether the model should be loaded automatically or not.
# The function first checks if `autoload` is False. If it is, the function yields a message indicating that the settings for the
# selected model have been updated, but the model has not been loaded yet.
# If `autoload` is True, the function checks if `selected_model` is equal to the string "None". If it is, the function yields a
# message indicating that no model has been selected.
# If `selected_model` is not equal to "None", the function attempts to load the selected model by calling the `load_model`
# function with the `shared.model_name` argument. If the model is successfully loaded, the function yields a message indicating that the
# model has been loaded successfully.
# If an exception occurs during the loading of the model, the function yields the traceback of the exception.
# The `shared` object is likely a global object that is shared between different parts of the code. The `unload_model` function is
# likely a function that unloads any currently loaded model. The `load_model` function is likely a function that loads a
# model and its associated tokenizer.
def load_model_wrapper(selected_model, autoload=False):
    # If autoload is False, yield a message stating that the settings have been updated and prompt the user to load the model
    if not autoload:
        yield f"The settings for {selected_model} have been updated.\nClick on \"Load the model\" to load it."
        return

    # If no model is selected, yield a message stating that no model has been selected
    if selected_model == 'None':
        yield "No model selected"
    else:
        try:
            # If a model is selected, yield a message stating that it is being loaded
            yield f"Loading {selected_model}..."
            # Set the shared model name to the selected model
            shared.model_name = selected_model
            # Unload any previously loaded model
            unload_model()
            # If the selected model is not an empty string, load the model and tokenizer and set them to the shared variables
            if selected_model != '':
                shared.model, shared.tokenizer = load_model(shared.model_name)
            # Yield a message stating that the model has been successfully loaded
            yield f"Successfully loaded {selected_model}"
        except:
            # If an error occurs during the loading process, yield the traceback of the error
            yield traceback.format_exc()

# Define a function called `load_lora_wrapper` that takes one argument `selected_loras`, which is a list of LoRAs
# (Layers of Reasoning Algorithms) to be applied to the loaded model.
#
# The function first yields a message indicating which LoRAs are being applied to the loaded model.
# It then calls the `add_lora_to_model` function with the `selected_loras` argument to apply the selected LoRAs to the model.
# Finally, the function yields a message indicating that the LoRAs have been successfully applied.
#
# The output of the function will depend on the selected LoRAs and whether a model has been loaded.
# If a model has not been loaded, the function will yield a message indicating that no model has been loaded.
# If a model has been loaded and the LoRAs are successfully applied, the function will yield a message indicating which LoRAs
# have been applied to the model.
# If an error occurs during the application of the LoRAs, the function will yield an error message.
def load_lora_wrapper(selected_loras):
    # Check if a model has been loaded
    if shared.model is None:
        yield "No model has been loaded"
        return

    # Apply the selected LoRAs to the loaded model
    try:
        add_lora_to_model(selected_loras)
        yield f"Successfully applied the following LoRAs to {shared.model_name}:\n\n{', '.join(selected_loras)}"
    except:
        yield "Error applying the LoRAs"

# Function called `load_preset_values` takes in three arguments `preset_menu`, `state`, and `return_dict`.
# The function reads in a file located in the `presets` directory with the name specified by `preset_menu`, and
# extracts key-value pairs from it. The keys are used to update a dictionary called `generate_params`,
# which contains default values for various parameters used in a text generation model.
# The function then updates the `generate_params` dictionary with the extracted values,
# and returns either the updated `state` dictionary and the updated parameter values,
# or just the updated parameter values if `return_dict` is set to `True`.
# The function also includes some logic to ensure that the `temperature` parameter is capped at a maximum value of 1.99.
def load_preset_values(preset_menu, state, return_dict=False):
    generate_params = {
        'do_sample': True,
        'temperature': 1,
        'top_p': 1,
        'typical_p': 1,
        'repetition_penalty': 1,
        'encoder_repetition_penalty': 1,
        'top_k': 50,
        'num_beams': 1,
        'penalty_alpha': 0,
        'min_length': 0,
        'length_penalty': 1,
        'no_repeat_ngram_size': 0,
        'early_stopping': False,
    }

    with open(f'presets/{preset_menu}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=')
            key = key.strip()
            value = eval(value.strip())
            if key in generate_params:
                generate_params[key] = value

    generate_params['temperature'] = min(1.99, generate_params['temperature'])

    if return_dict:
        return generate_params
    else:
        state.update(generate_params)
        return state, *[generate_params[k] for k in generate_params]

# Takes a file object as input, which is expected to be a ZIP file containing a `meta.json` file and a `prompt.txt` file.
# Extracts the `meta.json` file from the ZIP file, loads its contents, and gets the name of the soft prompt from the `name` field in the JSON data.
# Then saves the uploaded ZIP file to the `softprompts` directory with the name specified in the `meta.json` file.
# Finally, the function returns the name specified in the `meta.json` file.
#
# The output of this function is the name of the soft prompt specified in the `meta.json` file.
# This name is used to load the soft prompt later on.
def upload_soft_prompt(file):
    # Extract the meta.json file from the uploaded ZIP file and load its contents
    with zipfile.ZipFile(io.BytesIO(file)) as zf:
        zf.extract('meta.json')
        j = json.loads(open('meta.json', 'r').read())
        name = j['name']
        # Remove the meta.json file
        Path('meta.json').unlink()

    # Save the uploaded ZIP file to the softprompts directory with the name specified in the meta.json file
    with open(Path(f'softprompts/{name}.zip'), 'wb') as f:
        f.write(file)

    # Return the name specified in the meta.json file
    return name

# This code defines a function called `open_save_prompt` that opens a prompt for the user to enter a
# filename to save the chat history.
#
# The function first generates a filename based on the current date and time using the `datetime.now().strftime` method.
# It then returns three Gradio update objects: one to update the value of the prompt with the generated
# filename, and two to show and hide the prompt, respectively.
def open_save_prompt():
    # Generate a filename based on the current date and time
    fname = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    # If the generated filename is empty, use a default filename
    if not fname:
        fname = "chat_history.txt"
    # Return three Gradio update objects to show and hide the prompt and update its value with the generated filename
    return gr.update(value=fname, visible=True), gr.update(visible=False), gr.update(visible=True)


def save_prompt(text, fname):
    if fname != "":
        with open(Path(f'prompts/{fname}.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        message = f"Saved to prompts/{fname}.txt"
    else:
        message = "Error: No prompt name given."

    return message, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

# The function begins with a check to see if fname is empty or None (if not fname).
# If fname is empty, the function returns an empty string ('').
# The code then checks if the fname starts with the prefix 'Instruct-' (if fname.startswith('Instruct-')).
# If it does, the prefix is removed using regular expression substitution (re.sub('^Instruct-', '', fname)).
# The resulting file name is expected to match a YAML file in the 'characters/instruction-following/' directory.
# If the file name starts with 'Instruct-', the code opens the corresponding YAML file in read mode ('r') with UTF-8
# encoding ('utf-8') using a context manager (with open(...) as f).
# The file content is then loaded into the data variable using the yaml.safe_load function.
# Next, an empty string named output is initialized.
# If the data dictionary has a key 'context', its value is appended to the output variable (output += data['context']).
# The code defines a dictionary named replacements that holds various replacement strings.
# These replacements are used to modify the data['turn_template'] string.
# The code then replaces specific substrings in data['turn_template'] using the utils.replace_all function and the replacements dictionary.
# The resulting string is added to the output variable.
# The replacements involve replacing an empty string with data['user'],
# an empty string with data['bot'],
# and <|user-message|> with the string 'Input'.
# Finally, the function returns the output string after stripping any leading or trailing whitespace (output.strip()).
def load_prompt(fname):
    if not fname:
        return ''

    if fname.startswith('Instruct-'):
        fname = re.sub('^Instruct-', '', fname)
        with open(f'characters/instruction-following/{fname}.yaml', 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            output = ''
            if 'context' in data:
                output += data['context']

            replacements = {
                '<|user|>': data['user'],
                '<|bot|>': data['bot'],
                '<|user-message|>': 'Input',
            }

            output += utils.replace_all(data['turn_template'].split('<|bot-message|>')[0], replacements)
            return output.strip()
    else:
        with open(f'prompts/{fname}.txt', 'r', encoding='utf-8') as f:
            text = f.read().strip()
            return text

# This function takes a string `text` as input and returns the number of tokens in the input.
# It does so by splitting the input string into tokens using whitespace as the delimiter and
# then counting the number of resulting tokens.
# The function then returns a string that includes the number of tokens in the input.
def count_tokens(text):
    tokens = len(text.split())
    return f'{tokens} tokens in the input.'


# The function first imports the `download-model` module and splits the `repo_id` string into `model` and `branch` parts.
# It then calls the `sanitize_model_and_branch_names` function from the `download-model` module to clean up the `model` and `branch` names.
# Next, the function calls the `get_download_links_from_huggingface` function from the `download-model`
# module to get the download links for the specified `model` and `branch` from the Hugging Face model hub.
# It also retrieves the SHA256 hash of the downloaded files and a boolean flag indicating whether the model is a LoRA model.
# The function then calls the `get_output_folder` function from the `download-model` module to get the output
# folder where the downloaded files will be stored. If the `check` flag is set to `True`, the
# function calls the `check_model_files` function from the `download-model` module to check if the files have
# already been downloaded. If the `check` flag is set to `False`, the function calls the
# `download_model_files` function from the `download-model` module to download the files to the output folder.
# Finally, the function yields a string indicating that the download process has completed
# or, if an exception occurs, a traceback of the exception.
def download_model_wrapper(repo_id: str) -> Generator[str, None, None]:
    try:
        downloader = importlib.import_module("download-model")
        repo_id_parts = repo_id.split(":")
        model = repo_id_parts[0] if len(repo_id_parts) > 0 else repo_id
        branch = repo_id_parts[1] if len(repo_id_parts) > 1 else "main"
        check = False

        yield "Cleaning up the model/branch names"
        model, branch = downloader.sanitize_model_and_branch_names(model, branch)

        yield "Getting the download links from Hugging Face"
        links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch, text_only=False)

        yield "Getting the output folder"
        output_folder = downloader.get_output_folder(model, branch, is_lora)

        if check:
            yield "Checking previously downloaded files"
            downloader.check_model_files(model, branch, links, sha256, output_folder)
        else:
            yield f"Downloading files to {output_folder}"
            downloader.download_model_files(model, branch, links, sha256, output_folder, threads=1)
            yield "Download completed!"
    except Exception as e:
        yield traceback.format_exc()


# The function first gets a list of model parameter names using the `ui.list_model_elements()` function.
# It then loops through each parameter name and checks if it exists in the `state` dictionary.
# If it does, the function checks if the parameter is a GPU memory parameter and adds its value to a list of GPU memory values.
# If the parameter is not a GPU memory parameter, the function checks if it is an initial parameter and if its
# value is different from the default value.
# If it is, the function skips the parameter.
# If the parameter is not a GPU memory parameter and its value is `None` or `0`, the function sets its value to the default value.
# The function also makes some simple conversions for certain parameter types.
# Finally, the function sets the value of each parameter using the `setattr()` function. After looping
# through all the parameters, the function checks if any of the GPU memory values are positive.
# If there are any positive GPU memory values, the function sets the `gpu_memory` parameter to a list of strings
# representing the GPU memory values in MiB. If there are no positive GPU memory values, the function sets the `gpu_memory` parameter to `None`.
def update_model_parameters(state, initial=False):
    elements = ui.list_model_elements()  # the names of the parameters
    gpu_memories = []

    for i, element in enumerate(elements):
        if element not in state:
            continue

        value = state[element]
        if element.startswith('gpu_memory'):
            gpu_memories.append(value)
            continue

        if initial and vars(shared.args)[element] != vars(shared.args_defaults)[element]:
            continue

        # Setting null defaults
        if element in ['wbits', 'groupsize', 'model_type'] and value == 'None':
            value = vars(shared.args_defaults)[element]
        elif element in ['cpu_memory'] and value == 0:
            value = vars(shared.args_defaults)[element]

        # Making some simple conversions
        if element in ['wbits', 'groupsize', 'pre_layer']:
            value = int(value)
        elif element == 'cpu_memory' and value is not None:
            value = f"{value}MiB"

        if element in ['pre_layer']:
            value = [value] if value > 0 else None

        setattr(shared.args, element, value)

    found_positive = False
    for i in gpu_memories:
        if i > 0:
            found_positive = True
            break

    if not (initial and vars(shared.args)['gpu_memory'] != vars(shared.args_defaults)['gpu_memory']):
        if found_positive:
            shared.args.gpu_memory = [f"{i}MiB" for i in gpu_memories]
        else:
            shared.args.gpu_memory = None

# The function first initializes a dictionary called `settings` with some pre-defined configuration values.
# It then initializes an empty dictionary called `model_settings`.
# The function then loops through each pattern in the `settings` dictionary and
# checks if the lowercase version of the `model` string matches the pattern using regular expressions.
# If there is a match, the function adds the configuration values for that pattern to the `model_settings` dictionary.
# Finally, the function returns the `model_settings` dictionary, which contains the configuration values specific to the input `model`.
def get_model_specific_settings(model):
    settings = shared.model_config
    model_settings = {}

    for pat in settings:
        if re.match(pat.lower(), model.lower()):
            for k in settings[pat]:
                model_settings[k] = settings[pat][k]

    return model_settings

# The function first calls another function called `get_model_specific_settings` to get a dictionary of settings
# specific to the given `model`. It then iterates over the keys in this dictionary and checks if each key
# is present in the `state` dictionary. If a key is present in both dictionaries, the value in the `state`
# dictionary is replaced with the value from the `model_settings` dictionary.
# Finally, the function returns the modified `state` dictionary (or a new dictionary if `return_dict` is `True`).
def load_model_specific_settings(model, state, return_dict=False):
    model_settings = get_model_specific_settings(model)
    for k in model_settings:
        if k in state:
            state[k] = model_settings[k]

    return state


def save_model_settings(model, state):
    if model == 'None':
        yield ("Not saving the settings because no model is loaded.")
        return

    with Path(f'{shared.args.model_dir}/config-user.yaml') as p:
        if p.exists():
            user_config = yaml.safe_load(open(p, 'r').read())
        else:
            user_config = {}

        model_regex = model + '$'  # For exact matches
        if model_regex not in user_config:
            user_config[model_regex] = {}

        for k in ui.list_model_elements():
            user_config[model_regex][k] = state[k]

        with open(p, 'w') as f:
            f.write(yaml.dump(user_config))

        yield (f"Settings for {model} saved to {p}")


def create_model_menus():
    # Finding the default values for the GPU and CPU memories
    total_mem = []
    for i in range(torch.cuda.device_count()):
        total_mem.append(math.floor(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)))

    default_gpu_mem = []
    if shared.args.gpu_memory is not None and len(shared.args.gpu_memory) > 0:
        for i in shared.args.gpu_memory:
            if 'mib' in i.lower():
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)))
            else:
                default_gpu_mem.append(int(re.sub('[a-zA-Z ]', '', i)) * 1000)
    while len(default_gpu_mem) < len(total_mem):
        default_gpu_mem.append(0)

    total_cpu_mem = math.floor(psutil.virtual_memory().total / (1024 * 1024))
    if shared.args.cpu_memory is not None:
        default_cpu_mem = re.sub('[a-zA-Z ]', '', shared.args.cpu_memory)
    else:
        default_cpu_mem = 0

    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        shared.gradio['model_menu'] = gr.Dropdown(choices=utils.get_available_models(), value=shared.model_name, label='Model')
                        ui.create_refresh_button(shared.gradio['model_menu'], lambda: None, lambda: {'choices': utils.get_available_models()}, 'refresh-button')

                with gr.Column():
                    with gr.Row():
                        shared.gradio['lora_menu'] = gr.Dropdown(multiselect=True, choices=utils.get_available_loras(), value=shared.lora_names, label='LoRA(s)')
                        ui.create_refresh_button(shared.gradio['lora_menu'], lambda: None, lambda: {'choices': utils.get_available_loras(), 'value': shared.lora_names}, 'refresh-button')

        with gr.Column():
            with gr.Row():
                shared.gradio['lora_menu_apply'] = gr.Button(value='Apply the selected LoRAs')
            with gr.Row():
                load = gr.Button("Load the model", visible=not shared.settings['autoload_model'])
                unload = gr.Button("Unload the model")
                reload = gr.Button("Reload the model")
                save_settings = gr.Button("Save settings for this model")

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Transformers parameters')
                with gr.Row():
                    with gr.Column():
                        for i in range(len(total_mem)):
                            shared.gradio[f'gpu_memory_{i}'] = gr.Slider(label=f"gpu-memory in MiB for device :{i}", maximum=total_mem[i], value=default_gpu_mem[i])
                        shared.gradio['cpu_memory'] = gr.Slider(label="cpu-memory in MiB", maximum=total_cpu_mem, value=default_cpu_mem)

                    with gr.Column():
                        shared.gradio['auto_devices'] = gr.Checkbox(label="auto-devices", value=shared.args.auto_devices)
                        shared.gradio['disk'] = gr.Checkbox(label="disk", value=shared.args.disk)
                        shared.gradio['cpu'] = gr.Checkbox(label="cpu", value=shared.args.cpu)
                        shared.gradio['bf16'] = gr.Checkbox(label="bf16", value=shared.args.bf16)
                        shared.gradio['load_in_8bit'] = gr.Checkbox(label="load-in-8bit", value=shared.args.load_in_8bit)

        with gr.Column():
            with gr.Box():
                gr.Markdown('GPTQ parameters')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['wbits'] = gr.Dropdown(label="wbits", choices=["None", 1, 2, 3, 4, 8], value=shared.args.wbits if shared.args.wbits > 0 else "None")
                        shared.gradio['groupsize'] = gr.Dropdown(label="groupsize", choices=["None", 32, 64, 128, 1024], value=shared.args.groupsize if shared.args.groupsize > 0 else "None")

                    with gr.Column():
                        shared.gradio['model_type'] = gr.Dropdown(label="model_type", choices=["None", "llama", "opt", "gptj"], value=shared.args.model_type or "None")
                        shared.gradio['pre_layer'] = gr.Slider(label="pre_layer", minimum=0, maximum=100, value=shared.args.pre_layer[0] if shared.args.pre_layer is not None else 0)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['autoload_model'] = gr.Checkbox(value=shared.settings['autoload_model'], label='Autoload the model', info='Whether to load the model as soon as it is selected in the Model dropdown.')

            shared.gradio['custom_model_menu'] = gr.Textbox(label="Download custom model or LoRA", info="Enter the Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main")
            shared.gradio['download_model_button'] = gr.Button("Download")

        with gr.Column():
            with gr.Box():
                gr.Markdown('llama.cpp parameters')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['threads'] = gr.Slider(label="threads", minimum=0, step=1, maximum=32, value=shared.args.threads)
                        shared.gradio['n_batch'] = gr.Slider(label="n_batch", minimum=1, maximum=2048, value=shared.args.n_batch)
                        shared.gradio['n_gpu_layers'] = gr.Slider(label="n-gpu-layers", minimum=0, maximum=128, value=shared.args.n_gpu_layers)

                    with gr.Column():
                        shared.gradio['no_mmap'] = gr.Checkbox(label="no-mmap", value=shared.args.no_mmap)
                        shared.gradio['mlock'] = gr.Checkbox(label="mlock", value=shared.args.mlock)

            with gr.Row():
                shared.gradio['model_status'] = gr.Markdown('No model is loaded' if shared.model_name == 'None' else 'Ready')

    # In this event handler, the interface state is read and updated
    # with the model defaults (if any), and then the model is loaded
    # unless "autoload_model" is unchecked
    shared.gradio['model_menu'].change(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        load_model_specific_settings, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['interface_state']).then(
        ui.apply_interface_values, shared.gradio['interface_state'], [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        load_model_wrapper, [shared.gradio[k] for k in ['model_menu', 'autoload_model']], shared.gradio['model_status'], show_progress=False)

    load.click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        partial(load_model_wrapper, autoload=True), shared.gradio['model_menu'], shared.gradio['model_status'], show_progress=False)

    unload.click(
        unload_model, None, None).then(
        lambda: "Model unloaded", None, shared.gradio['model_status'])

    reload.click(
        unload_model, None, None).then(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        update_model_parameters, shared.gradio['interface_state'], None).then(
        partial(load_model_wrapper, autoload=True), shared.gradio['model_menu'], shared.gradio['model_status'], show_progress=False)

    save_settings.click(
        ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
        save_model_settings, [shared.gradio[k] for k in ['model_menu', 'interface_state']], shared.gradio['model_status'], show_progress=False)

    shared.gradio['lora_menu_apply'].click(load_lora_wrapper, shared.gradio['lora_menu'], shared.gradio['model_status'], show_progress=False)
    shared.gradio['download_model_button'].click(download_model_wrapper, shared.gradio['custom_model_menu'], shared.gradio['model_status'], show_progress=False)
    shared.gradio['autoload_model'].change(lambda x: gr.update(visible=not x), shared.gradio['autoload_model'], load)


def create_settings_menus(default_preset):

    generate_params = load_preset_values(default_preset if not shared.args.flexgen else 'Naive', {}, return_dict=True)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                shared.gradio['preset_menu'] = gr.Dropdown(choices=utils.get_available_presets(), value=default_preset if not shared.args.flexgen else 'Naive', label='Generation parameters preset')
                ui.create_refresh_button(shared.gradio['preset_menu'], lambda: None, lambda: {'choices': utils.get_available_presets()}, 'refresh-button')
        with gr.Column():
            shared.gradio['seed'] = gr.Number(value=shared.settings['seed'], label='Seed (-1 for random)')

    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('Custom generation parameters ([click here to view technical documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig))')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['temperature'] = gr.Slider(0.01, 1.99, value=generate_params['temperature'], step=0.01, label='temperature', info='Primary factor to control randomness of outputs. 0 = deterministic (only the most likely token is used). Higher value = more randomness.')
                        shared.gradio['top_p'] = gr.Slider(0.0, 1.0, value=generate_params['top_p'], step=0.01, label='top_p', info='If not set to 1, select tokens with probabilities adding up to less than this number. Higher value = higher range of possible random results.')
                        shared.gradio['top_k'] = gr.Slider(0, 200, value=generate_params['top_k'], step=1, label='top_k', info='Similar to top_p, but select instead only the top_k most likely tokens. Higher value = higher range of possible random results.')
                        shared.gradio['typical_p'] = gr.Slider(0.0, 1.0, value=generate_params['typical_p'], step=0.01, label='typical_p', info='If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.')
                    with gr.Column():
                        shared.gradio['repetition_penalty'] = gr.Slider(1.0, 1.5, value=generate_params['repetition_penalty'], step=0.01, label='repetition_penalty', info='Exponential penalty factor for repeating prior tokens. 1 means no penalty, higher value = less repetition, lower value = more repetition.')
                        shared.gradio['encoder_repetition_penalty'] = gr.Slider(0.8, 1.5, value=generate_params['encoder_repetition_penalty'], step=0.01, label='encoder_repetition_penalty', info='Also known as the "Hallucinations filter". Used to penalize tokens that are *not* in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.')
                        shared.gradio['no_repeat_ngram_size'] = gr.Slider(0, 20, step=1, value=generate_params['no_repeat_ngram_size'], label='no_repeat_ngram_size', info='If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.')
                        shared.gradio['min_length'] = gr.Slider(0, 2000, step=1, value=generate_params['min_length'], label='min_length', info='Minimum generation length in tokens.')
                shared.gradio['do_sample'] = gr.Checkbox(value=generate_params['do_sample'], label='do_sample')
        with gr.Column():
            with gr.Box():
                gr.Markdown('Contrastive search')
                shared.gradio['penalty_alpha'] = gr.Slider(0, 5, value=generate_params['penalty_alpha'], label='penalty_alpha')

                gr.Markdown('Beam search (uses a lot of VRAM)')
                with gr.Row():
                    with gr.Column():
                        shared.gradio['num_beams'] = gr.Slider(1, 20, step=1, value=generate_params['num_beams'], label='num_beams')
                        shared.gradio['length_penalty'] = gr.Slider(-5, 5, value=generate_params['length_penalty'], label='length_penalty')
                    with gr.Column():
                        shared.gradio['early_stopping'] = gr.Checkbox(value=generate_params['early_stopping'], label='early_stopping')

            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        shared.gradio['truncation_length'] = gr.Slider(value=shared.settings['truncation_length'], minimum=shared.settings['truncation_length_min'], maximum=shared.settings['truncation_length_max'], step=1, label='Truncate the prompt up to this length', info='The leftmost tokens are removed if the prompt exceeds this length. Most models require this to be at most 2048.')
                        shared.gradio['custom_stopping_strings'] = gr.Textbox(lines=1, value=shared.settings["custom_stopping_strings"] or None, label='Custom stopping strings', info='In addition to the defaults. Written between "" and separated by commas. For instance: "\\nYour Assistant:", "\\nThe assistant:"')
                    with gr.Column():
                        shared.gradio['ban_eos_token'] = gr.Checkbox(value=shared.settings['ban_eos_token'], label='Ban the eos_token', info='Forces the model to never end the generation prematurely.')
                        shared.gradio['add_bos_token'] = gr.Checkbox(value=shared.settings['add_bos_token'], label='Add the bos_token to the beginning of prompts', info='Disabling this can make the replies more creative.')

                        shared.gradio['skip_special_tokens'] = gr.Checkbox(value=shared.settings['skip_special_tokens'], label='Skip special tokens', info='Some specific models need this unset.')
                        shared.gradio['stream'] = gr.Checkbox(value=not shared.args.no_stream, label='Activate text streaming')

    with gr.Accordion('Soft prompt', open=False):
        with gr.Row():
            shared.gradio['softprompts_menu'] = gr.Dropdown(choices=utils.get_available_softprompts(), value='None', label='Soft prompt')
            ui.create_refresh_button(shared.gradio['softprompts_menu'], lambda: None, lambda: {'choices': utils.get_available_softprompts()}, 'refresh-button')

        gr.Markdown('Upload a soft prompt (.zip format):')
        with gr.Row():
            shared.gradio['upload_softprompt'] = gr.File(type='binary', file_types=['.zip'])

    shared.gradio['preset_menu'].change(load_preset_values, [shared.gradio[k] for k in ['preset_menu', 'interface_state']], [shared.gradio[k] for k in ['interface_state', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']])
    shared.gradio['softprompts_menu'].change(load_soft_prompt, shared.gradio['softprompts_menu'], shared.gradio['softprompts_menu'], show_progress=True)
    shared.gradio['upload_softprompt'].upload(upload_soft_prompt, shared.gradio['upload_softprompt'], shared.gradio['softprompts_menu'])


def set_interface_arguments(interface_mode, extensions, bool_active):
    modes = ["default", "notebook", "chat", "cai_chat"]
    cmd_list = vars(shared.args)
    bool_list = [k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes]

    shared.args.extensions = extensions
    for k in modes[1:]:
        setattr(shared.args, k, False)
    if interface_mode != "default":
        setattr(shared.args, interface_mode, True)

    for k in bool_list:
        setattr(shared.args, k, False)
    for k in bool_active:
        setattr(shared.args, k, True)

    shared.need_restart = True


def create_interface():

    # Defining some variables
    gen_events = []

    if shared.model_name is not None:
        default_preset = shared.settings['presets'][next((k for k in shared.settings['presets'] if re.match(k.lower(), shared.model_name.lower())), 'default')]
    else:
        default_preset = shared.settings['presets']['default']

    if shared.model_name is not None:
        default_text = load_prompt(shared.settings['prompts'][next((k for k in shared.settings['prompts'] if re.match(k.lower(), shared.model_name.lower())), 'default')])
    else:
        default_text = load_prompt(shared.settings['prompts']['default'])

    title = 'Text generation web UI'

    # Authentication variables
    auth = None
    if shared.args.gradio_auth_path is not None:
        gradio_auth_creds = []
        with open(shared.args.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]
        auth = [tuple(cred.split(':')) for cred in gradio_auth_creds]

    # Importing the extension files and executing their setup() functions
    if shared.args.extensions is not None and len(shared.args.extensions) > 0:
        extensions_module.load_extensions()

    # css/js strings
    css = ui.css if not shared.is_chat() else ui.css + ui.chat_css
    js = ui.main_js if not shared.is_chat() else ui.main_js + ui.chat_js
    css += apply_extensions('css')
    js += apply_extensions('js')

    with gr.Blocks(css=css, analytics_enabled=False, title=title, theme=ui.theme) as shared.gradio['interface']:

        # Create chat mode interface
        if shared.is_chat():
            shared.input_elements = ui.list_interface_input_elements(chat=True)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['Chat input'] = gr.State()
            shared.gradio['dummy'] = gr.State()

            with gr.Tab('Text generation', elem_id='main'):
                shared.gradio['display'] = gr.HTML(value=chat_html_wrapper(shared.history['visible'], shared.settings['name1'], shared.settings['name2'], 'chat', 'cai-chat'))
                shared.gradio['textbox'] = gr.Textbox(label='Input')
                with gr.Row():
                    shared.gradio['Stop'] = gr.Button('Stop', elem_id='stop')
                    shared.gradio['Generate'] = gr.Button('Generate', elem_id='Generate', variant='primary')
                    shared.gradio['Continue'] = gr.Button('Continue')

                with gr.Row():
                    shared.gradio['Copy last reply'] = gr.Button('Copy last reply')
                    shared.gradio['Regenerate'] = gr.Button('Regenerate')
                    shared.gradio['Replace last reply'] = gr.Button('Replace last reply')

                with gr.Row():
                    shared.gradio['Impersonate'] = gr.Button('Impersonate')
                    shared.gradio['Send dummy message'] = gr.Button('Send dummy message')
                    shared.gradio['Send dummy reply'] = gr.Button('Send dummy reply')

                with gr.Row():
                    shared.gradio['Remove last'] = gr.Button('Remove last')
                    shared.gradio['Clear history'] = gr.Button('Clear history')
                    shared.gradio['Clear history-confirm'] = gr.Button('Confirm', variant='stop', visible=False)
                    shared.gradio['Clear history-cancel'] = gr.Button('Cancel', visible=False)

                shared.gradio['mode'] = gr.Radio(choices=['chat', 'chat-instruct', 'instruct'], value=shared.settings['mode'] if shared.settings['mode'] in ['chat', 'instruct', 'chat-instruct'] else 'chat', label='Mode', info='Defines how the chat prompt is generated. In instruct and chat-instruct modes, the instruction template selected under "Chat settings" must match the current model.')
                shared.gradio['chat_style'] = gr.Dropdown(choices=utils.get_available_chat_styles(), label='Chat style', value=shared.settings['chat_style'], visible=shared.settings['mode'] != 'instruct')

            with gr.Tab('Chat settings', elem_id='chat-settings'):
                with gr.Row():
                    shared.gradio['character_menu'] = gr.Dropdown(choices=utils.get_available_characters(), label='Character', elem_id='character-menu', info='Used in chat and chat-instruct modes.')
                    ui.create_refresh_button(shared.gradio['character_menu'], lambda: None, lambda: {'choices': utils.get_available_characters()}, 'refresh-button')

                with gr.Row():
                    with gr.Column(scale=8):
                        shared.gradio['name1'] = gr.Textbox(value=shared.settings['name1'], lines=1, label='Your name')
                        shared.gradio['name2'] = gr.Textbox(value=shared.settings['name2'], lines=1, label='Character\'s name')
                        shared.gradio['context'] = gr.Textbox(value=shared.settings['context'], lines=4, label='Context')
                        shared.gradio['greeting'] = gr.Textbox(value=shared.settings['greeting'], lines=4, label='Greeting')

                    with gr.Column(scale=1):
                        shared.gradio['character_picture'] = gr.Image(label='Character picture', type='pil')
                        shared.gradio['your_picture'] = gr.Image(label='Your picture', type='pil', value=Image.open(Path('cache/pfp_me.png')) if Path('cache/pfp_me.png').exists() else None)

                shared.gradio['instruction_template'] = gr.Dropdown(choices=utils.get_available_instruction_templates(), label='Instruction template', value='None', info='Change this according to the model/LoRA that you are using. Used in instruct and chat-instruct modes.')
                shared.gradio['name1_instruct'] = gr.Textbox(value='', lines=2, label='User string')
                shared.gradio['name2_instruct'] = gr.Textbox(value='', lines=1, label='Bot string')
                shared.gradio['context_instruct'] = gr.Textbox(value='', lines=4, label='Context')
                shared.gradio['turn_template'] = gr.Textbox(value=shared.settings['turn_template'], lines=1, label='Turn template', info='Used to precisely define the placement of spaces and new line characters in instruction prompts.')
                with gr.Row():
                    shared.gradio['chat-instruct_command'] = gr.Textbox(value=shared.settings['chat-instruct_command'], lines=4, label='Command for chat-instruct mode', info='<|character|> gets replaced by the bot name, and <|prompt|> gets replaced by the regular chat prompt.')

                with gr.Row():
                    with gr.Tab('Chat history'):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('## Upload')
                                shared.gradio['upload_chat_history'] = gr.File(type='binary', file_types=['.json', '.txt'])

                            with gr.Column():
                                gr.Markdown('## Download')
                                shared.gradio['download'] = gr.File()
                                shared.gradio['download_button'] = gr.Button(value='Click me')

                    with gr.Tab('Upload character'):
                        gr.Markdown('## JSON format')
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown('1. Select the JSON file')
                                shared.gradio['upload_json'] = gr.File(type='binary', file_types=['.json'])

                            with gr.Column():
                                gr.Markdown('2. Select your character\'s profile picture (optional)')
                                shared.gradio['upload_img_bot'] = gr.File(type='binary', file_types=['image'])

                        shared.gradio['Upload character'] = gr.Button(value='Submit')
                        gr.Markdown('## TavernAI PNG format')
                        shared.gradio['upload_img_tavern'] = gr.File(type='binary', file_types=['image'])

            with gr.Tab("Parameters", elem_id="parameters"):
                with gr.Box():
                    gr.Markdown("Chat parameters")
                    with gr.Row():
                        with gr.Column():
                            shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                            shared.gradio['chat_prompt_size'] = gr.Slider(minimum=shared.settings['chat_prompt_size_min'], maximum=shared.settings['chat_prompt_size_max'], step=1, label='Maximum prompt size in tokens', value=shared.settings['chat_prompt_size'])

                        with gr.Column():
                            shared.gradio['chat_generation_attempts'] = gr.Slider(minimum=shared.settings['chat_generation_attempts_min'], maximum=shared.settings['chat_generation_attempts_max'], value=shared.settings['chat_generation_attempts'], step=1, label='Generation attempts (for longer replies)', info='New generations will be called until either this number is reached or no new content is generated between two iterations')
                            shared.gradio['stop_at_newline'] = gr.Checkbox(value=shared.settings['stop_at_newline'], label='Stop generating at new line character')

                create_settings_menus(default_preset)

        # Create notebook mode interface
        elif shared.args.notebook:
            shared.input_elements = ui.list_interface_input_elements(chat=False)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['last_input'] = gr.State('')
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Tab('Raw'):
                            shared.gradio['textbox'] = gr.Textbox(value=default_text, elem_classes="textbox", lines=27)

                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()

                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

                        with gr.Row():
                            shared.gradio['Generate'] = gr.Button('Generate', variant='primary', elem_classes="small-button")
                            shared.gradio['Stop'] = gr.Button('Stop', elem_classes="small-button")
                            shared.gradio['Undo'] = gr.Button('Undo', elem_classes="small-button")
                            shared.gradio['Regenerate'] = gr.Button('Regenerate', elem_classes="small-button")

                    with gr.Column(scale=1):
                        gr.HTML('<div style="padding-bottom: 13px"></div>')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        with gr.Row():
                            shared.gradio['prompt_menu'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt')
                            ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button')

                        shared.gradio['open_save_prompt'] = gr.Button('Save prompt')
                        shared.gradio['save_prompt'] = gr.Button('Confirm save prompt', visible=False)
                        shared.gradio['prompt_to_save'] = gr.Textbox(elem_classes="textbox_default", lines=1, label='Prompt name:', interactive=True, visible=False)
                        shared.gradio['count_tokens'] = gr.Button('Count tokens')
                        shared.gradio['status'] = gr.Markdown('')

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

        # Create default mode interface
        else:
            shared.input_elements = ui.list_interface_input_elements(chat=False)
            shared.gradio['interface_state'] = gr.State({k: None for k in shared.input_elements})
            shared.gradio['last_input'] = gr.State('')
            with gr.Tab("Text generation", elem_id="main"):
                with gr.Row():
                    with gr.Column():
                        shared.gradio['textbox'] = gr.Textbox(value=default_text, elem_classes="textbox_default", lines=27, label='Input')
                        shared.gradio['max_new_tokens'] = gr.Slider(minimum=shared.settings['max_new_tokens_min'], maximum=shared.settings['max_new_tokens_max'], step=1, label='max_new_tokens', value=shared.settings['max_new_tokens'])
                        with gr.Row():
                            shared.gradio['Generate'] = gr.Button('Generate', variant='primary', elem_classes="small-button")
                            shared.gradio['Stop'] = gr.Button('Stop', elem_classes="small-button")
                            shared.gradio['Continue'] = gr.Button('Continue', elem_classes="small-button")
                            shared.gradio['open_save_prompt'] = gr.Button('Save prompt', elem_classes="small-button")
                            shared.gradio['save_prompt'] = gr.Button('Confirm save prompt', visible=False, elem_classes="small-button")
                            shared.gradio['count_tokens'] = gr.Button('Count tokens', elem_classes="small-button")

                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    shared.gradio['prompt_menu'] = gr.Dropdown(choices=utils.get_available_prompts(), value='None', label='Prompt')
                                    ui.create_refresh_button(shared.gradio['prompt_menu'], lambda: None, lambda: {'choices': utils.get_available_prompts()}, 'refresh-button')

                            with gr.Column():
                                shared.gradio['prompt_to_save'] = gr.Textbox(elem_classes="textbox_default", lines=1, label='Prompt name:', interactive=True, visible=False)
                                shared.gradio['status'] = gr.Markdown('')

                    with gr.Column():
                        with gr.Tab('Raw'):
                            shared.gradio['output_textbox'] = gr.Textbox(elem_classes="textbox_default_output", lines=27, label='Output')

                        with gr.Tab('Markdown'):
                            shared.gradio['markdown'] = gr.Markdown()

                        with gr.Tab('HTML'):
                            shared.gradio['html'] = gr.HTML()

            with gr.Tab("Parameters", elem_id="parameters"):
                create_settings_menus(default_preset)

        # Model tab
        with gr.Tab("Model", elem_id="model-tab"):
            create_model_menus()

        # Training tab
        with gr.Tab("Training", elem_id="training-tab"):
            training.create_train_interface()

        # Interface mode tab
        with gr.Tab("Interface mode", elem_id="interface-mode"):
            modes = ["default", "notebook", "chat"]
            current_mode = "default"
            for mode in modes[1:]:
                if getattr(shared.args, mode):
                    current_mode = mode
                    break

            cmd_list = vars(shared.args)
            bool_list = sorted([k for k in cmd_list if type(cmd_list[k]) is bool and k not in modes + ui.list_model_elements()])
            bool_active = [k for k in bool_list if vars(shared.args)[k]]

            shared.gradio['interface_modes_menu'] = gr.Dropdown(choices=modes, value=current_mode, label="Mode")
            shared.gradio['extensions_menu'] = gr.CheckboxGroup(choices=utils.get_available_extensions(), value=shared.args.extensions, label="Available extensions")
            shared.gradio['bool_menu'] = gr.CheckboxGroup(choices=bool_list, value=bool_active, label="Boolean command-line flags")
            shared.gradio['reset_interface'] = gr.Button("Apply and restart the interface")

            # Reset interface event
            shared.gradio['reset_interface'].click(
                set_interface_arguments, [shared.gradio[k] for k in ['interface_modes_menu', 'extensions_menu', 'bool_menu']], None).then(
                lambda: None, None, None, _js='() => {document.body.innerHTML=\'<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>\'; setTimeout(function(){location.reload()},2500); return []}')

        # chat mode event handlers
        if shared.is_chat():
            shared.input_params = [shared.gradio[k] for k in ['Chat input', 'interface_state']]
            clear_arr = [shared.gradio[k] for k in ['Clear history-confirm', 'Clear history', 'Clear history-cancel']]
            shared.reload_inputs = [shared.gradio[k] for k in ['name1', 'name2', 'mode', 'chat_style']]

            gen_events.append(shared.gradio['Generate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['textbox'].submit(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: (x, ''), shared.gradio['textbox'], [shared.gradio['Chat input'], shared.gradio['textbox']], show_progress=False).then(
                chat.generate_chat_reply_wrapper, shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Regenerate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, regenerate=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Continue'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                partial(chat.generate_chat_reply_wrapper, _continue=True), shared.input_params, shared.gradio['display'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False)
            )

            gen_events.append(shared.gradio['Impersonate'].click(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                lambda x: x, shared.gradio['textbox'], shared.gradio['Chat input'], show_progress=False).then(
                chat.impersonate_wrapper, shared.input_params, shared.gradio['textbox'], show_progress=False)
            )

            shared.gradio['Replace last reply'].click(
                chat.replace_last_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy message'].click(
                chat.send_dummy_message, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Send dummy reply'].click(
                chat.send_dummy_reply, shared.gradio['textbox'], None).then(
                lambda: '', None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Clear history-confirm'].click(
                lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr).then(
                chat.clear_chat_log, [shared.gradio[k] for k in ['greeting', 'mode']], None).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Stop'].click(
                stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['mode'].change(
                lambda x: gr.update(visible=x != 'instruct'), shared.gradio['mode'], shared.gradio['chat_style'], show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])


            shared.gradio['chat_style'].change(chat.redraw_html, shared.reload_inputs, shared.gradio['display'])
            shared.gradio['instruction_template'].change(
                partial(chat.load_character, instruct=True), [shared.gradio[k] for k in ['instruction_template', 'name1_instruct', 'name2_instruct']], [shared.gradio[k] for k in ['name1_instruct', 'name2_instruct', 'dummy', 'dummy', 'context_instruct', 'turn_template']])

            shared.gradio['upload_chat_history'].upload(
                chat.load_history, [shared.gradio[k] for k in ['upload_chat_history', 'name1', 'name2']], None).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['Copy last reply'].click(chat.send_last_reply_to_input, None, shared.gradio['textbox'], show_progress=False)
            shared.gradio['Clear history'].click(lambda: [gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)], None, clear_arr)
            shared.gradio['Clear history-cancel'].click(lambda: [gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)], None, clear_arr)
            shared.gradio['Remove last'].click(
                chat.remove_last_message, None, shared.gradio['textbox'], show_progress=False).then(
                chat.save_history, shared.gradio['mode'], None, show_progress=False).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['download_button'].click(lambda x: chat.save_history(x, timestamp=True), shared.gradio['mode'], shared.gradio['download'])
            shared.gradio['Upload character'].click(chat.upload_character, [shared.gradio['upload_json'], shared.gradio['upload_img_bot']], [shared.gradio['character_menu']])
            shared.gradio['character_menu'].change(
                partial(chat.load_character, instruct=False), [shared.gradio[k] for k in ['character_menu', 'name1', 'name2']], [shared.gradio[k] for k in ['name1', 'name2', 'character_picture', 'greeting', 'context', 'dummy']]).then(
                chat.redraw_html, shared.reload_inputs, shared.gradio['display'])

            shared.gradio['upload_img_tavern'].upload(chat.upload_tavern_character, [shared.gradio['upload_img_tavern'], shared.gradio['name1'], shared.gradio['name2']], [shared.gradio['character_menu']])
            shared.gradio['your_picture'].change(
                chat.upload_your_profile_picture, shared.gradio['your_picture'], None).then(
                partial(chat.redraw_html, reset_cache=True), shared.reload_inputs, shared.gradio['display'])

        # notebook/default modes event handlers
        else:
            shared.input_params = [shared.gradio[k] for k in ['textbox', 'interface_state']]
            if shared.args.notebook:
                output_params = [shared.gradio[k] for k in ['textbox', 'markdown', 'html']]
            else:
                output_params = [shared.gradio[k] for k in ['output_textbox', 'markdown', 'html']]

            gen_events.append(shared.gradio['Generate'].click(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply_wrapper, shared.input_params, output_params, show_progress=False)  # .then(
                # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            gen_events.append(shared.gradio['textbox'].submit(
                lambda x: x, shared.gradio['textbox'], shared.gradio['last_input']).then(
                ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                generate_reply_wrapper, shared.input_params, output_params, show_progress=False)  # .then(
                # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
            )

            if shared.args.notebook:
                shared.gradio['Undo'].click(lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False)
                gen_events.append(shared.gradio['Regenerate'].click(
                    lambda x: x, shared.gradio['last_input'], shared.gradio['textbox'], show_progress=False).then(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply_wrapper, shared.input_params, output_params, show_progress=False)  # .then(
                    # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[0]; element.scrollTop = element.scrollHeight}")
                )
            else:
                gen_events.append(shared.gradio['Continue'].click(
                    ui.gather_interface_values, [shared.gradio[k] for k in shared.input_elements], shared.gradio['interface_state']).then(
                    generate_reply_wrapper, [shared.gradio['output_textbox']] + shared.input_params[1:], output_params, show_progress=False)  # .then(
                    # None, None, None, _js="() => {element = document.getElementsByTagName('textarea')[1]; element.scrollTop = element.scrollHeight}")
                )

            shared.gradio['Stop'].click(stop_everything_event, None, None, queue=False, cancels=gen_events if shared.args.no_stream else None)
            shared.gradio['prompt_menu'].change(load_prompt, shared.gradio['prompt_menu'], shared.gradio['textbox'], show_progress=False)
            shared.gradio['open_save_prompt'].click(open_save_prompt, None, [shared.gradio[k] for k in ['prompt_to_save', 'open_save_prompt', 'save_prompt']], show_progress=False)
            shared.gradio['save_prompt'].click(save_prompt, [shared.gradio[k] for k in ['textbox', 'prompt_to_save']], [shared.gradio[k] for k in ['status', 'prompt_to_save', 'open_save_prompt', 'save_prompt']], show_progress=False)
            shared.gradio['count_tokens'].click(count_tokens, shared.gradio['textbox'], shared.gradio['status'], show_progress=False)

        shared.gradio['interface'].load(None, None, None, _js=f"() => {{{js}}}")
        shared.gradio['interface'].load(partial(ui.apply_interface_values, {}, use_persistent=True), None, [shared.gradio[k] for k in ui.list_interface_input_elements(chat=shared.is_chat())], show_progress=False)
        # Extensions tabs
        extensions_module.create_extensions_tabs()

        # Extensions block
        extensions_module.create_extensions_block()

    # Launch the interface
    shared.gradio['interface'].queue()
    if shared.args.listen:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_name=shared.args.listen_host or '0.0.0.0', server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)
    else:
        shared.gradio['interface'].launch(prevent_thread_lock=True, share=shared.args.share, server_port=shared.args.listen_port, inbrowser=shared.args.auto_launch, auth=auth)

# This main entry point loads custom settings from a JSON file, sets default  model settings based on the loaded
# settings, and loads available extensions.
# It then checks for a selected model, either through a command-line argument or a menu, and loads the selected model.
# It also updates the program's persistent interface state and launches a web UI.
# Finally, it restarts the interface.
if __name__ == "__main__":
    # Loading custom settings
    settings_file = shared.args.settings or 'settings.json'
    if Path(settings_file).exists():
        logging.info(f"Loading settings from {settings_file}...")
        new_settings = json.loads(open(settings_file, 'r').read())
        shared.settings.update(new_settings)

    # Set default model settings based on settings.json
    shared.model_config['.*'] = {
        'wbits': 'None',
        'model_type': 'None',
        'groupsize': 'None',
        'pre_layer': 0,
        'mode': shared.settings['mode'],
        'skip_special_tokens': shared.settings['skip_special_tokens'],
        'custom_stopping_strings': shared.settings['custom_stopping_strings'],
    }
    shared.model_config.move_to_end('.*', last=False)  # Move to the beginning

    # Default extensions
    extensions_module.available_extensions = utils.get_available_extensions()
    extensions = shared.settings['chat_default_extensions'] if shared.is_chat() else shared.settings['default_extensions']
    for extension in extensions:
        shared.args.extensions = shared.args.extensions or []
        if extension not in shared.args.extensions:
            shared.args.extensions.append(extension)

    available_models = utils.get_available_models()

    # Model defined through --model
    shared.model_name = shared.args.model or available_models[0] if len(available_models) == 1 else None

    # Select the model from a command-line menu
    if shared.args.model_menu:
        if len(available_models) == 0:
            logging.error('No models are available! Please download at least one.')
            sys.exit(0)
        else:
            print('The following models are available:\n')
            for i, model in enumerate(available_models):
                print(f'{i+1}. {model}')

            print(f'\nWhich one do you want to load? 1-{len(available_models)}\n')
            i = int(input()) - 1
            print()

        shared.model_name = available_models[i]

    # If any model has been selected, load it
    if shared.model_name:
        model_settings = get_model_specific_settings(shared.model_name)
        shared.settings.update(model_settings)  # hijacking the interface defaults
        update_model_parameters(model_settings, initial=True)  # hijacking the command-line arguments

        # Load the model
        shared.model, shared.tokenizer = load_model(shared.model_name)
        if shared.args.lora:
            add_lora_to_model(shared.args.lora)

    # Force a character to be loaded
    if shared.is_chat():
        shared.persistent_interface_state.update({
            'mode': shared.settings['mode'],
            'character_menu': shared.args.character or shared.settings['character'],
            'instruction_template': shared.settings['instruction_template']
        })

    # Launch the web UI
    create_interface()
    while True:
        time.sleep(0.5)
        if shared.need_restart:
            shared.need_restart = False
            shared.gradio['interface'].close()
            time.sleep(0.5)
            create_interface()
