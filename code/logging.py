# RAG/TAG Tiger (c) 2024 Stuart Riffle - logging.py

import termcolor

raglog_verbose = False
raglog_in_color = True
raglog_curr_theme = "tiger"
raglog_color_themes = {
    "tiger": {
        "error": "red",
        "warning": "yellow",
        "info": "white",
        "debug": "grey",
        "verbose": "cyan",
    },
}

def raglog_set_verbosity(verbose):
    raglog_verbose = verbose

def raglog(msg, color=None, **kwargs):
    if raglog_in_color:
        termcolor.cprint(msg, color, **kwargs)
    else:
        print(msg, **kwargs)

def raglog_style_to_color_string(style):
    if raglog_in_color and raglog_curr_theme in raglog_color_themes:
        if style in raglog_color_themes[raglog_curr_theme]:
            return raglog_color_themes[raglog_curr_theme][style]
    return None

def raglog_in_style(msg, style=None, **kwargs):
    color = raglog_style_to_color_string(style)
    raglog(msg, color, **kwargs)

def raglog_verbose(msg, color=None, **kwargs):
    if raglog_verbose:
        raglog(msg, **kwargs)

def raglog_error(msg, exit_code=0, prefix="\t", suffix="", **kwargs):
    error_desc = "FATAL " if exit_code else ""
    raglog(f"{prefix}{error_desc}ERROR: {msg}{suffix}", **kwargs)
    if exit_code:
        exit(exit_code)

