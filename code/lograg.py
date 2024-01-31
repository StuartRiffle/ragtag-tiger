# RAG/TAG Tiger (c) 2024 Stuart Riffle - logging.py

import termcolor

lograg_is_verbose = False
lograg_in_color = True
lograg_default_fore_color = "light_grey"
lograg_default_back_color = "on_black"
lograg_curr_theme = "tiger"
lograg_color_themes = {
    "tiger": {
        "default":          [lograg_default_fore_color, lograg_default_back_color],
        "verbose":          ["dark_grey",        "on_black"],
        #"verbose":          ["cyan",        "on_black"],
        "chat-provider":    ["light_grey",  "on_blue"],
        "chat-model":       ["light_grey",  "on_blue"],
        "chat-mode":        ["light_grey",  "on_dark_grey"],
        "query-mode":       ["white",  "on_green"],
        "chat-message":     ["white",       "on_black"],
        "chat-response":    ["light_yellow",      "on_black"],
        "query-response":   ["light_green",      "on_black"],
        "chat-prompt":      ["dark_grey",      "on_black"],
        "thinking-message": ["dark_grey",   "on_black"],
        "error-message":    ["white",       "on_red"],
    },
}

def lograg_set_verbose(verbose):
    global lograg_is_verbose
    lograg_is_verbose = verbose

def lograg_set_color(color):
    global lograg_in_color
    lograg_in_color = color

def lograg_is_color():
    return lograg_in_color

def lograg(msg, color=None, on_color=None, **kwargs):
    if lograg_in_color:
        termcolor.cprint(msg, color, on_color, **kwargs)
    else:
        print(msg, **kwargs)

def lograg_style_to_colors(style):
    if lograg_in_color:
        if lograg_curr_theme in lograg_color_themes:
            theme = lograg_color_themes[lograg_curr_theme]
            if style in theme:
                fore =  theme[style][0]
                back =  theme[style][1]
                return fore, back
    return None, None

def lograg_in_style(msg, style=None, **kwargs):
    theme_fore, theme_back = lograg_style_to_colors(style)
    forecol = theme_fore or lograg_default_fore_color
    backcol = theme_back or lograg_default_back_color
    lograg(msg, color=forecol, on_color=backcol, **kwargs)

def lograg_verbose(msg, color=None, **kwargs):
    if lograg_is_verbose:
        lograg_in_style(msg, style="verbose", **kwargs)

def lograg_error(msg, exit_code=0, prefix="\t", suffix="", **kwargs):
    error_desc = "FATAL " if exit_code else ""
    lograg(f"{prefix}{error_desc}ERROR: {msg}{suffix}", **kwargs)
    if exit_code:
        exit(exit_code)

