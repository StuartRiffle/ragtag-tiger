
#------------------------------------------------------------------------------
# Misc loggery
#------------------------------------------------------------------------------

def log(msg, **kwargs):
    if not args.quiet:
       print(msg, **kwargs)

def log_verbose(msg, **kwargs):
    if args.verbose:
        log(msg, **kwargs)

def log_error(msg, exit_code=0, prefix="\t", suffix="", **kwargs):
    error_desc = "FATAL " if exit_code else ""
    log(f"{prefix}{error_desc}ERROR: {msg}{suffix}", **kwargs)
    if exit_code:
        exit(exit_code)

