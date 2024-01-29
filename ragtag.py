# RAG/TAG Tiger
# Copyright (c) 2024 Stuart Riffle

import time
start_time = time.time()

from util.timer import TimerUntil, time_since


program_name            = "RAG/TAG Tiger"
program_version         = "0.1.0"
program_license         = "MIT"
program_copyright       = "Copyright (c) 2024 Stuart Riffle"
program_description     = "Update and query a LlamaIndex vector index"
program_repository      = "github.com/stuartriffle/ragtag-tiger"

openai_model_default    = "gpt-3.5-turbo-instruct"
google_model_default      = "models/text-bison-001"
anthropic_model_default = "claude-2"
perplexity_default      = "llama-2-70b-chat"
replicate_default       = "mistralai/mixtral-8x7b-instruct-v0.1"
default_llm_provider    = "huggingface"
hf_model_nicknames      = { "default": "codellama/CodeLlama-7b-Instruct-hf" }
llamaindex_chat_modes   = ["best", "context", "condense_question", "simple", "react", "openai"]
llamaindex_query_modes  = ["accumulate", "compact", "compact_accumulate", "generation", "no_text", "refine", "simple_summarize", "tree_summarize"]
default_timeout         = 180

#------------------------------------------------------------------------------
# Parse command line arguments and response files
#------------------------------------------------------------------------------

import argparse

class ResponseFileArgumentParser(argparse.ArgumentParser):
    """Ignore comments and whitespace in response files"""
    def convert_arg_line_to_args(self, arg_line):
        line = arg_line.strip()
        if len(line) > 0 and not line.startswith("#"):
            return [line]
        return []
    
def human_size_type(size):
    try:    return humanfriendly.parse_size(size)
    except: raise argparse.ArgumentTypeError(f"Invalid size: {size}")

parser = ResponseFileArgumentParser(description=program_description, fromfile_prefix_chars='@')

arg = parser.add_argument
arg("--quiet",          help="suppress all output except errors", action="store_true")
arg("--verbose",        help="enable extended/debug output", action="store_true")
arg("--version",        help="print the version number and exit", action="store_true")

arg = parser.add_argument_group("Vector database").add_argument
arg("--index-load",     help="Load the vector index from a given path", metavar="PATH")
arg("--index-store",    help="Save the updated vector index to a given path", metavar="PATH")

arg = parser.add_argument_group("Document indexing").add_argument
arg("--source",         help="Folder of files to be indexed recursively", action="append", metavar="FOLDER")
arg("--source-spec",    help="Index files matching a pathspec, like \"**/*.cpp\"", action="append", metavar="SPEC")
arg("--source-list",    help="Text file with a list of filenames/pathspecs to index", action="append", metavar="FILE")
arg("--custom-loader",  help="Download from hub, i.e. \"JPEGReader:jpg,jpeg\"", action="append", metavar="SPEC")
arg("--index-unknown",  help="Index files with unrecognized extensions as text", action="store_true")
arg("--ignore-archives",help="Do not index files inside zip/tar/etc archives", action="store_true")
arg("--ignore-types",   help="Do not index these file extensions, even if supported", action="append", metavar="EXT")
arg("--size-limit",     help="Ignore huge text files unlikely to contain interesting", type=human_size_type, default=0, metavar="SIZE")
arg("--no-cache-load",  help="Do not use the local cache for loaders", action="store_true")

arg = parser.add_argument_group("Language model").add_argument
arg("--llm-provider",   help="Inference provider/interface", choices=["openai", "google", "llamacpp", "huggingface", "perplexity", "replicate"], metavar="NAME")
arg("--llm-model",      help="Model name/path/etc for provider", metavar="NAME")
arg("--llm-server",     help="Inference server URL (if needed)", metavar="URL")
arg("--llm-api-key",    help="API key for inference server (if needed)", metavar="KEY")
arg("--llm-param",      help="Inference parameter, like \"temperature=0.9\" etc", nargs="+", metavar="KVP")
arg("--llm-config",     help="Condensed LLM config: provider,model,server,api-key,params...", action="append", metavar="CFG")
arg("--llm-config-mod", help="Moderator LLM to consolidate the responses of multiple providers", metavar="CFG")
arg("--llm-mod-mode",   help="Moderator query response mode", choices=llamaindex_query_modes, default="tree_summarize")
arg("--llm-verbose",    help="enable extended/debug output from the LLM", action="store_true")
arg("--torch-device",   help="Device override, like \"cpu\" or \"cuda:1\" (for second GPU)", metavar="DEVICE")
arg("--context",        help="Command line context/system prompt", action="append", metavar="TEXT")
arg("--context-file",   help="File containing a snippet of context",action="append", metavar="FILE")

arg = parser.add_argument_group("Query processing").add_argument
arg("--query",          help="Command line query", action="append", metavar="TEXT")
arg("--query-list",     help="File containing short queries, one per line", action="append", metavar="FILE")
arg("--query-file",     help="File containing one long query", action="append", metavar="FILE")
arg("--query-log",      help="Log queries and responses to a text file", metavar="FILE")
arg("--query-log-json", help="Log queries and responses (plus some metadata) to a JSON file", metavar="FILE")
arg("--query-mode",     help="Query response mode", choices=llamaindex_query_modes, default="tree_summarize")
arg("--tag-queries",    help="The name/header in the transcript for user queries", metavar="NAME", default="User")
arg("--tag-responses",  help="The name/header in the transcript for engine responses", metavar="NAME", default="Tiger")

arg = parser.add_argument_group("Interactive chat").add_argument
arg("--chat",           help="Enter chat after any query processing", action="store_true")
arg("--chat-init",      help="Extra instructions/personality for the chat LLM", action="append", metavar="TEXT")
arg("--chat-init-file", help="File containing a snippet of chat LLM instructions", action="append", metavar="FILE")
arg("--chat-log",       help="Append chat queries and responses to a text file", metavar="FILE")
arg("--chat-mode",      help="Chat response mode", choices=llamaindex_chat_modes, default="best")

import os, shlex

ragtag_flags = shlex.split(os.environ.get("RAGTAG_FLAGS", ""))
for flag in ragtag_flags:
    os.sys.argv.insert(1, flag)

args = parser.parse_args()
spacer = "\n" if args.verbose else ""
print(f'{spacer}{program_name} v{program_version}')

if args.verbose:
    print(f"{program_copyright}\n{program_repository}\n")
if args.version:
    exit(0)
    
print("Waking up tiger...")

#------------------------------------------------------------------------------
# Default values etc, update these as needed
#------------------------------------------------------------------------------

import json, pathspec, tempfile, shutil, torch, hashlib, py7zr, humanfriendly, weakref
import email, email.policy, email.parser, email.message
from llama_index.text_splitter import CodeSplitter

built_in_loaders = set([
    # SimpleDirectoryReader supports these out-of-the-box
    '.pdf', '.csv', '.md', '.mbox', '.ipynb',
    '.docx', '.epub', '.hwp', '.ppt', '.pptm', '.pptx',
    '.jpeg', '.jpg', '.png', '.mp3', '.mp4', # ?
])

available_hub_loaders = {
    # Known custom loaders available on https://llamahub.ai
    ".json":    "JSONReader",
    ".xlsx":    "PandasExcelReader",
    ".graphql": "SDLReader",
    ".gql":     "SDLReader",
}

source_code_splitters = [
    # Source code files get syntax-aware chunking
    ([".c", ".h"],          CodeSplitter(language="cpp")),            
    ([".cl"],               CodeSplitter(language="commonlisp")),
    ([".cpp", ".hpp"],      CodeSplitter(language="cpp")),
    ([".cxx", ".hxx"],      CodeSplitter(language="cpp")),
    ([".cs"],               CodeSplitter(language="c_sharp")),
    ([".css"],              CodeSplitter(language="css")),
    ([".dockerfile"],       CodeSplitter(language="dockerfile")),
    ([".dot"],              CodeSplitter(language="dot")),
    ([".el", ".emacs"],     CodeSplitter(language="elisp")),
    ([".elm"],              CodeSplitter(language="elm")),
    ([".ex", ".exs"],       CodeSplitter(language="elixir")),
    ([".f", ".f90"],        CodeSplitter(language="fortran")),
    ([".go"],               CodeSplitter(language="go")),
    ([".hs"],               CodeSplitter(language="haskell")),
    ([".html", ".htm"],     CodeSplitter(language="html")),
    ([".inc", ".inl"],      CodeSplitter(language="cpp")),
    ([".java"],             CodeSplitter(language="java")),
    ([".jl"],               CodeSplitter(language="julia")),
    ([".js"],               CodeSplitter(language="javascript")),
    ([".kt", ".kts"],       CodeSplitter(language="kotlin")),
    ([".lisp", ".lsp"],     CodeSplitter(language="commonlisp")),
    ([".lua"],              CodeSplitter(language="lua")),
    ([".m"],                CodeSplitter(language="objc")),
    ([".ml", ".mli"],       CodeSplitter(language="ocaml")),
    ([".php"],              CodeSplitter(language="php")),
    ([".pl"],               CodeSplitter(language="perl")),
    ([".py"],               CodeSplitter(language="python")),
    ([".r"],                CodeSplitter(language="r")),
    ([".rb"],               CodeSplitter(language="ruby")),
    ([".rs"],               CodeSplitter(language="rust")),
    ([".scala"],            CodeSplitter(language="scala")),
    ([".sh"],               CodeSplitter(language="bash")),
    ([".sql"],              CodeSplitter(language="sql")),
    ([".sqlite"],           CodeSplitter(language="sqlite")),
    ([".ts"],               CodeSplitter(language="typescript")),
    ([".yaml", ".yml"],     CodeSplitter(language="yaml")),
]

mime_file_types = set([
    # Pull out embedded text/html and uuencoded files
    ".eml", ".msg",

    # Special case: sometimes .doc files are actually MIME, not old binary Word documents
    ".doc",
])

shutil.register_unpack_format('7zip', ['.7z'], py7zr.unpack_7zarchive)
archive_file_types = set([
    # Unpack these archive formats so we can index their contents too
    ".zip", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".tbz2", ".xz", ".txz", 

    # FIXME - these common but unsupported here
    # ".rar", ".lzma", ".lz", ".lz4", ".zst", 
])

chunk_as_text = set([
    # Plain text files, no special handling 
    ".txt", ".TXT", ".rtf", ".log", ".asc", ".ini", ".cfg", 

    # FIXME - use a proper splitter/loader for these 
    ".hlsl", ".hlsli", ".fxh", ".glsl", ".glsli", ".shader",
    ".asm", ".s",
    ".xml",
])



#------------------------------------------------------------------------------
# A scope timer for verbose mode
#------------------------------------------------------------------------------


log_verbose(f"\t...at your service ({time_since(start_time)})")


#------------------------------------------------------------------------------
# Support for searching inside container types like file archives and MIME
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Gather the search specs (like "foo/bar/**/*.cpp")
#------------------------------------------------------------------------------
       
search_specs = []

for folder in args.source or []:
    spec = os.path.join(folder, "**/*")
    search_specs.append(spec)

for spec in args.source_spec or []:
    search_specs.append(spec)

for file in args.source_list or []:
    try:
        with open(file, "r", encoding="utf-8") as f:
            specs = strip_and_remove_comments(f.read())
            search_specs.extend(specs)
    except Exception as e:
        log_error(e)

#------------------------------------------------------------------------------
# Gather the user queries
#------------------------------------------------------------------------------

queries = args.query or []

for file in args.query_file or []:
    log(f"Loading query from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            query_text = strip_and_remove_comments(f.read())
            queries.append(query_text)
    except Exception as e: log_error(e)

for file in args.query_list or []:
    log(f"Loading single-line queries from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            short_queries = strip_and_remove_comments(f.read()).splitlines()
            queries.extend(short_queries)
    except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Construct the system prompt
#------------------------------------------------------------------------------

system_prompt_lines = []

if args.context:
    log(f"Adding system prompt from the command line...")
    for snippet in args.context:
        system_prompt_lines.append(snippet)

for file in args.context_file or []:
    log(f"Adding system prompt from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            snippet = strip_and_remove_comments(f.read())
            system_prompt_lines.extend(snippet.splitlines())
    except Exception as e: log_error(e)

system_prompt = "\n".join(system_prompt_lines).strip()

if len(system_prompt.strip()) > 0:
    log_verbose(f"System prompt:\n{system_prompt}")


#------------------------------------------------------------------------------
# Load chat bot instructions
#------------------------------------------------------------------------------

chat_init = args.chat_init or []

if args.chat:
    for file in args.chat_init_file or []:
        log(f"Loading chat instructions from \"{os.path.normpath(file)}\"...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                chat_init_text = strip_and_remove_comments(f.read())
                chat_init.append(chat_init_text)
        except Exception as e: log_error(e)

    chat_init_text = "\n".join(chat_init)
    if len(chat_init_text.strip()) > 0:
        log_verbose(f"Chat bot instructions:\n{chat_init_text}")

#------------------------------------------------------------------------------
# Find files matching the search specs
#------------------------------------------------------------------------------


files_to_index = []
temp_folder = None

if len(search_specs) > 0:
    log_verbose(f"Relative paths will be based on current directory \"{os.path.normpath(os.getcwd())}\"")
    log(f"Searching for files...")
    with TimerUntil("complete"):
        files_to_check = []
        for search_spec in search_specs:
            matches = match_files_to_index(search_spec)
            files_to_check.extend(matches)

        files_before_filtering = len(files_to_check)
        if args.size_limit > 0:
            log_verbose(f"Ignoring files larger than {humanfriendly.format_size(args.size_limit)}...")
            files_to_check = [f for f in files_to_check if os.path.getsize(f) <= args.size_limit]
            files_ignored = (files_before_filtering - len(files_to_check))
            if files_ignored > 0:
                log_verbose(f"\t...{files_ignored} oversized files ignored")

        while files_to_check:
            container_file_types = archive_file_types | mime_file_types
            containers, non_containers = separate_files_by_extension(files_to_check, container_file_types)
            files_to_index.extend(non_containers)

            if args.ignore_archives:
                archives, _ = separate_files_by_extension(containers, archive_file_types)
                containers.remove(archives)
            if not containers:
                break

            all_unpacked_files = []
            for container in containers:
                if not temp_folder:
                    try:
                        temp_folder = os.path.normpath(tempfile.mkdtemp())
                        log_verbose(f"Temporary files will be stored in \"{temp_folder}\"")
                        log(f"Unpacking {len(containers)} containers...")
                    except Exception as e:
                        log_error(f"failed creating temporary folder \"{temp_folder}\": {e}", exit_code=1)

                unpacked_files = unpack_temp_container(container, temp_folder)
                all_unpacked_files.extend(unpacked_files)

            # Unpack nested containers
            files_to_check = all_unpacked_files

files_to_index = [cleanpath(f) for f in files_to_index]
files_to_index = sorted(set(files_to_index))

#------------------------------------------------------------------------------
# Download custom loaders for recognized/requested types
#------------------------------------------------------------------------------

code_ext = set()
for extensions, _ in source_code_splitters:
    code_ext.update(extensions)

supported_ext = built_in_loaders.copy()
supported_ext.update(available_hub_loaders.keys())
supported_ext.update(code_ext)
supported_ext.update(chunk_as_text)

if args.ignore_types:
    ignored_extenstions = []
    for ext in args.ignore_types:
        if ext in supported_ext:
            supported_ext.remove(ext)
            ignored_extenstions.append(ext)
    if ignored_extenstions:
        ignored_extenstion_list = ", ".join(sorted(ignored_extenstions)).replace(".", "").strip(", ").lower()
        log_verbose(f"Ignoring files with specified extensions: \"{', '.join(ignored_extenstions)}\"")

files_with_ext = {}
for file_path in files_to_index:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    files_with_ext[ext] = files_with_ext.get(ext, 0) + 1

if args.verbose and len(files_to_index) > 0:
    log_verbose(f"Supported files found:")
    sorted_items = sorted(files_with_ext.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_items:
        if ext in supported_ext:
            log_verbose(f"\t{count:<5} {ext}")

unsupported_ext = set(files_with_ext.keys()) - supported_ext
if len(unsupported_ext) > 0:
    type_list = ", ".join(sorted(unsupported_ext)).replace(".", "").strip(", ").lower()
    if args.index_unknown:
        log_verbose(f"WARNING: indexing unknown file types as plain text: {type_list}")
    else:
        log_verbose(f"Ignoring these unsupported file types: {type_list}")
        for ext in unsupported_ext:
            del files_with_ext[ext]
        files_before = len(files_to_index)
        files_to_index = [f for f in files_to_index if os.path.splitext(f)[1] in supported_ext]
        if len(files_to_index) < files_before:
            log_verbose(f"\t...{files_before - len(files_to_index)} files ignored")

loader_specs = args.custom_loader or []
for ext in files_with_ext.keys():
    if ext in available_hub_loaders:
        loader_name = available_hub_loaders[ext]
        loader_specs.append(f"{loader_name}:{ext.strip('.')}")

custom_loaders = {}
file_extractor_list = {}

if len(loader_specs) > 0:
    log(f"Downloading file loaders from the LlamaIndex hub...")
    from llama_index import download_loader
    for loader_spec in loader_specs:
        try:
            loader_class, _, extensions = loader_spec.partition(':')
            if not extensions:
                log_error(f"invalid loader spec \"{loader_spec}\"")
                continue

            if not loader_class in custom_loaders:
                with TimerUntil(f"{loader_class}"):
                    custom_loaders[loader_class] = download_loader(loader_class, refresh_cache=args.no_cache_load)

            for ext in extensions.split(","):
                ext = "." + ext.strip(". ")
                file_extractor_list[ext] = custom_loaders[loader_class]()

        except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Load and chunk all those documents
#------------------------------------------------------------------------------
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

document_nodes = []

def load_document_nodes(splitter, file_list, show_progress=False):
    doc_reader = SimpleDirectoryReader(
        input_files=file_list, 
        file_extractor=file_extractor_list, 
        exclude_hidden=True)
    
    loaded_docs = doc_reader.load_data(show_progress=args.verbose)
    parser = splitter or SimpleNodeParser.from_defaults()
    doc_nodes = parser.get_nodes_from_documents(loaded_docs)
    return doc_nodes

if len(files_to_index) > 0:
    files_processed = 0
    try:
        non_code_files = [f for f in files_to_index if os.path.splitext(f)[1] not in code_ext]
        if len(non_code_files) > 0:
            info = f" {len(non_code_files)}" if args.verbose else ""
            log(f"Loading{info} documents...")
            with TimerUntil("all documents loaded"):
                non_code_nodes = load_document_nodes(None, non_code_files, show_progress=args.verbose)
                document_nodes.extend(non_code_nodes)
                files_processed += len(non_code_files)

        if len(non_code_files) < len(files_to_index):
            log(f"Chunking source code...")
            with TimerUntil("all code chunked"):
                for extensions, code_splitter in source_code_splitters:
                    code_files = [f for f in files_to_index if os.path.splitext(f)[1] in extensions]
                    if len(code_files) > 0:
                        code_nodes = load_document_nodes(code_splitter, code_files, show_progress=args.verbose)
                        log_verbose(f"\t{len(code_files)} files parsed as \"{code_splitter.language}\"")
                        document_nodes.extend(code_nodes)
                        files_processed += len(code_files)

    except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Clean up temporary files (or try to, anyway)
#------------------------------------------------------------------------------


if temp_folder:
    clean_up_temporary_files(temp_folder)
if temp_folder:
    log_verbose(f"\t(will try again before exiting)")



#------------------------------------------------------------------------------
# Process the queries on all given LLM configurations in turn
#------------------------------------------------------------------------------

def load_llm(provider, model, server, api_key, params, set_service_context=True):
    result = None
    streaming_supported = True
    try:
        with TimerUntil("ready"):
            model_kwargs = dict([param.split("=") for param in params]) if params else {}

            ### OpenAI
            if provider == "openai":
                api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
                if not server:
                    model_name = model or openai_model_default
                    log(f"Preparing OpenAI model \"{model_name}\"...")
                    from llama_index.llms import OpenAI
                    result = OpenAI(
                        model=model_name,
                        timeout=default_timeout,
                        api_key=api_key,
                        additional_kwargs=model_kwargs,
                        verbose=args.llm_verbose)
                else:
                    # API compatible server
                    model_name = model or "default"
                    log(f"Preparing model \"{model_name}\" on server \"{server}\"...")
                    from llama_index.llms import OpenAILike
                    result = OpenAILike(
                        model=model_name,
                        additional_kwargs=model_kwargs,
                        api_base=server,
                        max_tokens=1000,
                        max_iterations=100,
                        timeout=default_timeout,
                        verbose=args.llm_verbose)
                
            ### Google
            elif provider == "google":
                api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                model_name = model or google_model_default
                log(f"Preparing Google model \"{model_name}\"...")
                from llama_index.llms import PaLM
                result = PaLM(
                    api_key=api_key,
                    model_name=model_name,
                    generate_kwargs=model_kwargs)
                streaming_supported = False
                
            ### Llama.cpp
            elif provider == "llamacpp":
                if torch.cuda.is_available():
                    # FIXME - this does nothing
                    model_kwargs["n_gpu_layers"] = -1
                    model_kwargs["device"] = "cuda"
                log(f"Preparing llama.cpp model \"{os.path.normpath(model)}\"...")
                from llama_index.llms import LlamaCPP
                result = LlamaCPP(
                    model_path=model,
                    model_kwargs=model_kwargs,
                    verbose=args.llm_verbose)
                
            ### Perplexity
            elif provider == "perplexity":
                api_key = api_key or os.environ.get("PERPLEXITYAI_API_KEY", "")
                model_name = model or perplexity_default
                log(f"Preparing Perplexity model \"{model_name}\"...")
                from llama_index.llms import Perplexity
                result = Perplexity(
                    api_key=api_key,
                    model=model_name,
                    model_kwargs=model_kwargs)
                
            ### Replicate
            elif provider == "replicate":
                api_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
                model_name = model or replicate_default
                log(f"Preparing Replicate model \"model_name)\"...")
                from llama_index.llms import Replicate
                result = Replicate(
                    model=model_name,
                    additional_kwargs=model_kwargs)
            
            ### HuggingFace
            else:
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                model_desc = ""
                model_name = model or "default"
                if model_name in hf_model_nicknames:
                    model_desc = f" (\"{model_name}\")"
                    model_name = hf_model_nicknames[model_name]
                log(f"Preparing HuggingFace model \"{model_name}\"{model_desc}...")
                from llama_index.llms import HuggingFaceLLM
                result = HuggingFaceLLM(
                    model_name=model_name,
                    model_kwargs=model_kwargs, 
                    device_map=args.torch_device or "auto",

                    system_prompt=system_prompt)

            if set_service_context:
                from llama_index import ServiceContext
                from llama_index import set_global_service_context

                service_context = ServiceContext.from_defaults(embed_model="local", llm=result)
                set_global_service_context(service_context)

    except Exception as e: 
        log_error(f"failure initializing LLM: {e}", exit_code=1)

    return result, streaming_supported

def split_llm_config(config):
    """Split an LLM from a config string like "provider,model,server,api-key,param1,param2,..." into its components"""
    config   = config.strip("\"' ")
    fields   = config.split(",")
    provider = fields[0].strip() if len(fields) > 0 else default_llm_provider
    model    = fields[1].strip() if len(fields) > 1 else None
    server   = fields[2].strip() if len(fields) > 2 else None
    api_key  = fields[3].strip() if len(fields) > 3 else None
    params   = fields[4:]        if len(fields) > 4 else []
    return provider, model, server, api_key, params

def load_llm_config(config, set_service_context=True):
    """Load an LLM from a config string like "provider,model,server,api-key,param1,param2,..."""
    provider, model, server, api_key, params = split_llm_config(config)
    return load_llm(provider.lower(), model, server, api_key, params, set_service_context)

llm = None
vector_index = None
service_context = None
query_engine = None
transcript_lines = []
llm_config_list = args.llm_config or []

query_engine_params = {
    "response_mode":    args.query_mode,
    "system_prompt":    system_prompt,
    "service_context":  service_context,
}
json_log = {
    "comment": "",
    "id": "",
    "timestamp": "",
    "context": system_prompt,
    "queries": [],
}

if args.llm_provider or args.llm_server or args.llm_api_key or args.llm_param:
    # Build a configuration string out of the individual options
    config_str = f"{args.llm_provider or 'huggingface'},{args.llm_model or ''},{args.llm_server or ''},{args.llm_api_key or ''}"
    for param in args.llm_param or []:
        config_str += f",{param.strip().replace(' ', '')}"
   
    llm_config_list.insert(config_str, 0)

moderator_loaded_last = False
if args.llm_config_mod:
    if args.llm_config_mod in llm_config_list:
        # If the moderator goes last, it can generate summaries without reloading
        llm_config_list.remove(args.llm_config_mod)
        llm_config_list.append(args.llm_config_mod)
        moderator_loaded_last = True

for llm_config in llm_config_list:
    llm, streaming_supported = load_llm_config(llm_config)
    curr_provider, curr_model, curr_server, _, curr_params = split_llm_config(llm_config)

    #--------------------------------------------------------------------------
    # Update the vector database
    #--------------------------------------------------------------------------

    from llama_index import VectorStoreIndex

    if not vector_index:
        if args.index_load:
            info = f" from \"{os.path.normpath(args.index_load)}\"" if args.verbose else ""
            log(f"Loading vector index{info}...")
            try:
                with TimerUntil("loaded"):
                    from llama_index import StorageContext, load_index_from_storage
                    storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
                    vector_index = load_index_from_storage(storage_context, show_progress=args.verbose)            
            except Exception as e: log_error(e)

        if not vector_index:
            log_verbose(f"Creating a new vector index in memory...")
            try:
                vector_index = VectorStoreIndex([])
            except Exception as e: log_error(e, exit_code=1)
            
        if len(document_nodes) > 0:
            info = f" {len(document_nodes)}" if args.verbose else ""
            log(f"Indexing{info} document nodes...")
            try:
                with TimerUntil("indexing complete"):
                    vector_index.insert_nodes(document_nodes, show_progress=args.verbose)
            except Exception as e: log_error(e)

        if args.index_store:
            info = f" in \"{os.path.normpath(args.index_store)}\"" if args.verbose else ""
            log(f"Storing vector index{info}...")
            try:
                with TimerUntil("index stored"):
                    if not os.path.exists(args.index_store):
                        os.makedirs(args.index_store)
                    vector_index.storage_context.persist(persist_dir=args.index_store)
            except Exception as e: log_error(e)

    #------------------------------------------------------------------------------
    # Initialize the query engine
    #------------------------------------------------------------------------------

    if vector_index:
        try:
            query_engine_params["streaming"] = streaming_supported
            query_engine = vector_index.as_query_engine(**query_engine_params)

        except Exception as e: log_error(f"can't initialize query engine: {e}", exit_code=1)

    #------------------------------------------------------------------------------
    # Process all the queries
    #------------------------------------------------------------------------------

    tag_queries = (args.tag_queries or "").strip("\"' ")
    tag_responses = (args.tag_responses or "").strip("\"' ")

    query_prefix = f"### {tag_queries}\n" if tag_queries else ""
    response_prefix = f"### {tag_responses}\n" if tag_responses else ""

    json_log["comment"] = f"{program_name} v{program_version} ({program_repository})"
    json_log["id"] = hashlib.md5("\n".join(queries).encode()).hexdigest()
    json_log["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC")

    if len(queries) > 0:
        query_count = f"{len(queries)} queries" if len(queries) > 1 else "query"
        #log(f"Running {query_count}...")
        with TimerUntil(f"all queries complete"):
            for query_idx in range(len(queries)):
                query = queries[query_idx]
                query_log_exists = False
                for log_idx in range(len(json_log["queries"])):
                    if json_log["queries"][log_idx]["query"] == query:
                        query_log_exists = True
                        break

                if not query_log_exists:
                    query_record = { 
                        "query": query,
                        "response": "",
                        "moderator": "",
                        "id": "",
                        "drafts": [],
                    }
                    log_idx = len(json_log["queries"])
                    json_log["queries"].append(query_record)

                response_tokens = []
                response_record = {
                    "params": curr_params or [],
                    "provider": curr_provider,
                    "model": curr_model or "",
                    "server": curr_server or "",
                }

                try:
                    with TimerUntil("query complete"):
                        log_verbose(f"\n{query_prefix}{query}\n\t...thinking ", end="", flush=True)
                        query_start_time = time.time()

                        if streaming_supported:
                            streaming_response = query_engine.query(query)
                            for token in streaming_response.response_gen:
                                if len(response_tokens) == 0:
                                    if not token.strip():
                                        continue
                                    log_verbose(f"({time_since(query_start_time)})\n{response_prefix}", end="", flush=True)

                                response_tokens.append(token)
                                log_verbose(token, end="", flush=True)
                            log_verbose("")
                            llm_response = "".join(response_tokens).strip()
                        else:
                            llm_response = str(query_engine.query(query))
                            log_verbose(f"({time_since(query_start_time)})\n{response_prefix}{llm_response}")

                        transcript_lines.append(f"{query_prefix}{query}\n")
                        transcript_lines.append(f"{response_prefix}{llm_response}\n\n")

                except Exception as e:
                    llm_response = "ERROR"
                    response_record["error"] = str(e)
                    log_verbose("")
                    log_error(e)

                response_record["response"] = llm_response
                response_record["time"] = time.time() - query_start_time

                json_log["queries"][log_idx]["drafts"].append(response_record)
                if not "response" in json_log["queries"][log_idx]:
                    json_log["queries"][log_idx]["response"] = llm_response

        log_verbose("")

#------------------------------------------------------------------------------
# Consolidate responses after querying multiple models
#------------------------------------------------------------------------------

template_consolidate = load_stock_text("template/consolidate.txt") or ""
template_query       = load_stock_text("template/consolidate-query.txt") or ""
template_response    = load_stock_text("template/consolidate-response.txt") or ""

if args.llm_config_mod:
    log(f"Generating final answers...")
    with TimerUntil("complete"): 
        if not moderator_loaded_last:
            llm, _ = load_llm_config(args.llm_config_mod, set_service_context=True)

        query_engine_params["streaming"] = False
        query_engine_params["response_mode"] = args.llm_mod_mode
        query_engine = vector_index.as_query_engine(**query_engine_params)

        transcript_lines.append(f"\n\n# Summaries\n")

        for query_record_idx in range(len(json_log["queries"])):
            try:
                query_record = json_log["queries"][query_record_idx]
                query = query_record["query"]
                drafts = query_record["drafts"] or []
                if len(drafts) < 1:
                    continue

                prompt = template_consolidate
                prompt += f"### QUERY\n\n{query}\n\n"
                for response_idx in range(len(drafts)):
                    llm_response = drafts[response_idx]
                    prompt += f"### RESPONSE {response_idx + 1}\n\n{llm_response['response']}\n\n"

                log_verbose(f"{query_prefix}{query}\n\t...consolidating responses, please hold... ", end="", flush=True)

                try:
                    with TimerUntil("done", prefix=""):
                        full_analysis = str(query_engine.query(prompt))
                except Exception as e:
                    log_error(f"failure generating summary: {e}")
                    continue

                validation = full_analysis.split("## VALIDATION")[1].split("## EVALUATION")[0].strip()
                evaluation = full_analysis.split("## EVALUATION")[1].split("## SUMMARY")[0].strip()
                summary = full_analysis.split("## SUMMARY")[-1].strip()

                if not validation or not evaluation or not summary:
                    log_error(f"summary formatting error")
                    continue

                json_log["queries"][query_record_idx]["id"] = hashlib.md5(query.encode()).hexdigest()
                json_log["queries"][query_record_idx]["moderator"] = args.llm_config_mod
                json_log["queries"][query_record_idx]["validation"] = validation
                json_log["queries"][query_record_idx]["evaluation"] = evaluation
                json_log["queries"][query_record_idx]["response"] = summary

                log_verbose(f"{response_prefix}{summary}\n")

                transcript_lines.append(f"## Query {query_record_idx + 1}\n{query.strip()}\n\n")
                transcript_lines.append(f"### Validation\n\n{validation.strip()}\n\n")
                transcript_lines.append(f"### Evaluation\n\n{evaluation.strip()}\n\n")
                transcript_lines.append(f"### Summary\n\n{summary.strip()}\n\n")

            except Exception as e: 
                log_error(e)

#------------------------------------------------------------------------------
# Logs or it didn't happen
#------------------------------------------------------------------------------

if args.query_log:
    log(f"Appending query log to \"{os.path.normpath(args.query_log)}\"...")
    try:
        with open(args.query_log, "a", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))
    except Exception as e: log_error(e)

if args.query_log_json:
    log(f"Writing JSON log to \"{os.path.normpath(args.query_log_json)}\"...")
    try:
        with open(args.query_log_json, "w", encoding="utf-8") as f:
            raw_text = json.dumps(json_log, indent=4)
            f.write(raw_text)
    except Exception as e: log_error(e)


#------------------------------------------------------------------------------
# Initialize chat engine
#------------------------------------------------------------------------------

if args.chat:
    try:
        chat_engine_params = query_engine_params.copy()
        del chat_engine_params["response_mode"]
        chat_engine_params["chat_mode"] = args.chat_mode
        chat_engine_params["system_prompt"] = f"{system_prompt}\n{chat_init}"
        chat_engine = vector_index.as_chat_engine(**chat_engine_params)
    except Exception as e: log_error(e, exit_code=1)

#------------------------------------------------------------------------------
# Interactive chat mode
#------------------------------------------------------------------------------

if args.chat:
    log(f"Entering RAG chat (type \"bye\" to exit, CTRL-C to interrupt)...")
    log_verbose(f"\t- the LlamaIndex chat mode is \"{args.chat_mode}\"")
    if args.chat_log:
        log_verbose(f"\t- logging chat to \"{os.path.normpath(args.chat_log)}\"")
    log("")
    
    thinking_message = f"{response_prefix}...thinking... "
    exit_commands = ["bye", "goodbye", "exit", "quit", "peace", "done", "stop", "end"]
    user_prompt = query_prefix or "> "
    curr_engine = chat_engine

    while True:
        prompt_prefix = ""
        if args.verbose:
            curr_engine_desc = "chat" if curr_engine == chat_engine else "query"
            curr_engine_mode = curr_engine.chat_mode if curr_engine == chat_engine else query_engine.response_mode
            prompt_prefix = f"{curr_engine_desc}/{curr_engine_mode}"
            if query_prefix:
                prompt_prefix = f"({prompt_prefix}) "

        try:
            message = input(prompt_prefix + user_prompt).strip()
            if not message:
                continue
        except KeyboardInterrupt:
            continue

        if message.lower() in exit_commands:
            break

        if message.startswith("/"):
            command, _, message = message[1:].lower().partition(" ")

            if command == "mode":
                if message in llamaindex_chat_modes:
                    curr_engine = chat_engine
                    if message in llamaindex_chat_modes:
                        chat_engine.set_chat_mode(message)
                    else:
                        log(f"Valid chat modes are: {'' if args.verbose else llamaindex_chat_modes}")
                        if args.verbose:
                            log_stock_text("help/chat_modes.txt")
                    log(f"Chat response mode is \"{message}\"")
                elif message in llamaindex_query_modes:
                    command, message = "query", message
                    curr_engine = query_engine
                    if message in llamaindex_query_modes:
                        query_engine.set_response_mode(message)
                    else:
                        log(f"Valid query modes are: {llamaindex_query_modes}")
                        if args.verbose:
                            log_stock_text("help/query_modes.txt")
                    log(f"Query response mode is \"{message}\"")                
                else:
                    log(f"Chat modes:  {llamaindex_chat_modes}")
                    log(f"Query modes: {llamaindex_query_modes}")
                    continue


            elif command == "clear":
                chat_engine.reset()

            elif command == "verbose":
                args.verbose = (not message or message == "on")
                log(f"Verbose mode is {'on' if args.verbose else 'off'}")

            #elif command == "index":
            # mode index exit verbose /help
            # /provider /model /server /api-key /param 
            # /cliche craz
            # /clear /verbose /help
            # /log /response 
            # /device
            # /chat context    
            
            elif command in exit_commands:
                break
                
            elif command == "clear":
            elif command == "clear":
            elif command == "clear":
            elif command == "clear":
            elif command == "clear":
            elif command == "clear":
                chat_engine.reset()


            

            else:
                log(f"Commands: chat, query, mode, clear, verbose")
                   
            continue

        if args.chat_log:
            try:
                with open(args.chat_log, "a", encoding="utf-8") as f:
                    f.write(f"{query_prefix}{message}\n")
            except Exception as e: log_error(e)

        log(thinking_message, end="", flush=True)
        response_tokens = []

        try:
            if curr_engine == chat_engine:
                streaming_response = chat_engine.stream_chat(message)
            else:
                streaming_response = query_engine.query(query)
            
            for token in streaming_response.response_gen:
                if len(response_tokens) == 0:
                    if not token.strip():
                        continue
                    log(f"\r{' ' * len(thinking_message)}", end="\r")
                    log(response_prefix, end="", flush=True)

                response_tokens.append(token)
                log(token, end="", flush=True)
            log("")
        except KeyboardInterrupt:
            log("-[BREAK]")

        llm_response = "".join(response_tokens).strip()

        if args.chat_log:
            try:
                with open(args.chat_log, "a", encoding="utf-8") as f:
                    f.write(f"{response_prefix}{llm_response}\n")
            except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------

if temp_folder:
    clean_up_temporary_files()

log_verbose(f"\nTiger out, peace ({time_since(start_time)})")




def main():


if __name__ == "__main__":
    main()

