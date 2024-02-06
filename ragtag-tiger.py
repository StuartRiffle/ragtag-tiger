# RAG/TAG Tiger = ragtag.py
# Copyright (c) 2024 Stuart Riffle
# github.com/stuartriffle/ragtag-tiger

# Don't import everything here, only what is needed for command line parsing
import os, time, argparse, shlex

program_name            = "RAG/TAG Tiger"
program_version         = "0.1.0"
program_license         = "MIT"
program_copyright       = "(c) 2024 Stuart Riffle"
program_description     = "Create/update a vector index to perform RAG/TAG queries"
program_repository      = "https://github.com/stuartriffle/ragtag-tiger"
program_start_time      = time.time()

llamaindex_chat_modes   = ["best",  "condense_question", "context", "condense_plus_context", "simple",  "react", "openai"]
llamaindex_query_modes  = ["compact", "accumulate", "compact_accumulate", "generation", "no_text", "refine", "simple_summarize", "tree_summarize"]
llamaindex_alias_to_mode = {
    "agent":     "best",
    "con":       "condense_question",
    "look":      "context",
    "conlook":   "condense_plus_context",
    "simple":    "simple",
    "react":     "react",
    "openai":    "openai",
    "com":       "compact",
    "acc":       "accumulate",
    "comacc":    "compact_accumulate",
    "blind":     "generation",
    "raw":       "no_text",
    "ref":       "refine",
    "sum":       "simple_summarize",
    "tree":      "tree_summarize",
}

llamaindex_mode_to_alias = {}
for alias, mode in llamaindex_alias_to_mode.items():
    llamaindex_mode_to_alias[mode] = alias

llamaindex_alias_set = set(llamaindex_alias_to_mode.keys())
llamaindex_truename_set = set(llamaindex_mode_to_alias.keys())
llamaindex_mode_identifiers = llamaindex_alias_set | llamaindex_truename_set

def chat_query_mode_is_valid(mode):
    return mode in llamaindex_query_modes or mode in llamaindex_alias_set

def chat_query_mode_alias(mode):
    if mode in llamaindex_alias_set:
        return mode
    if mode in llamaindex_mode_to_alias:
        return llamaindex_mode_to_alias[mode]
    return None

def chat_query_mode_truename(mode):
    if mode in llamaindex_truename_set:
        return mode
    if mode in llamaindex_alias_to_mode:
        return llamaindex_alias_to_mode[mode]
    return None

def chat_query_mode_is_chat(mode):
    return chat_query_mode_truename(mode) in llamaindex_chat_modes

class ResponseFileArgumentParser(argparse.ArgumentParser):
    """Ignore comments and whitespace in response files"""
    def convert_arg_line_to_args(self, arg_line):
        line = arg_line.strip()
        if len(line) > 0 and not line.startswith("#"):
            return [line]
        return []
    
def human_size_type(size):
    # Support size specifications like "32k" and "64MB"
    try: return humanfriendly.parse_size(size)
    except: raise argparse.ArgumentTypeError(f"Invalid size: {size}")

parser = ResponseFileArgumentParser(description=program_description, fromfile_prefix_chars='@')

arg = parser.add_argument
arg("--verbose",        help="enable extended/debug output", action="store_true")
arg("--version",        help="print the version number and exit", action="store_true")
arg("--no-color",       help="disable color text output", action="store_true")
arg("--no-delete-temp", help="for debugging, do not clean up working files", action="store_true")

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
arg("--gitignore",      help="Apply rules from .gitignore files found", action="store_true")

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
arg("--chat-mode",      help="Chat response mode", choices=llamaindex_mode_identifiers, default="tree_summarize")
arg("--chat-llm-config",help="Override chat LLM configuration", metavar="CFG")

print(f'{program_name} {program_version}')
print(f'{program_copyright}')
print('')

ragtag_flags = shlex.split(os.environ.get("RAGTAG_FLAGS", ""))
for flag in ragtag_flags:
    os.sys.argv.insert(1, flag)

args = parser.parse_args()
if args.version:
    exit(0)

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
   
print("Waking up tiger...")

from util.lograg import *
verbose_enabled = args.verbose
lograg_set_verbose(verbose_enabled)

running_in_terminal = os.isatty(os.sys.stdout.fileno())
if running_in_terminal and not args.no_color:
    if os.name == "nt":
        try:
            # Makes ANSI color codes work in Windows
            os.system("color")
        except: pass
else:
    lograg_set_color(False)

from util.files import cleanpath
printable_prog_name = cleanpath(os.sys.argv[0], make_unique=True)
command_line_args = " ".join(shlex.quote(arg) for arg in os.sys.argv[1:])
command_line_args = command_line_args.replace(" --", "\n\t    --")
lograg_verbose(f"\t{printable_prog_name}\n\t    {command_line_args}")

import json, tempfile, hashlib, humanfriendly
from llama_index.text_splitter import CodeSplitter
from tqdm import tqdm
from util.files import *
from util.extensions import *
from util.timer import TimerUntil, time_since
from util.llm import load_llm_config, split_llm_config
from util.unpack import unpack_container_to_temp

lograg_verbose(f"\t...at your service ({time_since(program_start_time)})")

#------------------------------------------------------------------------------
# Assemble the inputs
#------------------------------------------------------------------------------

# Search specs (like "foo/bar/**/*.cpp")
search_specs = []

for folder in args.source or []:
    spec = os.path.join(folder, "**/*")
    search_specs.append(spec)

for spec in args.source_spec or []:
    search_specs.append(spec)

for file in args.source_list or []:
    specs = load_and_strip_text(file)
    search_specs.extend(specs)

# User queries
queries = args.query or []

for file in args.query_file or []:
    lograg(f"Loading query from \"{file}\"...")
    query_text = load_and_strip_text(file)
    queries.append(query_text)

for file in args.query_list or []:
    lograg(f"Loading single-line queries from \"{file}\"...")
    short_queries = load_and_strip_text(file)
    queries.extend(short_queries.splitlines())

# System prompt
system_prompt_lines = []

if args.context:
    for snippet in args.context:
        system_prompt_lines.append(snippet)

for file in args.context_file or []:
    lograg(f"Adding system prompt from \"{cleanpath(file)}\"...")
    snippet = load_and_strip_text(file)
    system_prompt_lines.extend(snippet.splitlines())

system_prompt = "\n".join(system_prompt_lines).strip()
if system_prompt:
    lograg_verbose(f"System prompt:\n{system_prompt}")

# Chat instructions
chat_init = args.chat_init or []

if args.chat:
    for file in args.chat_init_file or []:
        lograg(f"Loading chat instructions from \"{cleanpath(file)}\"...")
        chat_init_text = load_and_strip_text(file)
        chat_init.append(chat_init_text)

chat_init_text = "\n".join(chat_init)
if chat_init_text:
    lograg_verbose(f"Chat bot instructions:\n{chat_init_text}")

#------------------------------------------------------------------------------
# Search for files
#------------------------------------------------------------------------------

files_to_index = []
temp_folder = None

if len(search_specs) > 0:
    lograg_verbose(f"Relative paths will be based on current directory \"{cleanpath(os.getcwd())}\"")
    lograg(f"Searching for files...")
    with TimerUntil("complete"):
        files_to_check = []
        for search_spec in search_specs:
            matches = match_files_to_index(search_spec)
            files_to_check.extend(matches)

        files_before_filtering = len(files_to_check)
        if args.size_limit > 0:
            lograg_verbose(f"Ignoring files larger than {humanfriendly.format_size(args.size_limit)}...")
            files_to_check = [f for f in files_to_check if os.path.getsize(f) <= args.size_limit]
            files_ignored = (files_before_filtering - len(files_to_check))
            if files_ignored > 0:
                lograg_verbose(f"\t...{files_ignored} oversized files ignored")

        while files_to_check:
            container_file_types = archive_file_types | mime_file_types
            containers, non_containers = separate_files_by_extension(files_to_check, container_file_types)
            files_to_index.extend(non_containers)

            if args.ignore_archives:
                archives, _ = separate_files_by_extension(containers, archive_file_types)
                if archives:
                    lograg_verbose(f"Ignoring {len(archives)} archive files...")
                    for arch in archives:
                        if arch in containers:
                            containers.remove(arch)

            if not containers:
                break

            all_unpacked_files = []
            for container in containers:
                if not temp_folder:
                    try:
                        temp_folder = cleanpath(tempfile.mkdtemp())
                        lograg_verbose(f"Temporary files will be stored in \"{temp_folder}\"")
                        lograg(f"Unpacking {len(containers)} containers...")
                    except Exception as e:
                        lograg_error(f"failed creating temporary folder, can't unpack containers: {e}")
                        break

                if temp_folder:
                    unpacked_files = unpack_container_to_temp(container, temp_folder)
                    all_unpacked_files.extend(unpacked_files)

            # Unpack nested containers
            files_to_check = all_unpacked_files

files_to_index = [cleanpath(f) for f in files_to_index]
files_to_index = sorted(set(files_to_index))

#------------------------------------------------------------------------------
# Apply .gitignore rules
#------------------------------------------------------------------------------

def identify_gitignored_files(file_list):
    from gitignore import GitIgnore
    gitignore_files = [f for f in file_list if os.path.basename(f) == ".gitignore"]

    files_to_remove = set()
    files_to_remove.update(gitignore_files)

    for gitignore_file in gitignore_files:
        try:
            gitignore = GitIgnore.from_file(gitignore_file)
            for filename in file_list:
                if gitignore.ignored(filename):
                    files_to_remove.add(filename)
        except Exception as e:
            lograg_error(f"error processing .gitignore file \"{gitignore_file}\": {e}")

    return files_to_remove

if args.gitignore and files_to_index:
    gitignore_files = [f for f in files_to_index if os.path.basename(f) == ".gitignore"]
    if gitignore_files:
        with TimerUntil("done"):
            lograg(f"Applying rules from {len(gitignore_files)} .gitignore files...")
            files_to_ignore = identify_gitignored_files(files_to_index)

            lograg_verbose(f"...identified {len(files_to_ignore)} files to ignore")
            files_to_index = [f for f in files_to_index if f not in files_to_ignore]


#------------------------------------------------------------------------------
# Download any custom loaders needed
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
        if not ext.startswith("."):
            ext = "." + ext
        if ext in supported_ext:
            supported_ext.remove(ext)
            ignored_extenstions.append(ext)
    if ignored_extenstions:
        ignored_extenstion_list = ", ".join(sorted(ignored_extenstions)).replace(".", "").strip(", ").lower()
        lograg_verbose(f"Ignoring files with specified extensions: \"{', '.join(ignored_extenstions)}\"")

files_with_ext = {}
for file_path in files_to_index:
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    files_with_ext[ext] = files_with_ext.get(ext, 0) + 1

if verbose_enabled and len(files_to_index) > 0:
    lograg_verbose(f"Supported files found:")
    sorted_items = sorted(files_with_ext.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_items:
        if ext in supported_ext:
            lograg_verbose(f"\t{count:<5} {ext}")

unsupported_ext = set(files_with_ext.keys()) - supported_ext
if len(unsupported_ext) > 0:
    type_list = ", ".join(sorted(unsupported_ext)).replace(".", "").strip(", ").lower()
    if args.index_unknown:
        lograg_verbose(f"WARNING: indexing unknown file types as plain text: {type_list}")
    else:
        lograg_verbose(f"Ignoring these unsupported file types: {type_list}")
        for ext in unsupported_ext:
            del files_with_ext[ext]
        files_before = len(files_to_index)
        files_to_index = [f for f in files_to_index if os.path.splitext(f)[1] in supported_ext]
        if len(files_to_index) < files_before:
            lograg_verbose(f"\t...{files_before - len(files_to_index)} files ignored")

loader_specs = args.custom_loader or []
for ext in files_with_ext.keys():
    if ext in available_hub_loaders:
        loader_name = available_hub_loaders[ext]
        loader_specs.append(f"{loader_name}:{ext.strip('.')}")

custom_loaders = {}
file_extractor_list = {}

if len(loader_specs) > 0:
    lograg(f"Downloading file loaders from the LlamaIndex hub...")
    from llama_index import download_loader
    for loader_spec in loader_specs:
        try:
            loader_class, _, extensions = loader_spec.partition(':')
            if not extensions:
                lograg_error(f"invalid loader spec \"{loader_spec}\"")
                continue

            if not loader_class in custom_loaders:
                with TimerUntil(f"{loader_class}"):
                    custom_loaders[loader_class] = download_loader(loader_class, refresh_cache=args.no_cache_load)

            for ext in extensions.split(","):
                ext = "." + ext.strip(". ")
                file_extractor_list[ext] = custom_loaders[loader_class]()

        except Exception as e: lograg_error(e)

#------------------------------------------------------------------------------
# Chunk all the things
#------------------------------------------------------------------------------
        
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.node_parser import SimpleNodeParser

document_nodes = []

def load_document_nodes(splitter, file_list, show_progress=False):
    doc_reader = SimpleDirectoryReader(
        input_files=file_list, 
        file_extractor=file_extractor_list, 
        exclude_hidden=True)
    
    loaded_docs = doc_reader.load_data(show_progress=show_progress)
    parser = splitter or SimpleNodeParser.from_defaults()
    doc_nodes = parser.get_nodes_from_documents(loaded_docs, show_progress=show_progress)
    return doc_nodes

files_processed = 0
if len(files_to_index) > 0:
    try:
        non_code_files = [f for f in files_to_index if os.path.splitext(f)[1] not in code_ext]
        if len(non_code_files) > 0:
            info = f" {len(non_code_files)}" if verbose_enabled else ""
            lograg(f"Loading{info} documents...")

            with TimerUntil("all documents loaded"):
                non_code_nodes = load_document_nodes(None, non_code_files, show_progress=verbose_enabled)
                document_nodes.extend(non_code_nodes)
                files_processed += len(non_code_files)

        if len(non_code_files) < len(files_to_index):
            lograg(f"Chunking source code...")

            with TimerUntil("code chunked"):
                failed_files = set()
                total_code_files_chunked = 0
                for extensions, code_splitter in source_code_splitters:
                    code_files = [f for f in files_to_index if os.path.splitext(f)[1] in extensions]
                    if not code_files:
                        continue

                    num_files_chunked = 0
                    try:
                        code_nodes = load_document_nodes(code_splitter, code_files, show_progress=verbose_enabled)
                        document_nodes.extend(code_nodes)
                        num_files_chunked += len(code_files)
                    except Exception as e:
                        # Something in that batch failed, so [sigh] process them one at a time
                        lograg_verbose(f"\tError while chunking a batch of {len(code_files)} files, isolating: {e}")
                        progress_bar = tqdm(code_files, desc=f"Chunking individually", leave=False, disable=not verbose_enabled)
                        for code_filename in progress_bar:
                            try:
                                code_nodes = load_document_nodes(code_splitter, [code_filename], show_progress=False)
                                document_nodes.extend(code_nodes)
                                num_files_chunked += 1
                            except: 
                                failed_files.add(code_filename)

                    files_processed += num_files_chunked
                    total_code_files_chunked += num_files_chunked
                    if num_files_chunked > 0:
                        lograg_verbose(f"\t{num_files_chunked} files parsed as \"{code_splitter.language}\"")
                    
                fail_message = f", {len(failed_files)} files failed parsing" if failed_files else ""
                lograg_verbose(f"\t{total_code_files_chunked} total code files processed{fail_message}")

    except Exception as e: lograg_error(e)

#------------------------------------------------------------------------------
# Delete temporary files
#------------------------------------------------------------------------------

if temp_folder and not args.no_delete_temp:
    clean_up_temporary_files(temp_folder)
    if os.path.exists(temp_folder):
        lograg_verbose(f"\t(will try again before exiting)")
    else:
        temp_folder = None

#------------------------------------------------------------------------------
# Process the queries on all given LLM configurations in turn
#------------------------------------------------------------------------------

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

llm_global_params = {}
for param in args.llm_param or []:
    try:
        key, _, value = param.partition("=")
        llm_global_params[key] = value
    except:
        lograg_error(f"invalid --llm-param, format must be key=value: \"{param}\"")

if args.llm_provider or args.llm_server or args.llm_api_key:
    # Build a configuration string out of the individual options
    config_str = f"{args.llm_provider or 'huggingface'},{args.llm_model or ''},{args.llm_server or ''},{args.llm_api_key or ''}"
    for param in args.llm_param or []:
        config_str += f",{param.strip().replace(' ', '')}"
    llm_config_list.insert(0, config_str)

moderator = None
moderator_loaded_last = False

if args.llm_config_mod:
    if args.llm_config_mod in llm_config_list:
        # FWIW, if the moderator goes last, it can generate summaries without being reloaded
        llm_config_list.remove(args.llm_config_mod)
        llm_config_list.append(args.llm_config_mod)
        moderator_loaded_last = True

def lazy_load_vector_index(curr_index):
    if curr_index:
        return curr_index
    
    from llama_index import VectorStoreIndex, StorageContext, ServiceContext

    vector_index = None
    if args.index_load:
        info = f" from \"{cleanpath(args.index_load)}\"" if verbose_enabled else ""
        lograg(f"Loading vector index{info}...")
        try:
            with TimerUntil("loaded"):
                from llama_index import load_index_from_storage
                storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
                vector_index = load_index_from_storage(storage_context, show_progress=verbose_enabled, insert_batch_size=10000)            
        except Exception as e: 
            lograg_error(e)

    if not vector_index:
        lograg_verbose(f"Creating a new vector index in memory...")
        try:
            local_service_context = ServiceContext.from_defaults(llm=None, embed_model="local")
            vector_index = VectorStoreIndex([], insert_batch_size=10000, service_context=local_service_context)
            vector_index.vector_store.persist()
        except Exception as e: 
            lograg_error(e, exit_code=1)
        
    if len(document_nodes) > 0:
        info = f" {len(document_nodes)}" if verbose_enabled else ""
        lograg(f"Indexing{info} document nodes...")
        try:
            with TimerUntil("indexing complete"):
                vector_index.insert_nodes(document_nodes, show_progress=verbose_enabled)
        except Exception as e: lograg_error(e)

    if args.index_store:
        info = f" in \"{cleanpath(args.index_store)}\"" if verbose_enabled else ""
        lograg(f"Storing vector index{info}...")
        try:
            with TimerUntil("index stored"):
                os.makedirs(args.index_store, exist_ok=True)
                vector_index.storage_context.persist(persist_dir=args.index_store)
        except Exception as e: lograg_error(e)    

    curr_index = vector_index
    return vector_index

if files_processed and not queries:
    vector_index = lazy_load_vector_index(vector_index)

tag_queries = (args.tag_queries or "").strip("\"' ")
tag_responses = (args.tag_responses or "").strip("\"' ")

query_prefix = f"### {tag_queries}\n" if tag_queries and not lograg_is_color() else ""
response_prefix = f"### {tag_responses}\n" if tag_responses and not lograg_is_color() else ""

if queries and llm_config_list:
    for llm_config in llm_config_list:

        llm, streaming_supported, service_context = load_llm_config(llm_config, llm_global_params)
        curr_provider, curr_model, curr_server, _, curr_params, _ = split_llm_config(llm_config)
        vector_index = lazy_load_vector_index(vector_index)
       
        if not curr_model:
            curr_model = llm.model_name

        try:
            query_engine_params["streaming"] = streaming_supported
            query_engine = vector_index.as_query_engine(**query_engine_params)
        except Exception as e: 
            lograg_error(f"can't initialize query engine: {e}", exit_code=1)

        #------------------------------------------------------------------------------
        # Process all the queries
        #------------------------------------------------------------------------------

        json_log["comment"] = f"{program_name} v{program_version} ({program_repository})"
        json_log["id"] = hashlib.md5("\n".join(queries).encode()).hexdigest()
        json_log["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S UTC")

        if len(queries) > 0:
            query_count = f"{len(queries)} queries" if len(queries) > 1 else "query"
            #lograg(f"Running {query_count}...")
            with TimerUntil(f"all queries complete"):
                for query_idx in range(len(queries)):
                    query = queries[query_idx]
                    query_log_exists = False
                    for lograg_idx in range(len(json_log["queries"])):
                        if json_log["queries"][lograg_idx]["query"] == query:
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
                        lograg_idx = len(json_log["queries"])
                        json_log["queries"].append(query_record)

                    response_tokens = []
                    response_record = {
                        "params": curr_params or [],
                        "provider": curr_provider,
                        "model": curr_model or "",
                        "server": curr_server or "",
                    }
                    known_end_markers = [
                        "\n---------------------",
                        "<|endoftext|>",
                        "multiple",
                    ]

                    try:
                        with TimerUntil("query complete"):
                            if verbose_enabled:
                                lograg_verbose(query_prefix, end="", flush=True)
                                lograg_in_style(f"{query}", style="batch-query")
                                lograg_verbose(f"\t...thinking ", end="", flush=True)
                                
                            query_start_time = time.time()
                            if streaming_supported:
                                last_token = ""
                                last_token_count = 0
                                streaming_response = query_engine.query(query)
                                for token in streaming_response.response_gen:
                                    if len(response_tokens) == 0:
                                        if not token.strip():
                                            # Ignore leading whitespace in response
                                            continue
                                        lograg_verbose(f"({time_since(query_start_time)})\n{response_prefix}", end="", flush=True)

                                    response_tokens.append(token)
                                    if verbose_enabled:
                                        lograg_in_style(token, style="batch-response", end="", flush=True)

                                    #check_marker = "".join(response_tokens).strip()
                                    #for marker in known_end_markers:
                                    #    if check_marker.endswith(marker):
                                    #        break
                                    
                                llm_response = "".join(response_tokens).strip()
                            else:
                                llm_response = str(query_engine.query(query)).strip()
                                lograg_verbose(f"({time_since(query_start_time)})")
                                lograg_verbose(f"{response_prefix}")
                                lograg_in_style(f"{llm_response}", style="batch-response", end="", flush=True)

                            transcript_lines.append(f"{query_prefix}{query}\n")
                            transcript_lines.append(f"{response_prefix}{llm_response}\n\n")

                            lograg_verbose("")

                    except Exception as e:
                        llm_response = "ERROR"
                        response_record["error"] = str(e)
                        lograg_verbose("")
                        lograg_error(e)

                    response_record["response"] = llm_response
                    response_record["time"] = time.time() - query_start_time

                    json_log["queries"][lograg_idx]["drafts"].append(response_record)
                    if not "response" in json_log["queries"][lograg_idx]:
                        json_log["queries"][lograg_idx]["response"] = llm_response


#------------------------------------------------------------------------------
# Consolidate responses after querying multiple models
#------------------------------------------------------------------------------

template_consolidate = load_stock_text("template/consolidate.txt")
template_query       = load_stock_text("template/consolidate-query.txt")
template_response    = load_stock_text("template/consolidate-response.txt")

if args.llm_config_mod and queries:
    lograg(f"Generating final answers...")

    with TimerUntil("complete"): 
        if not moderator_loaded_last:
            llm, _, _ = load_llm_config(args.llm_config_mod, set_service_context=True)
        if not vector_index:
            vector_index = lazy_load_vector_index(vector_index)

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

                lograg_verbose(f"{query_prefix}", end="", flush=True)
                if verbose_enabled:
                    lograg_in_style(query, style="batch-query")
                    lograg_verbose(f"\t...consolidating responses, please hold... ", end="", flush=True)

                try:
                    with TimerUntil("done", prefix=""):
                        full_analysis = str(query_engine.query(prompt))
                except Exception as e:
                    lograg_error(f"failure generating summary: {e}")
                    continue

                validation = full_analysis.split("## VALIDATION")[1].split("## EVALUATION")[0].strip()
                evaluation = full_analysis.split("## EVALUATION")[1].split("## SUMMARY")[0].strip()
                summary = full_analysis.split("## SUMMARY")[-1].strip()

                if not validation or not evaluation or not summary:
                    lograg_error(f"summary formatting error")
                    continue

                json_log["queries"][query_record_idx]["id"] = hashlib.md5(query.encode()).hexdigest()
                json_log["queries"][query_record_idx]["moderator"] = args.llm_config_mod
                json_log["queries"][query_record_idx]["validation"] = validation
                json_log["queries"][query_record_idx]["evaluation"] = evaluation
                json_log["queries"][query_record_idx]["response"] = summary

                lograg_verbose(response_prefix, end="", flush=True)
                if verbose_enabled:
                    lograg_in_style(summary, style="query-response")

                transcript_lines.append(f"## Query {query_record_idx + 1}\n{query.strip()}\n\n")
                transcript_lines.append(f"### Validation\n\n{validation.strip()}\n\n")
                transcript_lines.append(f"### Evaluation\n\n{evaluation.strip()}\n\n")
                transcript_lines.append(f"### Summary\n\n{summary.strip()}\n\n")

            except Exception as e: 
                lograg_error(e)

#------------------------------------------------------------------------------
# Logs or it didn't happen
#------------------------------------------------------------------------------

if queries:
    if args.query_log:
        lograg(f"Appending query log to \"{cleanpath(args.query_log)}\"...")
        try:
            with open(args.query_log, "a", encoding="utf-8") as f:
                f.write("\n".join(transcript_lines))
        except Exception as e: lograg_error(e)

    if args.query_log_json:
        lograg(f"Writing JSON log to \"{cleanpath(args.query_log_json)}\"...")
        try:
            with open(args.query_log_json, "w", encoding="utf-8") as f:
                raw_text = json.dumps(json_log, indent=4)
                f.write(raw_text)
        except Exception as e: lograg_error(e)

#------------------------------------------------------------------------------
# Interactive chat mode
#------------------------------------------------------------------------------

if args.chat:
    chat_llm_config = None
    if llm_config_list:
        chat_llm_config = llm_config_list[-1]
    if args.llm_config_mod:
        chat_llm_config = args.llm_config_mod
    if args.chat_llm_config:
        chat_llm_config = args.chat_llm_config
    if not chat_llm_config:
        lograg_error("no LLMs configured for chat use --chat-llm-config", exit_code=1)

    chat_engine = None
    query_engine = None
    show_lazy_banner = True

    curr_chat_mode_alias = chat_query_mode_alias(args.chat_mode) if args.chat_mode else None
    curr_chat_mode_alias = curr_chat_mode_alias or "agent"
    is_chat_mode = chat_query_mode_is_chat(curr_chat_mode_alias)
    force_reload_llm = True

    thinking_message = f"...thinking... "
    exit_commands = ["bye", "goodbye", "exit", "quit", "peace", "done", "stop", "end"]

    while True:
        try:
            if force_reload_llm or not llm or not vector_index or not chat_engine or not query_engine:
                from llama_index import set_global_service_context

                service_context = None
                query_engine = None
                chat_engine = None

                curr_provider, curr_model, curr_server, _, curr_params, _ = split_llm_config(chat_llm_config)
                llm, streaming_supported, service_context = load_llm_config(chat_llm_config, llm_global_params, set_service_context=False)

                # FIXME: forcing reload, there must be a good way to do this?
                vector_index = lazy_load_vector_index(None)
                
                chat_params = {
                    "service_context":  service_context,
                    "streaming":        streaming_supported,
                    "system_prompt":    system_prompt + '\n' + chat_init_text,
                }

                query_params = {
                    "service_context":  service_context,
                    "streaming":        streaming_supported,
                    "system_prompt":    system_prompt,
                }

                # FIXME - reinitializing both chat and query engines here, but only one will be used at a time
                mode_truename = chat_query_mode_truename(curr_chat_mode_alias)
                if mode_truename in llamaindex_chat_modes:
                    chat_params["chat_mode"] = mode_truename
                if mode_truename in llamaindex_query_modes:
                    query_params["response_mode"] = mode_truename

                chat_engine = vector_index.as_chat_engine(**chat_params)
                query_engine = vector_index.as_query_engine(**query_params)

                if show_lazy_banner:
                    lograg(f"Entering RAG chat (type \"bye\" to exit, CTRL-C to interrupt)...")
                    if args.chat_log:
                        lograg_verbose(f"\t- logging chat to \"{cleanpath(args.chat_log)}\"")
                    show_lazy_banner = False

                force_reload_llm = False

            chat_or_query = "chat" if is_chat_mode else "query"
            chat_mode_print_style = "chat-mode" if is_chat_mode else "query-mode"
            curr_interactive_alias = curr_chat_mode_alias

            if lograg_is_color():
                lograg("")
                if curr_model:
                    lograg_in_style(f" {curr_model} ", style="chat-model", end="")
                    lograg_in_style("-", style="chat-prompt", end="")
                lograg_in_style(f" {curr_provider} ", style="chat-provider", end="")
                lograg_in_style("-", style="chat-prompt", end="")
                lograg_in_style(f" {curr_interactive_alias} ", style=chat_mode_print_style, end="") 
                lograg_in_style(":", style="chat-prompt", end="")
                lograg_in_style(" ", style="chat-message", end="", flush=True)
            else:
                lograg(f"{curr_provider}:{curr_model or ''} ({chat_or_query} {curr_interactive_alias}): ", end="", flush=True)            

            message = input("").strip()
            if not message:
                continue

        except KeyboardInterrupt:
            continue

        if message.lower() in exit_commands:
            break

        if message.startswith("/"):
            command, _, message = message[1:].lower().partition(" ")
            if command in exit_commands:
                break

            if command == "mode":
                mode_true_name = chat_query_mode_truename(message)
                if mode_true_name:
                    mode_alias = chat_query_mode_alias(mode_true_name)
                    if mode_true_name in llamaindex_chat_modes:
                        lograg(f"Setting chat response mode \"{mode_true_name}\" (alias \"{mode_alias}\")")
                        is_chat_mode = True
                        curr_chat_mode_alias = mode_alias
                        force_reload_llm = True
                        continue
                    elif mode_true_name in llamaindex_query_modes:
                        lograg(f"Setting query response mode \"{mode_true_name}\" (alias \"{mode_alias}\")")
                        lograg_verbose(f"\tNOTE: query modes have no chat memory")
                        is_chat_mode = False
                        curr_chat_mode_alias = mode_alias
                        force_reload_llm = True
                        continue
                # FIXME - print better help
                lograg(f"Chat modes:  {llamaindex_chat_modes}")
                lograg(f"Query modes: {llamaindex_query_modes}")
                continue

            elif command == "clear":
                chat_engine.reset()
                continue

            elif command == "verbose":
                verbose_enabled = (not message or message == "on")
                lograg_set_verbose(verbose_enabled)
                lograg(f"Verbose mode {'ON' if verbose_enabled else 'off'}")
                continue

        if args.chat_log:
            try:
                with open(args.chat_log, "a", encoding="utf-8") as f:
                    f.write(f"{query_prefix}{message}\n")
            except Exception as e: lograg_error(e)

        lograg_in_style(thinking_message, style="thinking-message", end="", flush=True)
        response_tokens = []

        try:
            response_text_style = "chat-response" if is_chat_mode else "query-response"

            if streaming_supported:
                if is_chat_mode:
                    streaming_response = chat_engine.stream_chat(message)
                else:
                    streaming_response = query_engine.query(message)

                for token in streaming_response.response_gen:
                    if len(response_tokens) == 0:
                        if not token.strip():
                            continue
                        lograg(f"\r{' ' * len(thinking_message)}", end="\r")
                    response_tokens.append(token)
                    lograg_in_style(token, style=response_text_style, end="", flush=True)

            else:
                if is_chat_mode:
                    response_text = str(chat_engine.query(message))
                else:
                    response_text = str(query_engine.query(query))

                response_tokens = [response_text]
                lograg_in_style(response_text, style=response_text_style, end="", flush=True)

            lograg("")
        except KeyboardInterrupt:
            lograg_in_style("[BREAK]", style="error-message")
            response_tokens = ["ERROR"]

        llm_response = "".join(response_tokens).strip()

        if args.chat_log:
            try:
                with open(args.chat_log, "a", encoding="utf-8") as f:
                    f.write(f"{response_prefix}{llm_response}\n")
            except Exception as e: lograg_error(e)

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------

if temp_folder and not args.no_delete_temp:
    clean_up_temporary_files(temp_folder)

lograg(f"Tiger out, peace.")
lograg_verbose(f"({time_since(program_start_time)})")
