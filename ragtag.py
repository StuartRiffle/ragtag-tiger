# RAG/TAG Tiger
# Copyright (c) 2024 Stuart Riffle

import os, argparse, time, json, pathspec
from llama_index import VectorStoreIndex, StorageContext, ServiceContext, SimpleDirectoryReader
from llama_index import download_loader, set_global_service_context, load_index_from_storage
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import CodeSplitter
from llama_index.llms import OpenAI, OpenAILike, Anthropic, HuggingFaceLLM, LlamaCPP, LangChainLLM

program_name        = "RAG/TAG Tiger"
program_version     = "0.1.0"
program_license     = "MIT"
program_copyright   = "Copyright (c) 2024 Stuart Riffle"
program_description = "Update and query a LlamaIndex vector index"

#------------------------------------------------------------------------------
# File type support and default values etc, update these as needed
#------------------------------------------------------------------------------

built_in_loaders = set([
    '.csv', '.docx', '.epub', '.hwp', '.ipynb', '.jpeg', '.jpg', '.mbox', 
    '.md', '.mp3', '.mp4', '.pdf', '.png', '.ppt', '.pptm', '.pptx',
])

available_hub_loaders = {
    ".json":    "JSONReader",
    ".xlsx":    "PandasExcelReader",
    ".graphql": "SDLReader",
    ".txt":     "UnstructuredReader",
    ".rtf":     "UnstructuredReader",
    ".eml":     "UnstructuredReader",
    ".html":    "UnstructuredReader",
}

source_code_splitters = [
    ([".cpp", ".hpp", ".cxx", ".hxx", ".inc", ".inl", ".c", ".h"],
                CodeSplitter(language="cpp")),            
    ([".cs"],   CodeSplitter(language="c_sharp")),
    ([".py"],   CodeSplitter(language="python")),
    ([".lua"],  CodeSplitter(language="lua")),
    ([".java"], CodeSplitter(language="java")),
    ([".js"],   CodeSplitter(language="javascript")),
    ([".ts"],   CodeSplitter(language="typescript")),
    ([".glsl"], CodeSplitter(language="glsl")),
    ([".hlsl"], CodeSplitter(language="hlsl")),
]

model_nicknames = {
    "default": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", 
    #"default": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", # '<' not supported between instances of 'int' and 'str'
    #"default": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", # '<' not supported between instances of 'int' and 'str'
}

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#------------------------------------------------------------------------------
# Parse command line arguments and response files
#------------------------------------------------------------------------------

class ResponseFileArgumentParser(argparse.ArgumentParser):
    """Ignore comments and whitespace in response files"""
    def convert_arg_line_to_args(self, arg_line):
        line = arg_line.strip()
        if len(line) > 0 and not line.startswith("#"):
            return [line]
        return []

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
arg("--custom-loader",  help="Use loaders from LlamaIndex hub, specify like \"JPEGReader:jpg,jpeg\"", action="append", metavar="SPEC")
arg("--ignore-unknown", help="Ignore files with unrecognized extensions", action="store_true")
arg("--no-cache",       help="Do not use the local cache for loaders", action="store_true")

arg = parser.add_argument_group("Language model").add_argument
arg("--llm-provider",   help="Inference provider/interface", choices=["openai", "anthropic", "llamacpp", "huggingface"], default="huggingface", metavar="NAME")
arg("--llm-model",      help="Model name/path/etc for provider", metavar="NAME")
arg("--llm-server",     help="Inference server URL (if needed)", metavar="URL")
arg("--llm-api-key",    help="API key for inference server (if needed)", default="", metavar="KEY")
arg("--llm-secret",     help="Secret for inference server (if needed)", default="", metavar="SECRET")
arg("--llm-param",      help="Inference parameter, like \"temperature=0.9\" etc", nargs="+", metavar="KVP")
arg("--torch-device",   help="Device override, like \"cpu\" or \"cuda:1\" (for second GPU)", metavar="DEVICE")
arg("--context",        help="Command line context/system prompt", action="append", metavar="TEXT")
arg("--context-file",   help="File containing a snippet of context",action="append", metavar="FILE")

arg = parser.add_argument_group("Query processing").add_argument
arg("--query",          help="Command line query", action="append", metavar="TEXT")
arg("--query-list",     help="File containing short queries, one per line", action="append", metavar="FILE")
arg("--query-file",     help="File containing one long query", action="append", metavar="FILE")
arg("--query-log",      help="Log queries and responses to a text file", metavar="FILE")
arg("--query-log-json", help="Log queries and responses (plus some metadata) to a JSON file", metavar="FILE")
arg("--query-memory",   help="Maintain \"chat\" history between queries", action="store_true")
arg("--query-mode",     help="Query response mode", choices=["accumulate", "compact", "compact_accumulate", "generation", "no_text", "refine", "simple_summarize", "tree_summarize"], default="tree_summarize")
arg("--tag-queries",    help="The name/header in the transcript for user queries", metavar="NAME", default="Query")
arg("--tag-responses",  help="The name/header in the transcript for engine responses", metavar="NAME", default="Response")

arg = parser.add_argument_group("Interactive chat").add_argument
arg("--chat",           help="Enter chat after any query processing", action="store_true")
arg("--chat-init",      help="Extra instructions/personality for the chat LLM", action="append", metavar="TEXT")
arg("--chat-init-file", help="File containing a snippet of chat LLM instructions", action="append", metavar="FILE")
arg("--chat-log",       help="Append chat queries and responses to a text file", metavar="FILE")
arg("--chat-mode",      help="Chat response mode", choices=["best", "context", "condense_question", "simple", "react", "openai"], default="best")

print(f"{program_name} {program_version}")

args = parser.parse_args()
if args.version:
    exit(0)

#------------------------------------------------------------------------------
# Misc loggery
#------------------------------------------------------------------------------

start_time = time.time()

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

log_verbose(f"{program_copyright}\n")

#------------------------------------------------------------------------------
# A scope timer for verbose mode
#------------------------------------------------------------------------------

def time_since(before):
    return f"{time.time() - before:.3f} sec"

class TimerUntil:
    def __init__(self, msg, prefix="\t...", suffix="", **kwargs):
        self.msg = msg
        self.prefix = prefix
        self.suffix = suffix

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            log_verbose(f"{self.prefix}{self.msg} ({time_since(self.start_time)}{self.suffix})")

#------------------------------------------------------------------------------
# Gather all the file specs we have to search when indexing
#------------------------------------------------------------------------------
        
search_specs = []

def strip_and_remove_comments(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not line.startswith('#')]
    return "\n".join(lines)

for folder in args.source or []:
    spec = os.path.join(folder, "**/*")
    log_verbose(f"Including \"{os.path.normpath(spec)}\"...")
    search_specs.append(spec)

for spec in args.source_spec or []:
    log_verbose(f"Including \"{os.path.normpath(spec)}\"...")
    search_specs.append(spec)

for file in args.source_list or []:
    log_verbose(f"Including files from name/spec list in \"{os.path.normpath(file)}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            specs = strip_and_remove_comments(f.read())
            search_specs.extend(specs)
    except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Find files matching these specs
#------------------------------------------------------------------------------

def split_root_from_spec(spec):
    """Split off the root path from the wildcard part if possible"""
    for pos in range(len(spec)):
        if spec[pos] == "*" or spec[pos] == "?":
            sep_pos = spec.rfind(os.path.sep, 0, pos)
            if sep_pos >= 0:
                return spec[:sep_pos + 1], spec[sep_pos + 1:]
    return "", spec

files_to_index = []

if len(search_specs) > 0:
    log_verbose(f"Relative paths will be based on the current directory (\"{os.getcwd()}\")")
    log(f"Searching for files...")
    with TimerUntil("search complete"):
        for search_spec in search_specs:
            try:
                if os.path.isfile(search_spec):
                    files_to_index.append(search_spec)
                else:
                    file_spec_root, file_spec = split_root_from_spec(search_spec)
                    file_spec_pattern = file_spec.replace(os.path.sep, '/') # required by pathspec
                    relative_pathspec = pathspec.PathSpec.from_lines('gitwildmatch', [file_spec_pattern])

                    matches = relative_pathspec.match_tree(file_spec_root)
                    matches = [os.path.join(file_spec_root, match) for match in matches]

                    log_verbose(f"\t{len(matches)} files match \"{os.path.normpath(file_spec)}\" from \"{os.path.normpath(file_spec_root)}\"")
                    files_to_index.extend(matches)

            except Exception as e: log_error(e)

files_to_index = [os.path.abspath(os.path.normpath(f)) for f in files_to_index]
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

files_with_ext = {}
for file_path in files_to_index:
    _, ext = os.path.splitext(file_path)
    files_with_ext[ext] = files_with_ext.get(ext, 0) + 1

if args.verbose and len(files_to_index) > 0:
    log_verbose(f"Supported files found:")
    sorted_items = sorted(files_with_ext.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_items:
        if ext in supported_ext:
            log_verbose(f"\t{count:<5} {ext}")

unsupported_ext = set(files_with_ext.keys()) - supported_ext
if len(unsupported_ext) > 0:
    ext_action = "IGNORED" if args.ignore_unknown else "indexed as PLAIN TEXT"
    type_list = ", ".join(sorted(unsupported_ext)).replace(".", "").lower()
    log(f"These unsupported types will be {ext_action}: {type_list}")

if args.ignore_unknown:
    for ext in unsupported_ext:
        del files_with_ext[ext]
    files_before = len(files_to_index)
    files_to_index = [f for f in files_to_index if os.path.splitext(f)[1] in supported_ext]
    if len(files_to_index) < files_before:
        log_verbose(f"\t{files_before - len(files_to_index)} files ignored")

loader_specs = args.custom_loader or []
for ext in files_with_ext.keys():
    if ext in available_hub_loaders:
        loader_name = available_hub_loaders[ext]
        loader_specs.append(f"{loader_name}:{ext.strip('.')}")

custom_loaders = {}
file_extractor_list = {}

if len(loader_specs) > 0:
    log(f"Downloading file loaders from the LlamaIndex hub...")
    for loader_spec in loader_specs:
        try:
            loader_class, _, extensions = loader_spec.partition(':')
            if not extensions:
                log_error(f"invalid loader spec \"{loader_spec}\"")
                continue

            if not loader_class in custom_loaders:
                with TimerUntil(f"{loader_class}"):
                    custom_loaders[loader_class] = download_loader(loader_class, refresh_cache=args.no_cache)

            for ext in extensions.split(","):
                ext = "." + ext.strip(". ")
                file_extractor_list[ext] = custom_loaders[loader_class]

        except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Load and chunk all those documents
#------------------------------------------------------------------------------

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
            log(f"Loading {len(non_code_files)} documents...")
            with TimerUntil("all documents loaded"):
                non_code_nodes = load_document_nodes(None, non_code_files, show_progress=args.verbose)
                document_nodes.extend(non_code_nodes)
                files_processed += len(non_code_files)

        if len(non_code_files) < len(files_to_index):
            log(f"Chunking code...")
            with TimerUntil("all code chunked"):
                for extensions, code_splitter in source_code_splitters:
                    code_files = [f for f in files_to_index if os.path.splitext(f)[1] in extensions]
                    if len(code_files) > 0:
                        code_nodes = load_document_nodes(code_splitter, code_files, show_progress=args.verbose)
                        log_verbose(f"\t{len(code_files)} files parsed as \"{code_splitter.language}\"")
                        document_nodes.extend(code_nodes)
                        files_processed += len(code_files)

    except Exception as e: log_error(e)
    assert files_processed == len(files_to_index)

#------------------------------------------------------------------------------
# Collect the user queries
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

system_prompt = "\n".join(system_prompt_lines) + "\n"

if len(system_prompt.strip()) > 0:
    log_verbose(f"System prompt:\n{system_prompt}")

#------------------------------------------------------------------------------
# Instantiate an LLM
#------------------------------------------------------------------------------

llm = None

if len(queries) > 0 or args.chat:
    log(f"Initializing language model...")
    try:
        with TimerUntil("LLM ready"):

            if args.llm_provider == "openai":
                if args.llm_api_key:
                    os.environ["OPENAI_API_KEY"] = args.llm_api_key

                if args.llm_server:
                    llm = OpenAILike(
                        model=args.llm_model or "default",
                        api_base=args.llm_server,
                        verbose=args.verbose)
                    log_verbose(f"\tusing OpenAI API redirected to server \"{args.llm_server}\"")
                else:
                    llm = OpenAI(
                        model=args.llm_model,
                        verbose=args.verbose)
                    log_verbose(f"\tusing OpenAI API")
                
            elif args.llm_provider == "anthropic":
                llm = Anthropic(
                    model=args.llm_model,
                    base_url=args.llm_server,
                    verbose=args.verbose)
                log_verbose(f"\tusing Anthropic API")
                
            elif args.llm_provider == "llamacpp":
                llm = LlamaCPP(
                    model_path=args.llm_model,
                    verbose=args.verbose)
                log_verbose(f"\tusing LlamaCPP for local inference")
            
            else: # the default is "huggingface"
                model_desc = ""
                model_name = args.llm_model or "default"
                if model_name in model_nicknames:
                    model_desc = f"\"{model_name}\" which is HuggingFace model "
                    model_name = model_nicknames[model_name]
                log_verbose(f"\tloading model {model_desc}\"{model_name}\"")

                llm = HuggingFaceLLM(
                    model_name, 
                    device_map=args.torch_device or "auto",
                    system_prompt=system_prompt)
                log_verbose(f"\tusing HuggingFace for local inference")

    except Exception as e: 
        log_error(f"failed initializing LLM: {e}", exit_code=1)

#------------------------------------------------------------------------------
# Service context
#------------------------------------------------------------------------------

if llm:
    log_verbose(f"Setting global service context...")
    try:
        with TimerUntil("context set"):
            service_context = ServiceContext.from_defaults(
                embed_model="local", 
                llm=llm)
            set_global_service_context(service_context)

    except Exception as e: log_error(e, exit_code=1)

#------------------------------------------------------------------------------
# Update the vector database
#------------------------------------------------------------------------------

vector_index = None

if args.index_load:
    log(f"Loading the vector index in \"{os.path.normpath(args.index_load)}\"...")
    try:
        with TimerUntil("loaded"):
            storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
            vector_index = load_index_from_storage(storage_context, show_progress=args.verbose)            
    except Exception as e: log_error(e)

if not vector_index:
    log_verbose(f"Creating a new vector index in memory...")
    try:
        vector_index = VectorStoreIndex([], service_context=service_context)
    except Exception as e: log_error(e, exit_code=1)
    
if len(document_nodes) > 0:
    log(f"Indexing {len(document_nodes)} document nodes...")
    try:
        with TimerUntil("indexing complete"):
            vector_index.insert_nodes(document_nodes, show_progress=args.verbose)
    except Exception as e: log_error(e)

if args.index_store:
    log(f"Storing vector index in \"{os.path.normpath(args.index_store)}\"...")
    try:
        with TimerUntil("index stored"):
            if not os.path.exists(args.index_store):
                os.makedirs(args.index_store)
            vector_index.storage_context.persist(persist_dir=args.index_store)
    except Exception as e: log_error(e)


#------------------------------------------------------------------------------
# Initialize the query engine
#------------------------------------------------------------------------------

query_engine = None
query_engine_params = {
    "response_mode":    args.query_mode,
    "system_prompt":    system_prompt,
    "service_context":  service_context,
}

if len(queries) > 0:
    log(f"Initializing query engine...")
    try:
        with TimerUntil("engine ready"):        
            query_engine = vector_index.as_query_engine(**query_engine_params)
    except Exception as e:
        log_error(e, exit_code=1)

#------------------------------------------------------------------------------
# Process all the queries
#------------------------------------------------------------------------------
    
if len(queries) > 0:
    log(f"Running {len(queries)} queries...")
    with TimerUntil(f"all queries complete"):
        chat_log = ""
        json_log = {
            "context":      system_prompt,
            "queries":      []
        }

        for query in queries:
            query_record = { "query": query }
            response = ""

            try:
                with TimerUntil("query complete"):
                    visible_history = chat_log if args.query_memory else ""
                    user_prompt = f"{args.tag_queries}: {query}"
                    prompt = system_prompt + visible_history + user_prompt
                
                    query_start_time = time.time()
                    response = query_engine.query(prompt)
                    query_record["response_time"] = time_since(query_start_time)

            except Exception as e:
                query_record["error"] = str(e)
                response = "[ERROR]"
                log_error(e)

            query_record["response"] = response
            json_log["queries"].append(query_record)

            response_text = f"{args.tag_responses}: {response}"
            interaction = f"{user_prompt}\n{response_text}\n"
            chat_log += interaction
            log_verbose(interaction)

    if args.query_log:
        log(f"Writing query log to \"{args.query_log}\"...")
        try:
            with open(args.query_log, "w", encoding="utf-8") as f:
                f.write(chat_log)
        except Exception as e: log_error(e)

    if args.query_log_json:
        log(f"Writing JSON query log to \"{args.query_log_json}\"...")
        try:
            with open(args.query_log_json, "w", encoding="utf-8") as f:
                raw_text = json.dumps(json_log, indent=4)
                f.write(raw_text)
        except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Load chat bot instructions
#------------------------------------------------------------------------------

chat_init = args.chat_init or []

if args.chat:
    for file in args.chat_init_file or []:
        log(f"Loading chat context/instructions from \"{os.path.normpath(file)}\"...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                chat_init_text = strip_and_remove_comments(f.read())
                chat_init.append(chat_init_text)
        except Exception as e: log_error(e)

    chat_init_text = "\n".join(chat_init)
    if len(chat_init_text.strip()) > 0:
        log_verbose(f"Chat bot instructions:\n{chat_init_text}")

#------------------------------------------------------------------------------
# Initialize chat engine
#------------------------------------------------------------------------------

if args.chat:
    log("Initializing chat engine...")
    try:
        chat_engine_params = query_engine_params.copy()
        del chat_engine_params["response_mode"]
        chat_engine_params["chat_mode"] = args.chat_mode
        chat_engine_params["system_prompt"] = f"{system_prompt}\n{chat_init}\n"

        with TimerUntil("engine ready"):
            chat_engine = vector_index.as_chat_engine(**chat_engine_params)

    except Exception as e: log_error(e, exit_code=1)

#------------------------------------------------------------------------------
# Interactive chat mode
#------------------------------------------------------------------------------

if args.chat:
    log(f"Entering interactive chat...")
    log(f" - The response mode is \"{args.chat_mode}\"")
    log(f" - Hit CTRL-C to interrupt a response in progress")
    log(f" - Say \"bye\" or something when you're done")
    log()
    
    exit_commands = ["bye", "something", "goodbye", "exit", "quit", "done", "stop", "end"]
    chat_lines = []

    while True:
        try:
            message = input("> ").strip()
            if message.lower() in exit_commands:
                break
        except KeyboardInterrupt:
            continue

        chat_lines.append(f"{args.tag_queries}: {message}")

        try:
            response_line = ""
            streaming_response = chat_engine.chat(message, streaming=True)
            for token in streaming_response.response_gen:
                response_line += token
                log(token, end="")
            log("")
        except KeyboardInterrupt:
            log(" [BREAK]")

        chat_lines.append(f"{args.tag_responses}: {response_line}")

    if args.chat_log and len(chat_lines) > 0:
        log(f"Appending chat log to \"{args.chat_log}\"...")
        try:
            all_lines = "\n".join(chat_lines) + "\n"
            with open(args.chat_log, "a", encoding="utf-8") as f:
                f.write(all_lines)
        except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------

log(f"Total run time {time_since(start_time)}.")
log("Tiger out, peace.")
