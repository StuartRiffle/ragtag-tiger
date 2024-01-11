# RAG/TAG Tiger
# Copyright (c) 2024 Stuart Riffle

import os, argparse, time, datetime, json, pathspec
from llama_index import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Document
from llama_index import load_index_from_storage, download_loader
from llama_index.text_splitter import CodeSplitter

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
    ([".cpp", ".hpp"],  CodeSplitter(language="cpp")),
    ([".cxx", ".hxx"],  CodeSplitter(language="cpp")),
    ([".inc", ".inl"],  CodeSplitter(language="cpp")),
    ([".c", ".h"],      CodeSplitter(language="c")),
    ([".cs"],           CodeSplitter(language="c_sharp")),
    ([".py"],           CodeSplitter(language="python")),
    ([".lua"],          CodeSplitter(language="lua")),
    ([".cu"],           CodeSplitter(language="cuda")),
    ([".java"],         CodeSplitter(language="java")),
    ([".js"],           CodeSplitter(language="javascript")),
    ([".ts"],           CodeSplitter(language="typescript")),
]

known_llm_models = {
    "default":      "EleutherAI/gpt-neo-2.7B",
    "airoboros":    "TheBloke/Airoboros-L2-70B-3.1.2-AWQ",
    "llama2":       "TheBloke/LLAMA-L2-70B-3.1.2-AWQ",
    "llama2-small": "TheBloke/LLAMA-L2-70B-3.1.2-AWQ",

}

#------------------------------------------------------------------------------
# Parse command line arguments and response files
#------------------------------------------------------------------------------

class ResponseFileArgumentParser(argparse.ArgumentParser):
    """Support comments and whitespace in response files"""
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

arg = parser.add_argument_group("Language model").add_argument
arg("--llm-server",     help="LLM inference server URL", metavar="URL")
arg("--llm-provider",   help="Commercial inference provider", choices=["openai", "claude", "bing", "gemini"])
arg("--llm-model",      help="Path or model name for local inference", default="default", metavar="NAME")
arg("--llm-api-key",    help="API key for inference server (if needed)", default="", metavar="KEY")
arg("--llm-secret",     help="Secret for inference server (if needed)", default="", metavar="SECRET")
arg("--llm-param",      help="Inference parameter, like \"temperature=0.9\" etc", nargs="+", metavar="KVP")
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

args = parser.parse_args()
if args.version:
    print(f"{program_version}\n")
    exit(0)

#------------------------------------------------------------------------------
# Misc loggery
#------------------------------------------------------------------------------

start_time = time.time()
all_errors = []

def log(msg, **kwargs):
    if not args.quiet:
       print(msg, **kwargs)

def log_verbose(msg, **kwargs):
    if args.verbose:
        log(msg, **kwargs)

def log_error(msg, exit_code=0, prefix="\t", suffix="", **kwargs):
    error_desc = "FATAL " if exit_code else ""
    error_message = f"{error_desc}ERROR: {msg}"
    all_errors.append(error_message)

    log(f"{prefix}{error_message}{suffix}", **kwargs)
    if exit_code:
        exit(exit_code)

log(f"{program_name} {program_version}")
log(f"{program_copyright}, {program_license} license\n")
log()

#------------------------------------------------------------------------------
# A scope timer for verbose mode
#------------------------------------------------------------------------------

def time_since(before):
    return f"{time.time() - before:.3f} sec"

class TimerUntil:
    def __init__(self, msg, prefix="\t", suffix="", **kwargs):
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
# Gather all the file specs to search for indexing
#------------------------------------------------------------------------------
        
search_specs = []

def strip_and_remove_comments(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not line.startswith('#')]
    return "\n".join(lines)

for folder in args.source or []:
    log(f"Including all files under folder \"{os.path.normpath(folder)}\"...")
    spec = os.path.join(folder, "**/*")
    search_specs.append(spec)

for spec in args.source_spec or []:
    log(f"Including files matching spec \"{spec}\"...")
    search_specs.append(spec)

for file in args.source_list or []:
    log(f"Including files from name/spec list in \"{os.path.normpath(file)}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            specs = strip_and_remove_comments(f.read())
            search_specs.extend(specs)
    except Exception as e: log_error(e)

log_verbose(f"Relative paths will be based on the current directory: \"{os.getcwd()}\"")

#------------------------------------------------------------------------------
# Find all files matching these specs
#------------------------------------------------------------------------------

def split_root_from_spec(spec):
    """Split off the root path from the wildcard part if possible"""
    for pos in range(len(spec)):
        if spec[pos] == "*" or spec[pos] == "?":
            sep_pos = spec.rfind(os.path.sep, 0, pos)
            if sep_pos >= 0:
                return spec[:sep_pos + 1], spec[sep_pos + 1:]
    return "", spec

log(f"Finding the specified files...")
files_to_index = []

for file_spec in search_specs:
    if os.path.isfile(file_spec):
        files_to_index.append(file_spec)
    else:
        try:
            file_spec_root, file_spec = split_root_from_spec(file_spec)
            file_spec = file_spec.replace('\\', '/')

            relative_pathspec = pathspec.PathSpec.from_lines('gitwildmatch', [file_spec])
            matches = relative_pathspec.match_tree(file_spec_root)
            matches = [os.path.join(file_spec_root, match) for match in matches]
            files_to_index.extend(matches)

        except Exception as e: log_error(e)

files_to_index = [os.path.abspath(os.path.normpath(f)) for f in files_to_index]
files_to_index = sorted(set(files_to_index))

files_with_ext = {}
for file_path in files_to_index:
    _, extension = os.path.splitext(file_path)
    files_with_ext[extension] = files_with_ext.get(extension, 0) + 1

if args.verbose and len(files_to_index) > 0:
    log_verbose(f"Document count by file type:")
    sorted_items = sorted(files_with_ext.items(), key=lambda x: x[1], reverse=True)
    for extension, count in sorted_items:
        log_verbose(f"\t{count:10} {extension}")

#------------------------------------------------------------------------------
# Download custom loaders for recognized/requested types
#------------------------------------------------------------------------------

all_supported_extensions = built_in_loaders.copy()
all_supported_extensions.update(available_hub_loaders.keys())
for extensions, _ in source_code_splitters:
    all_supported_extensions.update(extensions)

unsupported_extensions = set(files_with_ext.keys()) - all_supported_extensions
if len(unsupported_extensions) > 0:
    ext_action = "IGNORED" if args.ignore_unknown else "indexed as PLAIN TEXT"
    type_list = ", ".join(sorted(unsupported_extensions))
    log(f"These unsupported file types will be {ext_action}: {type_list}")

if args.ignore_unknown:
    for ext in unsupported_extensions:
        del files_with_ext[ext]

    files_before = len(files_to_index)
    files_to_index = [f for f in files_to_index if os.path.splitext(f)[1] in all_supported_extensions]
    if len(files_to_index) < files_before:
        log_verbose(f"\t{files_before - len(files_to_index)} files dropped as unsupported")

loader_specs = args.custom_loader or []

for extension in files_with_ext.keys():
    if extension in available_hub_loaders:
        loader_name = available_hub_loaders[extension]
        loader_specs.append(f"{loader_name}:{extension.strip('.')}")

file_extractor_list = {}

if len(loader_specs) > 0:
    log(f"Downloading file loaders from the LlamaIndex hub...")
    for loader_spec in loader_specs:
        loader_class, _, extensions = loader_spec.partition(':')
        if not extensions:
            log(f"\tWARNING: invalid loader spec \"{loader_spec}\"")
            continue

        try:
            with TimerUntil(f"{loader_class} downloaded"):
                loader = download_loader(loader_class)
                for extension in extensions.split(","):
                    extension = "." + extension.strip(". ")
                    file_extractor_list[extension] = loader()

        except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Chunk source code in a syntax-aware way
#------------------------------------------------------------------------------

class CodeAwareDirectoryReader(SimpleDirectoryReader):
    def readFile(self, file_path):
        try:
            for extensions, code_splitter in source_code_splitters:
                if file_path.endswith(tuple(extensions)):
                    with open(file_path, 'r') as f:
                        source_code = f.read()
                        chunks = code_splitter.chunk(source_code)
                        docs = [Document(file_path, chunk) for chunk in chunks]
                        return docs
        except Exception as e:
            log_error(f"chunking \"{os.path.normcase(file_path)}\": {e}")
                
        return super().readFile(file_path)

#------------------------------------------------------------------------------
# Load and chunk all those documents
#------------------------------------------------------------------------------

if len(files_to_index) > 0:
    log(f"Chunking {len(files_to_index)} files...")
    try:
        with TimerUntil("files loaded"):
            doc_reader = CodeAwareDirectoryReader(
                input_files=files_to_index, 
                file_extractor=file_extractor_list, 
                exclude_hidden=True)
            
            docs_to_index = doc_reader.load_data(show_progress=args.verbose)

    except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Update the vector database
#------------------------------------------------------------------------------

vector_index = None

if args.index_load:
    log(f"Loading the vector index in \"{os.path.normcase(args.index_load)}\"...")
    try:
        with TimerUntil("loaded"):
            storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
            vector_index = load_index_from_storage(storage_context, show_progress=args.verbose)            
    except Exception as e: log_error(e)

if not vector_index:
    log_verbose(f"Creating a new vector index in memory...")
    try:
        with TimerUntil("ready"):
            vector_index = VectorStoreIndex(nodes=[], show_progress=args.verbose)
    except Exception as e: log_error(e)
    
if len(docs_to_index or []) > 0:
    log(f"Indexing {len(docs_to_index)} documents...")
    with TimerUntil("all indexing complete"):
        for doc in docs_to_index:
            try:
                with TimerUntil(f"{doc.file_path}"):
                    vector_index.add_document(doc)
            except Exception as e:
                log_error(f"indexing {doc.file_path}: {e}")

if args.index_store:
    log(f"Storing vector index in \"{os.path.normcase(args.index_store)}\"...")
    try:
        if not os.path.exists(args.index_store):
            os.makedirs(args.index_store)
        with TimerUntil("stored"):
            vector_index.storage_context.persist(persist_dir=args.index_store, show_progress=args.verbose)
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

if args.verbose:
    indented_prompt = "\n".join([f"\t{line}" for line in system_prompt_lines])
    log_verbose(f"System prompt:\n{indented_prompt}")

system_prompt = "\n".join(system_prompt_lines) + "\n"

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

log_verbose(f"Total queries: {len(queries)}")

#------------------------------------------------------------------------------
# Initialize the query engine
#------------------------------------------------------------------------------

if args.llm_provider and args.llm_server:
    log_error(f"cannot specify both --llm-provider and --llm-server", exit_code=1)

model_name = args.llm_model or "default"
if model_name in known_llm_models:
    model_name = known_llm_models[model_name]

local_model = None
query_engine = None
query_engine_params = {
    "response_mode":    args.query_mode,
    "show_progress":    args.verbose,
    "model_name":       model_name,
    "api_key":          args.llm_api_key,
    "secret":           args.llm_secret,
    "params":           dict([param.split("=") for param in args.llm_param or []]),
}

log(f"Initializing query engine...")
try:
    if args.llm_provider:
        query_engine_params["provider"] = args.llm_provider
    elif args.llm_server:
        query_engine_params["server_url"] = args.llm_server
    else:
        with TimerUntil("model loaded"):
            from transformers import AutoModelForCausalLM
            local_model = AutoModelForCausalLM.from_pretrained(model_name, verbose=args.verbose)
            query_engine_params["model"] = local_model

    with TimerUntil("engine ready"):        
        query_engine = vector_index.as_query_engine(**query_engine_params)

except Exception as e:
    log_error(e, exit_code=1)

#------------------------------------------------------------------------------
# Process all the queries
#------------------------------------------------------------------------------
    
log(f"Running {len(queries)} queries...")
with TimerUntil("all queries complete"):
    chat_log = ""
    json_log = {
        "model":        query_engine.model_name,
        "timestamp":    datetime.now().isoformat(),
        "context":      system_prompt,
        "queries":      []
    }

    for query in queries:
        query_record = { "query": query }
        response = ""

        try:
            query_start_time = time.time()
            visible_history = chat_log if args.query_memory else ""
            user_prompt = f"{args.tag_queries}: {query}"
            prompt = system_prompt + visible_history + user_prompt
        
            with TimerUntil("query complete"):
                response = query_engine.query(prompt, verbose=args.verbose)
                query_record["response_time"] = time_since(query_start_time)

        except Exception as e:
            query_record["error"] = str(e)
            log_error(e)

        query_record["response"] = response
        json_log["queries"].append(query_record)

        response_text = f"{args.tag_responses}: {response}"
        interaction = f"{user_prompt}\n{response_text}\n"
        chat_log += interaction

        if args.verbose:
            indented_interaction = "\n".join([f"\t{line}" for line in interaction.splitlines()])
            log(indented_interaction)

if args.output_text:
    log(f"Writing query log to \"{args.output_text}\"...")
    try:
        with open(args.output_text, "w", encoding="utf-8") as f:
            f.write(chat_log)
    except Exception as e: log_error(e)

if args.output_json:
    log(f"Writing JSON query log to \"{args.output_json}\"...")
    try:
        with open(args.output_json, "w", encoding="utf-8") as f:
            raw_text = json.dumps(json_log, indent=4)
            f.write(raw_text)
    except Exception as e: log_error(e)

log("Closing query engine...")
try:
    with TimerUntil("engine closed"):
        query_engine.close()
except Exception as e: log_error(e)

#------------------------------------------------------------------------------
# Load chat bot instructions
#------------------------------------------------------------------------------

chat_init = args.chat_init or []

if args.chat:
    for file in args.chat_init_file or []:
        log(f"Loading chat context/instructions from \"{os.path.normcase(file)}\"...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                chat_init_text = strip_and_remove_comments(f.read())
                chat_init.append(chat_init_text)
        except Exception as e: log_error(e)

    if args.verbose:
        indented_init = "\n".join([f"\t{line}" for line in chat_init])
        log_verbose(f"Chat bot instructions:\n{indented_init}")

#------------------------------------------------------------------------------
# Chat engine initialization
#------------------------------------------------------------------------------

if args.chat:
    log("Initializing chat engine...")
    try:
        chat_engine_params = query_engine_params.copy()
        chat_engine_params.clear("response_mode")
        chat_engine_params["chat_mode"] = args.chat_mode
        chat_engine_params["system_prompt"] = f"{system_prompt}\n{chat_init}"

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
            log()
        except KeyboardInterrupt:
            log("[BREAK]")

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

if len(all_errors) > 0:
    print(f"All errors reported:")
    for error in all_errors:
        print(f"\t{error}")

log(f"Total run time {time_since(start_time)}.")
log("Tiger out, peace.")
