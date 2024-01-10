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

auto_download_loaders = ["JSONReader:json"]
fallback_llm_model = "EleutherAI/gpt-neo-2.7B"

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

arg = parser.add_argument_group("Language model").add_argument
arg("--llm-server",     help="LLM inference server URL", metavar="URL")
arg("--llm-provider",   help="Commercial inference provider", choices=["openai", "claude", "bing", "gemini"])
arg("--llm-model",      help="Path or HF model for local inference", default=fallback_llm_model, metavar="NAME")
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

#------------------------------------------------------------------------------
# Timings for verbose mode
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

log_verbose(f"Relative paths will be based on \"{os.getcwd()}\"")


#------------------------------------------------------------------------------
# Find files matching the specs
#------------------------------------------------------------------------------

def split_root_from_spec(spec):
    """Split off the root path from the wildcard part if possible"""
    for pos in range(len(spec)):
        if spec[pos] == "*" or spec[pos] == "?":
            sep_pos = spec.rfind(os.path.sep, 0, pos)
            if sep_pos >= 0:
                return spec[:sep_pos + 1], spec[sep_pos + 1:]
    return "", spec

log(f"Finding files...")
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

if args.verbose and len(files_to_index) > 0:
    log_verbose(f"Document count by file type:")

    files_with_ext = {}
    for file_path in files_to_index:
        _, extension = os.path.splitext(file_path)
        files_with_ext[extension] = files_with_ext.get(extension, 0) + 1

    sorted_items = sorted(files_with_ext.items(), key=lambda x: x[1], reverse=True)
    for extension, count in sorted_items:
        log_verbose(f"\t{count:10} {extension}")

#------------------------------------------------------------------------------
# Download any custom loaders from the hub
#------------------------------------------------------------------------------

loader_specs = auto_download_loaders
loader_specs.extend(args.custom_loader or [])
file_extractor_list = {}

log("Downloading file loaders from the LlamaIndex hub...")
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
# Source code can be chunked semantically
#------------------------------------------------------------------------------

source_code_splitters = [
    ([".cpp", ".c", ".hpp", ".h"],  CodeSplitter(language="cpp")),
    ([".cs"],                       CodeSplitter(language="c_sharp")),
    ([".py"],                       CodeSplitter(language="python")),
    ([".lua"],                      CodeSplitter(language="lua")),
    ([".cu"],                       CodeSplitter(language="cuda")),
    ([".java"],                     CodeSplitter(language="java")),
    ([".js"],                       CodeSplitter(language="javascript")),
    ([".ts"],                       CodeSplitter(language="typescript")),
]        

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
            log_error(f"chunking {file_path}: {e}")
                
        return super().readFile(file_path)


#------------------------------------------------------------------------------
# Load and chunk all those documents
#------------------------------------------------------------------------------

if len(files_to_index) > 0:
    log(f"Loading and chunking {len(files_to_index)} files...")

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
    log(f"Loading the vector index in \"{args.index_load}\"...")
    try:
        with TimerUntil("loaded"):
            storage_context = StorageContext.from_defaults(persist_dir=args.index_load)
            vector_index = load_index_from_storage(storage_context, show_progress=args.verbose)            

    except Exception as e: log_error(e)

if not vector_index:
    log_verbose(f"Creating a new vector index in memory...")
    try:
        with TimerUntil("created"):
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
    log(f"Storing vector index in \"{args.index_store}\"...")
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
    log(f"Adding system context from the command line...")
    for snippet in args.context:
        system_prompt_lines.append(snippet)

for file in args.context_file or []:
    log(f"Adding system context from \"{file}\"...")
    try:
        with open(file, "r", encoding="utf-8") as f:
            snippet = strip_and_remove_comments(f.read())
            system_prompt_lines.extend(snippet.splitlines())
    except Exception as e: log_error(e)

if args.verbose:
    indented_prompt = "\n".join([f"\t{line}" for line in system_prompt_lines])
    log_verbose(f"System prompt:\n{indented_prompt}")

system_prompt = "\n".join(system_prompt_lines) + '\n'


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
# Initialize a query engine
#------------------------------------------------------------------------------

query_engine_params = {
    "response_mode": args.query_mode,
    "show_progress": args.verbose,
    "model_name": args.llm_model or "",
    "api_key": args.llm_api_key,
    "secret": args.llm_secret,
}

if args.llm_provider and args.llm_server:
    log_error(f"cannot specify both --llm-provider and --llm-server", exit_code=1)

try:
    if args.llm_provider:
        query_engine_params["provider"] = args.llm_provider
    elif args.llm_server:
        query_engine_params["server_url"] = args.llm_server
    elif args.llm_model:
        with TimerUntil("model loaded"):
            from transformers import AutoModelForCausalLM
            local_model = AutoModelForCausalLM.from_pretrained(args.llm_model)
            query_engine_params["model"] = local_model

    with TimerUntil("query engine initialized"):        
        query_engine = vector_index.as_query_engine(**query_engine_params)

except Exception as e: log_error(e)    

if not query_engine:
    log_error(f"no query engine available", exit_code=1)


#------------------------------------------------------------------------------
# Run all the queries
#------------------------------------------------------------------------------
    
if len(queries) > 0:
    log(f"Running {len(queries)} queries...")

    before_queries = time.time()
    chat_log = ""
    json_log = {
        "model":        query_engine.model_name,
        "timestamp":    datetime.now().isoformat(),
        "context":      system_prompt,
        "queries":      []
    }

    for query in queries:
        query_record = { "query": query }
        try:
            visible_history = chat_log if args.query_memory else ""
            user_prompt = f"{args.tag_queries}: {query}"
            prompt = system_prompt + visible_history + user_prompt
           
            with TimerUntil("query complete"):
                response = query_engine.query(prompt, verbose=args.verbose)

        except Exception as e:
            query_record["error"] = str(e)
            response = ""
            log_error(e)

        query_record["response"] = response
        json_log["queries"].append(query_record)

        response_text = f"{args.tag_responses}: {response}"
        interaction = f"{user_prompt}\n{response_text}\n"
        chat_log += interaction

        if args.verbose:
            indented_interaction = "\n".join([f"\t{line}" for line in interaction.splitlines()])
            log(indented_interaction)

    # Commit the logs

    if args.output_text:
        log(f"Writing log to \"{args.output_text}\"...")
        try:
            with open(args.output_text, "w", encoding="utf-8") as f:
                f.write(chat_log)
        except Exception as e: log_error(e)

    if args.output_json:
        log(f"Writing JSON log to \"{args.output_json}\"...")
        try:
            with open(args.output_json, "w", encoding="utf-8") as f:
                raw_text = json.dumps(json_log, indent=4)
                f.write(raw_text)
        except Exception as e: log_error(e)

    log(f"Queries completed in {time_since(before_queries)}")
    query_engine.close()


#------------------------------------------------------------------------------
# Chat mode
#------------------------------------------------------------------------------

if args.chat:
    chat_init = args.chat_init or []
    for file in args.chat_init_file or []:
        log_verbose(f"Loading chat context/instructions from \"{file}\"...")
        try:
            with open(file, "r", encoding="utf-8") as f:
                chat_init_text = f.read().strip()
                chat_init.append(chat_init_text)
                log_verbose(chat_init_text)

        except Exception as e:
            log(f"\tERROR: {e}")

    log(f"Entering interactive chat...")
    log(f" - The response mode is \"{args.chat_mode}\"")
    log(f" - Hit CTRL-C to interrupt a response in progress")
    log(f" - Say \"bye\" or something like that when you're done")
    log()
    
    chat_engine_params = {
        "chat_mode": args.chat_mode,
        "system_prompt": f"{system_prompt}\n{chat_init}",
        "verbose": args.verbose, 
    }

    try:
        if local_model:
            chat_engine = vector_index.as_chat_engine(
                model=local_model, 
                **chat_engine_params)
        else:
            chat_engine = vector_index.as_chat_engine(
                server_url=args.llm_server, 
                api_key=args.llm_api_key, 
                **chat_engine_params)
            
    except Exception as e:
        log(f"\tERROR: {e}")

    if not chat_engine:
        log(f"ERROR: no chat engine available")
        exit(1)

    chat_lines = []
    exit_commands = ["bye", "goodbye", "exit", "quit", "done", "stop", "end"]

    while True:
        try:
            message = input("> ")
            if message.strip().lower() in exit_commands:
                break
        except KeyboardInterrupt:
            continue

        chat_lines.append(f"{args.tag_queries}: {message}")
        response_line = ""

        try:
            streaming_response = chat_engine.chat(message, streaming=True)
            for token in streaming_response.response_gen:
                response_line += token
                log(token, end="")

        except KeyboardInterrupt:
            log("[response interrupted]")

        chat_lines.append(f"{args.tag_responses}: {response_line}")
        log()

    if args.chat_log and len(chat_lines) > 0:
        write_action = "Appending" if os.path.exists(args.chat_log) else "Writing"
        log(f"{write_action} this chat log to \"{args.chat_log}\"...")

        try:
            with open(args.chat_log, "a", encoding="utf-8") as f:
                all_lines = "\n".join(chat_lines) 
                f.write(all_lines + "\n")
                
        except Exception as e:
            log(f"\tERROR: {e}")


#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------

if len(all_errors) > 0:
    print(f"Errors reported:")
    for error in all_errors:
        print(f"\t{error}")

log(f"Exiting after {time_since(start_time)}...")
log_verbose("Tiger out, peace")
