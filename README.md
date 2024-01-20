# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

<img align="right" width="200px" style="padding:10px" src="docs/tiger.jpg">

**RAG/TAG Tiger** is a simple [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that:
- provides a command line interface for doing primitive RAG queries on local documents/code
- runs queries on an in-process LLM, a local inference server, or a commercial endpoint
- loads/updates/stores vector indices to avoid redundant processing
- auto-downloads loaders from the [LlamaIndex hub](https://llamahub.ai) for custom file types
- uses syntax-aware chunking for source code
- supports pseudo-interactive "chat" on the command line

It's mostly the same boilerplate/glue you were going to have to write anyway. If this code saves you an afternoon of sifting through machine-generated LlamaIndex tutorials and arguing with ChatGPT, please feel free to buy me a coffee.

## Setup
```
git clone https://github.com/stuartriffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```
Do this in a [virtual environment](https://www.google.com/search?q=python%20virtual%20environments) if you can.

## Usage

The simplest way to perform a RAG query would be a command like:
```
python ragtag.py --source my/docs --query "But, why?"
```

**But don't do that.** It will index your documents from scratch every time, which is slow. It's better to ingest all your files once, and save the resulting index:
```
python ragtag.py --source my/docs --index-store my/index
```

Then use that index to perform your queries:
```
python ragtag.py --index-load my/index --query "Really though, why?"
```
This is **still** slow, because the index takes a long time to load. It's just not as slow as re-indexing everything. Use `--verbose` to see actual timings.

To minimize overhead, try to either submit all your queries in one run, or leave the program idle in "chat mode" when not in use. 

## Options

A full list is available on the help page:
```
python ragtag.py --help
```

It looks something like this:
```
Options:
  -h, --help             Show this help message and exit
  --quiet                Suppress all output except errors
  --verbose              Enable extended/debug output
  --version              Print the version number and exit
 
Vector database: 
  --index-load PATH      Load the vector index from a given path
  --index-store PATH     Save the updated vector index to a given path
 
Document indexing: 
  --source FOLDER        Folder of files to be indexed recursively
  --source-spec SPEC     Index files matching a pathspec, like "**/*.cpp"
  --source-list FILE     Text file with a list of filenames/pathspecs to index
  --custom-loader SPEC   Download from LlamaIndex hub, format "JPEGReader:jpg,jpeg"
  --no-cache             Do not use the local cache for loaders
  --ignore-unknown       Ignore files with unrecognized extensions
 
Language model: 
  --llm-provider NAME    { openai, anthropic, llamacpp, huggingface }
  --llm-model NAME       Model name/path/etc for provider
  --llm-server URL       Inference server URL (if needed)
  --llm-api-key KEY      API key for inference server (if needed)
  --llm-secret SECRET    Secret for inference server (if needed)
  --llm-param KVP [KVP ...] Inference parameter, "temperature=0.9" etc
  --llm-verbose          Enable extended/debug output from the LLM

Settings:
  --context TEXT         Command line context/system prompt
  --context-file FILE    File containing a snippet of system prompt
  --tag-queries NAME     The name/header in the transcript for user queries
  --tag-responses NAME   The name/header in the transcript for engine responses
  --torch-device DEVICE  Device override, like "cpu" or "cuda:1"

Query processing:
  --query TEXT           Command line query
  --query-list FILE      File containing short queries, one per line
  --query-file FILE      File containing one long query
  --query-log FILE       Log queries and responses to a text file
  --query-log-json FILE  Log queries and responses (plus some metadata) to JSON
  --query-mode           Query response mode, default: tree_summarize
                         { accumulate, compact, compact_accumulate, generation,
                           no_text, refine, simple_summarize, tree_summarize }
Interactive chat:
  --chat                 Enter chat after any query processing
  --chat-init TEXT       Extra instructions/personality for the chat LLM
  --chat-init-file FILE  File containing a snippet of chat LLM instructions
  --chat-log FILE        Append chat queries and responses to a text file
  --chat-mode            Chat response mode, default: best
                         { best, context, condense_question, simple, react, openai }
```

## Workflow

It's easier to edit your command if you put it in a shell script (or batch file), but split the parameters over multiple lines by ending them with `\` (or with `^` on Windows).

For example, a script to consult with a dangerously unqualified virtual doctor, using an in-process LLM and a temporary vector index for privacy, might look like this:
```
python ragtag.py                                        \
    --source          my/personal/medical_data          \
    --llm-model       mistralai/Mistral-7B-v0.2         \
    --llm-param       temperature=1.8                   \
    --context-file    instructions/jailbreak.txt        \
    --context-file    instructions/first_do_no_harm.txt \
    --chat-init-file  characters/doctor.txt             \
    --chat-log        it_still_hurts.txt                \
    --chat
```

A more flexible way to manage complex configuration is to factor out groups of arguments into "response files". The rules there are:
- **every argument must be on its own line**
- blank lines, indentation, and trailing whitespace are ignored
- internal whitespace is part of an argument, so no quotes needed
- lines starting with # are treated as comments
- comments alongside arguments (on the same line) are NOT supported

Like this:
```
# debug_server.args - example response file

--llm-provider
    openai
--llm-server      
    http://localhost:8081             
--llm-param       
    temperature=0.9
    seed=1337         
```

To use the response file, pull it in with `@` on the command line (the file extension doesn't matter). This has the same effect as typing all the arguments by hand:
```
python ragtag.py @debug_server.args  ...
```

For casual/occasional use this may be overthinking things.

## FAQ

**Q:**&nbsp; What does the name mean? <br>
**A:**&nbsp; The acronym "RAG" means Retrieval Augmented Generation. Instead of fine tuning a language model on your documents, you give it tools to search them for any details needed.

**Q:**&nbsp; What about "TAG"? <br>
**A:**&nbsp; That's a blanket term for tiger-augmented methods.

**Q:**&nbsp; Are those widely used? <br>
**A:**&nbsp; Not in production.

**Q:**&nbsp; But why is there a tiger here at all? <br>
**A:**&nbsp; I anthropomorphize small programs (they enjoy that), but to be fair a lot of species could do RAG in a pinch, and my choice of tiger here was arbitrary. We can revisit this.

**Q:**&nbsp; May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:**&nbsp; No.

**Q:**&nbsp; Good enough, how can I buy you that coffee? <br>
**A:**&nbsp; For clarity, the coffee was a metaphor and all contributions will be spent on drugs. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.buymeacoffee.com/stuartriffle">
  <img src="docs/coffee.png" width="200px">
</a>

Thank you for supporting open source software.






