**UNDER DEVELOPMENT - DO NOT USE THIS FOR IMPORTANT THINGS**

# RAG/TAG Tiger

<img align="right" width="200px" style="padding:10px" src="docs/tiger.jpg">

**RAG/TAG Tiger** is a simple [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that:
- provides a command line interface for doing primitive RAG queries on local documents/code
- runs queries using an in-process LLM, a local inference server, or a commercial endpoint
- loads/updates/stores vector indices to avoid redundant processing
- auto-downloads custom file loaders from the [LlamaIndex hub](https://llamahub.ai) 
- indexes documents inside archive files and email attachments
- uses language-aware chunking for source code
- supports pseudo-interactive "chat" from the command line, using any chat/query response mode

It's mostly the same boilerplate/glue code you were going to have to write anyway, so if this saves you an afternoon of sifting through machine-generated LlamaIndex tutorials and arguing with Copilot, please feel free to buy me a coffee.

# Setup
The steps are the same as other Python programs.

### 1) Install development tools
Details will vary by OS, but on [Debian]()/[Ubuntu]() you would use `apt` to install these packages:

```
sudo apt update -y
sudo apt-get install -y build-essential cmake git python3 pip python3-venv
```
On Windows you could use [WSL](https://learn.microsoft.com/en-us/windows/wsl/), or use something like [Chocolatey](https://chocolatey.org/install):
```
choco install /y python git cuda
```

### 2) Clone this repository
```
git clone https://github.com/stuartriffle/ragtag-tiger ragtag
cd ragtag
```
### 3) Create and activate a [virtual environment](https://docs.python.org/3/library/venv.html)
```
python3 -m venv .venv
. .venv/bin/activate
```
On Windows, activate the environment with `.venv\Scripts\activate.bat`

You can also use [conda](https://conda.io) to manage the Python environment.

### 4) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
If that fails, go to step 4.

### 5) Sanity check
```
python ragtag.py --version
```
If that does not print some kind of Python error message, we're in a good place. 

If it wants more packages, please add them to requirements.txt and submit a pull request!

# Usage

The simplest way to perform a RAG query would be:
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
This is **still** slow, because the index can take a long time to load. It's just not as slow as re-indexing everything. Use `--verbose` to see actual timings.

To minimize overhead, try to either submit all your queries in one run, or leave a window open with the program idling in "chat mode" for ad-hoc use. Be aware that there are multiple chat modes, and the default does not respond the same way as a query. Use the `/mode` command in chat to switch response types. 

For query-style responses in chat mode, type `/mode tree_summarize` at the chat prompt.

# Local inference

By default, **RAG/TAG Tiger** performs all inference locally, on your machine. Queries **should** run on GPU if your Python environment is configured for CUDA (which can be non-trivial).

### [HuggingFace](https://huggingface.co/models)
This is the default provider, so `--llm-provider huggingface` isn't strictly required. Use `--llm-model` to select specific models by name. They will be downloaded and cached for re-use. For example:
```
--llm-provider huggingface  --llm-model TheBloke/CodeLlama-34B-Instruct-AWQ
```

### [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
If you enable the "API" and "listen" checkboxes on the extensions page of the web UI, it will run an OpenAI-compatible inference server on port `5000`. Connect to it like this:
```
--llm-provider openai  --llm-server http://localhost:5000/v1
```

### [llama.cpp](https://github.com/ggerganov/llama.cpp)
If you run your own `llama.cpp` server, also run `examples/server/api_like_OAI.py` to enable the OpenAI compatibility layer, then:
```
--llm-provider openai  --llm-server http://YOUR_SERVER:8081
```

Or, to use the built-in `llama.cpp` [library](https://pypi.org/project/llama-cpp-python/) locally (without manually starting a server), set the provider to "llamacpp" and supply a pre-downloaded model:
```
--llm-provider llamacpp  --llm-model codellama-34b.Q4_K_M.gguf
```

# Commercial inference

RAG queries can exfiltrate chunks of _any_ documents you index, including apparently off-topic ones.

If that's not a problem:

### [OpenAI](https://platform.openai.com/)
- authenticate by setting `OPENAI_API_KEY` in your environment (or override with `--llm-api-key`)
- change [models](https://platform.openai.com/docs/models) using `--llm-model` (the default is `gpt-3.5-turbo-instruct`)
- do **not** set a custom `--llm-server`
```
--llm-provider openai  --llm-model gpt-4
```

### Google
 - authenticate by setting `GOOGLE_APPLICATION_CREDENTIALS` and `GEMINI_API_KEY` in your environment (or override with `--llm-api-key`)
 - change [models](https://platform.openai.com/docs/models) using `--llm-model` (the default is `gpt-3.5-turbo-instruct`)
 ```
--llm-provider google  --llm-model models/text-bison-001
```

# RAG gauntlet

There's another, more compact way to configure inference providers, which is `--llm-config`
```
provider[,model[,server[,api-key[,parameters...]]]]
```

Basically, you just glue all the settings together with commas.

Most of the time, all you need is the first couple of fields, and you can skip unused fields by leaving them empty. For a complicated example, to connect with a local (OpenAI API-compatible) [text-generation-webui](https://github.com/oobabooga/text-generation-webui) server, using a couple of custom inference parameters:

```
--llm-config openai,,http://localhost:5000/v1,,temperature=1.6,top_p=0.9
```

The benefit of this format is that now you can submit a *list* of inference providers with multiple `--llm-config` arguments, and **RAG/TAG Tiger** will run your queries through *all* of them, allowing you to compare the responses.

But nobody has time for that, so to complete the circle you can configure a **moderator** LLM using `--llm-config-mod`. The moderator will look at all the other responses, perform a short quality analysis, then consolidate everything into one final answer.

### LLM config examples

| | Model | Params | Context| --llm-config |
| --- | --- | --- | --- | --- |
| **OpenAI** | GPT 3.5 | | 4k | `openai,gpt-3.5-turbo-instruct` |
| | | | 16k | `openai,gpt-3.5-turbo-16k` |
| | GPT 4 | | 8k | `openai,gpt-4` |
| | | | 32k | `openai,gpt-4-32k` |
| | | | 128k | `openai,gpt-4-1106-preview` |
| **Google** | PaLM | | 8k | `google,text-bison-001` |
| | Gemini | | 30k | `google,models/gemini-pro` |
|  **Perplexity** | CodeLlama | 33B | 16k | `perplexity,codellama-34b-instruct` |
| | Llama 2 | 70B | 4k | `perplexity,llama-2-70b-chat` |
| **Replicate** | Mixtral | 8x 7B | 32k | `replicate,mistralai/mixtral-8x7b-instruct-v0.1` |
| | Nous Hermes 2 | 34B | 4k | `kcaverly/nous-hermes-2-yi-34b-gguf` |
| **HuggingFace** | Goliath | 120B | 4k | `huggingface,thebloke/goliath-120b-awq` |
| | BLOOM | 176B | 2k | `huggingface,thebloke/bloomchat-176b-v1-gptq` |
| | Falcon | 180B | 2k | `huggingface,thebloke/falcon-180B-chat-awq` |
| **text_generation_gui** | _(server)_ | | | `openai,,http://YOUR_SERVER:5000/v1` |
| | _(local)_ | | | `openai,,http://127.0.0.1:5000/v1` |
| **llama.cpp** | _(server)_ | | | `openai,,http://YOUR_SERVER:8081` |
| | _(local)_ | | | `openai,,http://127.0.0.1:8081` |
| | _(local file)_ | | | `llamacpp,YOUR_MODEL.gguf` |


# Options

A list of options is available on the help page:
```
python ragtag.py --help
```

It looks like this:
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
  --custom-loader SPEC   Download from hub, i.e. "JPEGReader:jpg,jpeg"
  --no-cache             Do not use the local cache for loaders
  --ignore-unknown       Ignore files with unrecognized extensions
  --index-archives       Index files inside .zip and other archives
 
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

# Workflow

Commands can get long, but they are easier to edit if you put them in a shell script (or batch file), and split the parameters over multiple lines by ending them with `\` (or with `^` on Windows).

For example, a script to consult with a dangerously unqualified virtual doctor, using a local LLM and a temporary document index for privacy:
```
python ragtag.py                                        \
    --source          my/personal/medical_data          \
    --llm-model       mistralai/Mistral-7B-v0.2         \
    --llm-param       temperature=1.8                   \
    --chat-init-file  characters/doctor.txt             \
    --context-file    instructions/first_do_no_harm.txt \
    --context-file    instructions/jailbreak.txt        \
    --chat-log        it_still_hurts.txt                \
    --chat
```

You can also set standard arguments in an environment variable called `RAGTAG_FLAGS`. They will be added to the end of every command.

A more flexible way to manage complex configuration is to factor out groups of arguments into response files, where...
- **every argument must be on its own line**
- blank lines, indentation, and trailing whitespace are ignored
- internal whitespace is part of an argument, no quotes needed
- lines starting with # are considered comments and ignored
- comments on the same line as arguments are NOT supported
```
# debug_server.args - example response file

--llm-provider
    openai
--llm-server      
    http://localhost:8081             
--llm-param       
    temperature=0.9
```

To use the response file, pull it in with `@` on the command line (the file extension doesn't matter). This has the same effect as typing all the arguments by hand:
```
python ragtag.py @debug_server.args  ...
```

For casual/occasional use this may be overthinking things.

# FAQ

**Q:**&nbsp; What does the name mean? <br>
**A:**&nbsp; The acronym "RAG" means Retrieval Augmented Generation. Instead of fine tuning a language model on your documents, you give it tools to search them for any details needed.

**Q:**&nbsp; What about "TAG"? <br>
**A:**&nbsp; That's a blanket term for tiger-augmented methods.

**Q:**&nbsp; Are those widely used? <br>
**A:**&nbsp; Not in production.

**Q:**&nbsp; But why is there a tiger here at all? <br>
**A:**&nbsp; I anthropomorphize small programs (they enjoy that), but honestly a lot of species could do RAG in a pinch, and my choice of tiger here was arbitrary. We can revisit this.

**Q:**&nbsp; May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:**&nbsp; No.

**Q:**&nbsp; Good enough, how can I buy you that coffee? <br>
**A:**&nbsp; For clarity, the coffee is a metaphor and contributions will be spent on drugs. 

<a href="https://www.buymeacoffee.com/stuartriffle">
  <img src="docs/coffee.png" width="200px">
</a>

Thank you for supporting open source software.
