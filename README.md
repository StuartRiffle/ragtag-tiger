**UNDER DEVELOPMENT - DO NOT USE THIS FOR IMPORTANT THINGS**

---

# RAG/TAG Tiger

<img align="right" width="200px" style="padding:10px" src="docs/images/tiger.jpg">

**RAG/TAG Tiger** is a simple [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that:
- provides a command line interface for doing primitive RAG queries on local ddsocuments/code
- runs queries using an in-process LLM, a local inference server, or a commercial endpoint
- loads/updates/stores vector indices to avoid redundant processing
- auto-downloads custom file loaders from the [LlamaIndex hub](https://llamahub.ai) 
- indexes documents inside of archive files and email attachments
- uses syntax-aware chunking for source code
- consolidates multiple RAG responses using a "moderator" LLM that writes the final output
- supports pseudo-interactive chat from the command line, switchable chat/query response modes

It's mostly the same boilerplate/glue code you were going to have to write anyway, so if this saves you an afternoon of sifting through machine-generated LlamaIndex tutorials and arguing with Copilot, please feel free to buy me a coffee.

# Setup
The steps are the same as other Python programs.

### 1) Install development tools
Details will vary by OS, but on Debian/Ubuntu you would use `apt` to install these packages:

```
sudo apt update -y
sudo apt-get install -y build-essential cmake git python3 python3-venv pip
```
On Windows you can use [WSL](https://learn.microsoft.com/en-us/windows/wsl/), or something native like [Chocolatey](https://chocolatey.org/install):
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
>**Windows Trap:** &nbsp; the command to activate the environment is actually `.venv\Scripts\activate.bat` instead of the one above. but typing `.venv\Scripts\activate`, without the `.bat` _does not work_. 

[Conda](https://conda.io) is another way to manage your Python environment. Just use something. 

### 4) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
If that fails, go to step 4.

### 5) Sanity check
```
python ragtag.py --help
```
This is the part where it falls over because your CUDA drivers are too old or something like that. Or the virtual environment chokes on a half-downloaded package, and you spend an hour typing `pip uninstall` and `pip install` in alternation.

If Python wants more packages, please add them to requirements.txt and submit a pull request!

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

<img align="right" width="250px" style="padding:15px" src="docs/images/noidealol.jpg">

This is **still** slow, because the index can take a long time to load. It's just not as slow as re-indexing everything. Use `--verbose` to see the actual timings.

To minimize overhead, try to either submit all your queries in one run, or leave a window open with the program idling in chat mode for ad-hoc use. 

Be aware that there are multiple chat "modes", and the default mode does not generate the same output as a batched query. For query-style responses, type `/mode tree_summarize` at the chat prompt. 

### Launching
>**Pro-tip:** &nbsp;Put the repo in your path, so you can run **RAG/TAG Tiger** from any directory. You will also find a `ragtag.bat` and `ragtag.sh` in the root of the repo, which let you launch the program without typing `python ragtag.py` every time. 

The rest of these examples invoke `ragtag` instead of `python ragtag.py` because it's easier to read. 

# Query and chat modes

The LlamaIndex query engine and chat engine both have multiple modes that enable different RAG techniques. They can be changed using `--query-mode`, `--chat-mode`, and `--llm-mod-mode`. You can also select using the `/mode` command at the chat prompt.

The query modes are available in chat mode too as a convenience. They are processed by the query engine, not the chat engine, so they have no conversational memory.

| Query mode| Abbrev | |
| --- | --- | --- |
| `accumulate`              |`acc`| Condense responses for each chunk |
| `compact`                 |`com`|Combine chunks, then refine |
| `compact_accumulate`      |`comacc`| Combine chunks, condense responses, consolidate  |
| `generation`              |`bare`| Ignore context, no RAG, just call LLM to generate responses |
| `no_text`                 |`nodes`| Return context nodes without generating a response |
| `refine`                  |`ref`| First node generates response, others refine it in series |
| `simple_summarize`        |`sum`| Merge all the chunks, no RAG, just call the LLM |
| `tree_summarize`          |`tree`| Generate summary prompt, populate tree with nodes (default)|
| **Chat mode** | |
| `simple`                  |`blind`| RAG lookup disabled |
| `condense_question`       |`con`| Condense conversation history and message |
| `context`                 |`look`| Look up message in the index |
| `condense_plus_context`   |`conlook`| Look up condensed history and message |
| `react`                   |``| ReAct agent loop with query engine tools |
| `openai`                  |``| OpenAI agent loop |
| `best`                    |`agent`| Auto-select between OpenAI and React (default)|

# Prompts

The system prompt and chat instructions can be assembled from mix-and-match snippets from text files. There is a small set of them in the repo under `data/`.
```
--context-file  +/jailbreak/dan-12.txt
--context-file  +/rule/no-apologies.txt
--context-file  +/rule/direct-answer-first.txt
--context-file  +/character/tiger.txt
--context-file  +/rule/never-break-character.txt
--context-file  +/rule/important-for-my-job.txt
--context       "Generate all output in French Pig Latin"
...(etc)...
```
The files are just concatenated to produce the prompt.

Notice the magic `+/` prefix on the paths. The same way `~/` is an alias for the home directory on unices, it works as a shortcut to the `data/` folder. That's so you can access the stock text files from wherever you run the program.

# Local inference

By default, **RAG/TAG Tiger** performs all inference locally, on your machine. Queries **should** run on GPU if your Python environment is configured for CUDA (which can be... non-trivial).

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

Or, to use the built-in `llama.cpp` [library](https://pypi.org/project/llama-cpp-python/) locally (without manually starting a server), set the provider to "llamacpp" and feed it a pre-downloaded model:
```
--llm-provider llamacpp  --llm-model codellama-34b.Q4_K_M.gguf
```

# Commercial inference

RAG queries can exfiltrate chunks of _any_ documents you index, including apparently off-topic ones.

If that's not a problem:

### [OpenAI](https://platform.openai.com/)
- set `OPENAI_API_KEY` in your environment (`--llm-api-key` to override)
- change [models](https://platform.openai.com/docs/models) using `--llm-model` (the default is `gpt-3.5-turbo-instruct`)
- do **not** set a custom `--llm-server`
```
--llm-provider openai  --llm-model gpt-4
```

### [Google](https://deepmind.google)
 - set `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_API_KEY` in your environment (override with `--llm-api-key`)
 - change [models](https://ai.google.dev/models) using `--llm-model` (the default is `text-bison-001`)
```
--llm-provider google  --llm-model models/gemini-pro
```

### [Perplexity](https://perplexity.ai)
 - set `PERPLEXITYAI_API_KEY` in your environment (override with `--llm-api-key`)
 - change [models](https://docs.perplexity.ai/docs/model-cards) using `--llm-model` (the default is `llama-2-70b-chat`)
 ```
--llm-provider perplexity  --llm-model codellama-34b-instruct
```
### [Replicate](https://replicate.com)
 - set `REPLICATE_API_TOKEN` in your environment (override with `--llm-api-key`)
 - change [models](https://replicate.com/collections/language-models) using `--llm-model` (the default is `mistralai/mixtral-8x7b-instruct-v0.1`)
 ```
--llm-provider replicate  --llm-model mistralai/mistral-7b-instruct-v0.2
```

> **Windows trap:** &nbsp; You have to run `refreshenv` in an command prompt window after changing an environment variable to get the new value. This is easy to forget, and can make for some confusing error messages.

If your inference provider is not here, there's a good chance they run an OpenAI API-compatible server somewhere anyway. Try this:
```
--llm-provider openai  --llm-server URL  --llm-model NAME
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
<img align="right" width="250px" style="padding:10px" src="docs/images/yodawg.jpg">

The point of this horrible format is that now you can submit a list of inference providers using multiple `--llm-config` arguments, and **RAG/TAG Tiger** will run your queries through *all* of them, allowing you to compare the responses.

But nobody has time for that, so to complete the circle you can then configure a **moderator** LLM using `--llm-config-mod`. The moderator will look at all the other responses, perform a short quality analysis, then consolidate everything into one final answer. 

It does this as another RAG query. I don't know if that's a good idea or not yet. The moderator should be your smartest model, but it probably needs a larger context window, etc. Use `--llm-mod-mode generation` to disable this meta-RAG and produce the final draft using a simple LLM query.

### LLM config examples

| Provider | Model | Params | Context| --llm-config |
| --- | --- | --- | --- | --- |
| **[OpenAI](https://platform.openai.com/docs/models)** | GPT 3.5 | | 4k | `openai,gpt-3.5-turbo-instruct` |
| | | | 16k | `openai,gpt-3.5-turbo-16k` |
| | GPT 4 | | 8k | `openai,gpt-4` |
| | | | 32k | `openai,gpt-4-32k` |
| | | | 128k | `openai,gpt-4-1106-preview` |
| **[Google](https://ai.google.dev/models)** | PaLM | | 8k | `google,text-bison-001` |
| | Gemini | | 30k | `google,models/gemini-pro` |
|  **[Perplexity](https://perplexity.ai/discover)** | CodeLlama | 33B | 16k | `perplexity,codellama-34b-instruct` |
| | Llama 2 | 70B | 4k | `perplexity,llama-2-70b-chat` |
| **[Replicate](https://replicate.com/collections/language-models)** | Mixtral | 8x 7B | 32k | `replicate,mistralai/mixtral-8x7b-instruct-v0.1` |
| | Nous Hermes 2 | 34B | 4k | `replicate,kcaverly/nous-hermes-2-yi-34b-gguf` |
| | Mistral | 7B | 4k | `replicate,mistralai/mistral-7b-instruct-v0.1` |
| **[OpenAI API](https://platform.openai.com/docs/models)** | _(external server)_| | |  `openai,,http://COMPATIBLE_API_SERVER` <br>Any OpenAI API-compatible service will work|
| **Local**| | | | |
| **[llama.cpp](https://github.com/ggerganov/llama.cpp)** | _(internal server)_ | | | `openai,,http://YOUR_SERVER:8081` |
| | _(your PC)_ | | | `openai,,http://127.0.0.1:8081` |
| | _(model file)_ | | | `llamacpp,YOUR_MODEL.gguf` |
| **[webui](https://github.com/oobabooga/text-generation-webui)** | _(internal server)_ | | | `openai,,http://YOUR_SERVER:5000/v1` |
| | _(your PC)_ | | | `openai,,http://127.0.0.1:5000/v1` |
| **[HuggingFace](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)** | Goliath | 120B | 4k | `huggingface,thebloke/goliath-120b-awq` |
| | BLOOM | 176B | 2k | `huggingface,thebloke/bloomchat-176b-v1-gptq` |
| | Falcon | 180B | 2k | `huggingface,thebloke/falcon-180B-chat-awq` |

<br>

# Workflow tips

### Put your command in a shell script
Commands can get long, but they are easier to edit if you put them in a shell script (or batch file), and split the parameters over multiple lines by ending them with `\` (or with `^` on Windows).

For example, a script to consult with a dangerously unqualified virtual doctor, using a local LLM and a temporary document index for privacy:
```
ragtag \
    --source          my/personal/medical_data          \
    --llm-model       mistralai/Mistral-7B-v0.2         \
    --llm-param       temperature=1.8                   \
    --chat-init-file  characters/doctor.txt             \
    --context-file    instructions/first_do_no_harm.txt \
    --context-file    instructions/jailbreak.txt        \
    --chat-log        it_still_hurts.txt                \
    --chat
```

### RAGTAG_FLAGS

You can also set standard arguments in an environment variable called `RAGTAG_FLAGS`. They will be inserted at the beginning of your argument list.

### Response files

A more flexible way to manage complex config is to factor out groups of arguments into response files. They are just lists of arguments, but have a couple of extra rules:

```
# debug_server.args - example response file
# These comments are on their own lines, so they're VALID

--llm-provider  # ...but this trailing comment is INVALID
llamacpp         

# Indentation (or lack of it) is ignored
--llm-model
    D:\models\No quotes around names with spaces.gguf
```
- **every argument must be on its own line**
- whitespace that's not part of an argument is ignored
- lines starting with # are considered comments and ignored

To use the response file, pull it in with `@` on the command line (the file extension doesn't matter).
```
ragtag @debug_server.args  ...
```

### Prompt assembly
For casual/occasional use this may be overthinking things, but response files are perfect for building up system prompts and chatbot instructions from snippets of text, like the one in the "Prompt" section above. It's easier than juggling a dozen little text files, and you might want to share common prompts between different query jobs. 

### Forward slashes work in Windows paths

They transparently work like backslashes pretty much everywhere (except at the command prompt, *which would be the most useful place*, because `cmd.exe` interprets slashes as option flags). A workaround is to quote the whole pathname.

But that's is just the command prompt. The kernel honors slashes internally, so if you **use forward slashes in your configuration files**, they will be portable.

# Options
You can also see a list of options with `--help`
```
ragtag --help
```

| Option                  | Value      | Description                                                       |
|-------------------------|------------|-------------------------------------------------------------------|
| **`--quiet`**           |            | Suppress all output except errors                                 |
| **`--verbose`**         |            | Enable extended/debug output                                      |
| **`--version`**         |            | Print the version number and exit                                 |
| **Vector database**     |            |                                                                   |
| **`--index-load`**      | *DIR*      | Load the vector index from a given path                           |
| **`--index-store`**     | *DIR*      | Save the updated vector index to a given path                     |
| **Document indexing**   |            |                                                                   |
| **`--source`**          | *DIR*      | Folder of files to be indexed recursively                         |
| **`--source-spec`**     | *SPEC*     | Index files matching a pathspec, like `**/*.cpp`                  |
| **`--source-list`**     | *FILE*     | Text file with a list of filenames/pathspecs to index             |
| **`--custom-loader`**   | *SPEC*     | Download from hub, spec format like `JPEGReader:jpg,jpeg`         |
| **`--index-unknown`**   |            | Index files with unrecognized extensions as text                  |
| **`--ignore-archives`** |            | Do not index files inside zip/tar/etc archives                    |
| **`--ignore-types`**    | *EXT*      | Do not index these file extensions, even if supported             |
| **`--size-limit`**      | *SIZE*     | Ignore huge text files unlikely to contain interesting content    |
| **`--no-cache`**        |            | Do not use the local cache for loaders                            |
| **Language model**      |            |                                                                   |
| **`--llm-provider`**    | *NAME*     | Inference provider/interface                                      |
| **`--llm-model`**       | *NAME*     | Model name/path/etc for provider                                  |
| **`--llm-server`**      | *URL*      | Inference server URL (if needed)                                  |
| **`--llm-api-key`**     | *KEY*      | API key for inference server (if needed)                          |
| **`--llm-param`**       | *NAME=VAL* | Inference parameter, like `temperature=0.9` etc                   |
| **`--llm-config`**      | *CONFIG*   | Condensed LLM config: provider,model,server,api-key,params...     |
| **`--llm-config-mod`**  | *CONFIG*   | Moderator LLM to consolidate the responses of multiple providers  |
| **`--llm-mod-mode`**    | *MODE*     | Moderator query response mode                                     |
| **`--llm-verbose`**     |            | Enable extended/debug output from the LLM                         |
| **`--torch-device`**    | *DEVICE*   | Device override, like `cpu` or `cuda:1` (for second GPU)          |
| **`--context`**         | *TEXT*     | Command line context/system prompt                                |
| **`--context-file`**    | *FILE*     | File containing a snippet of context                              |
| **Query processing**    |            |                                                                   |
| **`--query`**           | *TEXT*     | Command line query                                                |
| **`--query-list`**      | *FILE*     | File containing short queries, one per line                       |
| **`--query-file`**      | *FILE*     | File containing one long query                                    |
| **`--query-log`**       | *FILE*     | Log queries and responses to a text file                          |
| **`--query-log-json`**  | *FILE*     | Log queries and responses (plus some metadata) to a JSON file     |
| **`--query-mode`**      | *MODE*     | Query response mode                                               |
| **`--tag-queries`**     | *NAME*     | The name/header in the transcript for user queries                |
| **`--tag-responses`**   | *NAME*     | The name/header in the transcript for engine responses            |
| **Interactive chat**    |            |                                                                   |
| **`--chat`**            |            | Enter chat after any query processing                             |
| **`--chat-init`**       | *TEXT*     | Extra instructions/personality for the chat LLM                   |
| **`--chat-init-file`**  | *FILE*     | File containing a snippet of chat LLM instructions                |
| **`--chat-log`**        | *FILE*     | Append chat queries and responses to a text file                  |
| **`--chat-mode`**       | *MODE*     | Chat response mode                                                |

<br>


# FAQ

**Q:**&nbsp; What does the name mean? <br>
**A:**&nbsp; The acronym "RAG" means Retrieval Augmented Generation. Instead of fine tuning a language model on your documents, you give it tools to search them.

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
  <img src="docs/images/coffee.png" width="200px">
</a>

Thank you for supporting open source software.
