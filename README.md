# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

<img align="right" width="200px" style="padding:10px" src="docs/tiger.jpg">

**RAG/TAG Tiger** is a [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that provides a command line interface for doing primitive RAG queries on local documents/code.
- runs queries on an in-process LLM, an internal inference server, or a commercial endpoint
- loads/updates/stores vector indices to avoid redundant processing
- provides fine grained control over which files to index or be excluded
- auto-downloads loaders from the [LlamaIndex hub](https://llamahub.ai) to process custom file types
- allows basic control of LLM inference parameters (temperature, etc)
- can tack together system prompts and queries from multiple sources
- supports interactive "chat" mode on the command line


It's mostly the same boilerplate/glue you were going to have to write anyway. If this code saves you an afternoon of sifting through machine-generated LlamaIndex tutorials and arguing with ChatGPT, please feel free to buy me a coffee.




## Setup
```
git clone https://github.com/stuartriffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```
## Basic usage
The simplest way to perform a RAG query would be a command like this:
```
python ragtag.py --source my/docs --query "But, why?"
```
**But don't do that.** It will index your documents from scratch on every query, which is slow. It's better to ingest all your files one time, and save the resulting index:
```
python ragtag.py --source my/docs --index-store my/index
```
Then use that index to perform your queries:
```
python ragtag.py --index-load my/index --query "Really though, why?"
```
Check out the help page for a complete list of options:

```
python ragtag.py --help
```

## Workflow

A good way to build and iterate on a command is to put it in a shell script or batch file, but split over multiple lines, so you can see all the arguments at a glance (in a Windows batch file use `^` to join lines instead of `\`).

A script ingesting some presentations into an existing vector index might look like this:
```
python ragtag.py                                        \
    --index-load      my/database                       \
    --source          my/docs                           \
    --source-spec     my/presentations/**/*.pptx        \
    --custom-loader   PptxReader:pptx                   \
    --index-store     my/database
```

Running a set of prepared queries on them using a local inference server that exposes an OpenAI-compatible API:
```
python ragtag.py                                        \
    --index-load      my/database                       \
    --llm-provider    openai                            \
    --llm-server      http://localhost:8081             \
    --query-mode      tree_summarize                    \
    --query-list      queries/simple_questions.txt      \
    --query-log       responses.txt
```

Consulting a dangerously unqualified virtual doctor, using an in-process LLM and a temporary (memory-only) vector index for privacy:
```
python ragtag.py                                        \
    --source          my/personal/medical_data          \
    --llm-model       mistralai/Mistral-7B-v0.1         \
    --llm-param       temperature=1.8                   \
    --context-file    instructions/jailbreak.txt        \
    --context-file    instructions/first_do_no_harm.txt \
    --chat-init-file  characters/doctor.txt             \
    --chat-log        it_still_hurts.txt                \
    --chat
```

## Advanced workflow
A fancier way to manage complex configuration is to factor out groups of command line arguments into "response files" like this:
```
# debug_server.args - example response file
#
# The rules here:
#   - every argument must be on its own line  <------ IMPORTANT!
#   - the file extension doesn't matter
#   - blank lines, indentation, and trailing whitespace are ignored
#   - internal whitespace is part of an argument, no quotes needed
#   - lines starting with # are treated as comments
#   - comments next to arguments (on the same line) are NOT supported 

--llm-provider
    openai
--llm-server      
    http://localhost:8081             
--llm-param       
    temperature=0.9
    seed=1337         
    (etc...)
```

Then pull in the response files using `@` on the command line. This has the same effect as typing all those arguments by hand:
```
python ragtag.py @debug_server.args  ...
```
Now you can edit blocks of arguments in response files, instead of updating all your scripts every time you change servers or something.
```
python ragtag.py                        \
    @model_internal_7b.args             \
    @not_a_doctor.args                  \
    --source my/personal/medical_data   \
    --chat
```

For casual/occasional use this may be overthinking things.

## FAQ

**Q:** What does the name mean? <br>
**A:** The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning a language model on your documents, the idea is you give it ways to search them for details as needed.

**Q:** What about "TAG"? <br>
**A:** That's a blanket term for tiger-augmented methods.

**Q:** Are those widely used? <br>
**A:** Not in production.

**Q:** But why is there a tiger here at all? <br>
**A:** I anthropomorphize small programs because they like that, but to be fair a lot of species can handle RAG in a pinch, and my choice of tiger here was arbitrary. We can revisit this.

**Q:** May I have a warranty of merchantability and fitness for my purpose? <br>
**A:** No.

**Q:** Good enough, how can I buy you that coffee? <br>
**A:** For clarity, the coffee is a metaphor and any contributions will be spent on drugs. 


<a href="https://www.buymeacoffee.com/stuartriffle"><img src="docs/coffee.png" width="200px"></a>

Thank you for supporting open source software.






