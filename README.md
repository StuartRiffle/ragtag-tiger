# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

**RAG/TAG Tiger** is a [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that provides a point-and-shoot command line interface for doing RAG queries on local documents/code with an LLM.

I had modest RAG needs, but could not find an equally simple tool to use. I just wanted to RAG some files I had, not build a whole damned data ingestion pipeline. I wrote this script to connect the dots as crudely as possible. It has gained some convenience features since then, for which I apologize. I am trying to stop. 

A lot of this code is the same boilerplate/glue you were going to have to write anyway. If it saves you an afternoon of sifting through machine-generated LlamaIndex tutorials and arguing with ChatGPT, feel free to buy me a coffee!

## Features
- runs queries on a local LLM, an internal inference server, or a commercial endpoint like OpenAI
- loads/updates/stores vector indices to avoid redundant processing
- provides fine grained control over which files to index or be excluded, supports .gitignore
- downloads loaders from the [LlamaIndex hub](https://llamahub.ai) to process custom file types
- allows basic control of LLM inference parameters (temperature, etc)
- assembles system prompts and queries from multiple sources 
- supports interactive "chat" mode on the command line

Let's be clear that LlamaIndex (and other amazing open source projects) are doing all the actual work here. **RAG/TAG Tiger** is a thin front end for that functionality. 

## Setup
```
git clone https://github.com/StuartRiffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```
## Basic usage
```
python ragtag.py --source my/docs --query "But, why?"
```

## Workflow

For simple queries, the command line is fine, but editing there is painful.

A better way to iterate is to put your command into a little shell script or batch file, but split it over multiple lines, so you can see all the arguments at a glance. 

> These examples have a `\` at the end of every line because in a Mac or Linux shell script that means to combine it with the next line, so they all chain up into one big long command. Windows batch files use `^` for that.

A script ingesting some documents into an existing vector index might look like this:
```
python ragtag.py                                        \
    --index-load      my/database                       \
    --custom-loader   PptxReader:pptx                   \
    --source-spec     my/presentations/**/*.pptx        \
    --source-spec     my/code/**/*.(cpp|hpp|h)          \
    --source          my/docs                           \
    --index-store     my/database
```

Running a set of prepared queries on them as a batch process:
```
python ragtag.py                                        \
    --index-load      my/database                       \
    --llm-server      http://localhost:8080             \
    --llm-param       temperature=0.9 seed=1337         \
    --context-file    instructions/three_laws.txt       \
    --query-mode      tree_summarize                    \
    --query-list      queries/simple_questions.txt      \
    --query-file      queries/deep_dive.txt             \
    --query-log       responses.txt                     \
    --query           "how does all this work lol"
```

Consulting a dangerously unqualified virtual doctor, using an in-process LLM and a temporary (memory-only) vector index for privacy:
```
python ragtag.py                                        \
    --source          my/personal/medical_data          \
    --llm-model-name  mistralai/Mistral-7B-v0.1         \
    --llm-param       temperature=1.8                   \
    --context-file    instructions/jailbreak.txt        \
    --context-file    instructions/first_do_no_harm.txt \
    --chat-init-file  characters/doctor.txt             \
    --chat-log        it_still_hurts.txt                \
    --chat
```

It's possible to do all of those things in one command too: update a vector index, run some queries against it, then drop into chat mode to follow up.

A few more options are available. Check the help page for a complete list:

```
python ragtag.py --help
```

## Advanced workflow
See, I was trying to avoid this kind of thing, but here we are.

The best way to manage complex configuration is to factor out groups of arguments into "response files" like so:
```
# debug_server.args - example response file
#
# The rules here:
#   - every argument must be on its own line   <------ THE MAIN THING TO KNOW!
#   - the file extension doesn't matter
#   - blank lines, indentation, and trailing whitespace are ignored
#   - internal whitespace is considered part of an argument, no quotes needed
#   - lines starting with the character # are treated as comments, and ignored
#   - comments alongside arguments, on the same line, are NOT supported 

--llm-server      
    http://localhost:8081             
--llm-param       
    temperature=0.9
    seed=1337         
    ...
```

Then pull those arguments in using `@` on the command line:
```
python ragtag.py @debug_server.args @integration_database.args  ...
```
Scripts are shorter and a lot easier to understand with the off-topic noise removed:
```
python ragtag.py                        \
    @model_internal_7b.args             \
    @ersatz_doctor.args                 \
    --source my/personal/medical_data   \
    --chat
```

Now you can edit blocks of arguments in one response file instead of updating all your scripts every time you change servers or something.

For casual/occasional use this is probably overthinking things.

## FAQ

**Q:** What does the name mean? <br>
**A:** The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning a model on your documents, the idea is to use a stock model, but give it ways to search them for details as needed.

**Q:** What about "TAG"? <br>
**A:** That's a blanket term for tiger-augmented methods.

**Q:** Are those in wide use? <br>
**A:** Not in production.

**Q:** But why is there a tiger here at all? <br>
**A:** I anthropomorphize small programs because they like that. But to be fair a lot of species can do RAG queries in a pinch, and my choice of tiger here was arbitrary. We can revisit this.

**Q:** May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:** No.

**Q:** Good enough, how can I buy you that coffee? <br>
**A:** Just to be clear, the coffee is a metaphor and donations will be used to buy drugs.



