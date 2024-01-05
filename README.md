# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

**RAG/TAG Tiger** is a primitive [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that provides a point-and-shoot command line interface for doing basic RAG queries on local documents/code.

It is mostly the same boilerplate/glue you were going to have to write anyway, so I'm posting it hoping it can save somebody an afternoon googling LlamaIndex tutorials and arguing with ChatGPT. If this code improves 
your life, feel free to buy me a coffee!

## Features

- can run queries on a local LLM, an internal inference server, or a commercial endpoint like OpenAI
- loads/updates/stores a vector index to avoid redundant processing
- provides fine grained control over which files to index or be excluded
- optionally honors .gitignore files
- downloads loaders from the [LlamaIndex hub](https://llamahub.ai) to support custom file types
- allows control of LLM inference parameters (temperature, etc)
- assembles system prompts and queries from multiple sources
- supports interactive "chat" mode on the command line

LlamaIndex (and other fine projects) are doing all the actual work here. **RAG/TAG Tiger** is just front-end scaffolding.

## Setup
```
git clone https://stuartriffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```
## Basic usage

```
python ragtag.py --source my/docs --query "But, why?"
```

## Advanced usage

```
python ragtag.py --help
```

## Examples

Here are some contrived examples to give you an idea what more useful workflow looks like. The order of the arguments doesn't matter, and whitespace is ignored.

Ingesting some documents into an existing vector index:

```
python ragtag.py                                      \
    --index-load      my/database                     \
    --custom-loader   PptxReader:pptx                 \
    --source-spec     my/presentations/**/*.pptx      \
    --source-spec     my/code/**/*.(cpp|hpp|h)        \
    --source          my/docs                         \
    --index-store     my/database
```

Running a set of prepared queries on them as a batch process:
```
python ragtag.py                                      \
    --index-load      my/database                     \
    --llm-server      http://localhost:8080           \
    --llm-param       temperature=0.9 seed=1337       \
    --context-file    instructions/three_laws.txt     \
    --query-mode      tree_summarize                  \
    --query-list      queries/simple_questions.txt    \
    --query-file      queries/deep_dive.txt           \
    --query-log       responses.txt                   \
    --query           "how does all this work lol"
```

Chatting with a dangerously underqualified virtual physician:
```
python ragtag.py                                      \
    --index-load      my/medical_database             \
    --llm-model-name  mistralai/Mistral-7B-v0.1       \
    --llm-param       temperature=1.8                 \
    --context-file    first_do_no_harm.txt            \
    --chat-init-file  characters/doctor.txt           \
    --chat-log        it_still_hurts.txt              \
    --chat
```

You can actually do all of that in one command if you really want to: update the vector index, run some queries, then drop into chat mode. I won't type it out, but you get the idea.

## FAQ

**Q:** What does the name mean? <br>
**A:** The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning a model on your documents, the idea is to use a stock model, but give it a way to search your data as needed.

**Q:** What about "TAG"? <br>
**A:** That's a blanket term for tiger-augmented methods.

**Q:** Are those widely used? <br>
**A:** Not in production.

**Q:** But why is there a tiger here at all? <br>
**A:** I tend to anthropomorphize small programs (they like that), but honestly most species can handle RAG queries in a pinch, and the choice of tiger here was arbitrary. We can revisit this.

**Q:** May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:** No.

**Q:** Good enough, how can I buy you that coffee? <br>
**A:** Just to be clear, the coffee is a metaphor, and your donation will be used to buy drugs. Please contribute here: (todo)

