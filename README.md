# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

**RAG/TAG Tiger** is a [LlamaIndex](https://github.com/run-llama/llama_index) wrapper that provides a point-and-shoot command line interface for doing RAG queries on local documents/code.

It is mostly the same boilerplate/glue you were going to have to write anyway, so I'm posting it hoping it can save somebody an afternoon googling LlamaIndex tutorials and arguing with ChatGPT. If this code improves 
your life, feel free to buy me a coffee!

## Features
- can run queries on a local LLM, an internal inference server, or a commercial endpoint like OpenAI
- loads/updates/stores a vector index to avoid redundant processing
- provides fine grained control over which files to index or be excluded, supports .gitignore
- downloads loaders from the [LlamaIndex hub](https://llamahub.ai) to support custom file types
- allows basic control of LLM inference parameters (temperature, etc)
- assembles system prompts and queries from multiple sources
- supports interactive "chat" mode on the command line

LlamaIndex (and many other fine projects) are doing all the actual work here. **RAG/TAG Tiger** is just front-end scaffolding. 

## Setup
```
git clone https://github.com/StuartRiffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```
## Basic use
```
python ragtag.py --source my/docs --query "But, why?"
```

## Scripted workflow

For simple one-off queries, the command line works fine, but editing anything there is painful.

A better way to iterate is to put your command into a little shell script or batch file, but split over multiple lines, so you can see all the arguments.

> These examples have a `\` at the end of every line because in a Mac or Linux shell script that means to combine it with the next line, so they all chain up into one big long command. Windows batch files use `^` for the same purpose. You don't have to line everything up in pretty columns like I did.

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

Consulting a dangerously unqualified virtual physician, using a temporary (in-memory) vector index for privacy:
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

You could actually do all of that in one command, if you wanted to: update a vector index, run some queries against it, then drop into chat mode to follow up on them.

More settings are available. Check out the help page for a complete list:

```
python ragtag.py --help
```

## Advanced workflow

If things start to get complicated, the best way to manage configuration is to put groups of arguments into "response files" like this:
```
# debug_server.args - example response file
#
# The rules here:
#   - every argument must be on its own line   <------ THE MAIN THING TO KNOW!
#   - the file extension does not matter
#   - blank lines, indentation, and trailing whitespace are ignored
#   - internal whitespace is considered part of the argument, so no quotes needed
#   - lines starting with the character # are treated as comments, and skipped
#   - putting comments alongside arguments, on the same line, is NOT supported 

--llm-server      
	http://localhost:8080             
--llm-param       
    temperature=0.9
	seed=1337         
    ...
```

Then pull those files in using `@` on the command line, like building blocks: 
```
python ragtag.py @debug_server.args @standard_queries.args  ...
```

You now have the one true power - **indirection**. You can edit blocks of common settings in one place instead of needing (for example) to update all your scripts when you change inference providers.

## FAQ

**Q:** What does the name mean? <br>
**A:** The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning a model on your documents, the idea is to use a stock model, but give it ways to search your data as needed.

**Q:** What about "TAG"? <br>
**A:** That's a blanket term for tiger-augmented methods.

**Q:** Are those widely used? <br>
**A:** Not in production.

**Q:** But why is there a tiger here at all? <br>
**A:** I tend to anthropomorphize small programs (they like that), but honestly a lot of species can handle RAG queries in a pinch, and my choice of tiger here was arbitrary. We can revisit this.

**Q:** May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:** No.

**Q:** Good enough, how can I buy you that coffee? <br>
**A:** Just to be clear, the coffee was a metaphor and all donations will be used to buy drugs. Please contribute here: (todo)

