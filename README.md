# RAG/TAG Tiger

**UNDER DEVELOPMENT - STILL DEBUGGING - DO NOT USE THIS**

This is a LlamaIndex wrapper that provides a point-and-shoot command line interface for doing RAG queries on local documents/code.

It is mostly the same boilerplate/glue you were going to have to write anyway, so I'm posting it with the hope it can save somebody an afternoon googling LlamaIndex tutorials and arguing with ChatGPT. If this code improves your life, feel free to buy me a coffee!

## Setup
```
git clone http://stuartriffle/ragtag-tiger
cd ragtag-tiger
pip install -r requirements.txt
```

## Basic usage

`python ragtag.py --source-folder /my/docs --query "But, why?"`

## Advanced usage

`python ragtag.py --help`

Set the corresponding options, flip some switches, and this program:
- can run queries on a local LLM, an internal inference server, or a commercial endpoint like OpenAI
- loads/updates/stores a vector index to avoid redundant processing
- provides fine grained control over which files to index or be excluded
- optionally honors .gitignore files
- downloads loaders from the LlamaIndex hub to support custom file types
- allows control of LLM inference parameters (temperature, etc)
- assembles system prompts and queries from multiple sources
- supports interactive "chat" on the command line

LlamaIndex (and other fine libraries) are doing all the actual work here. `RAG/TAG Tiger` is front-end scaffolding.

## FAQ

**Q:** What does the name mean? <br>
**A:** The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning language model weights on your documents, the idea is to use a stock model, but give it a way to search your data for details.

**Q:** What about "TAG"? <br>
**A:** That's a blanket term for tiger-augmented methods.

**Q:** Are those common? <br>
**A:** Not in production.

**Q:** But why is there a tiger here at all? <br>
**A:** I tend to anthropomorphize small programs (they like that), but honestly most species can handle RAG queries in a pinch, and the choice of tiger here was arbitrary. We can revisit this.

**Q:** May I have a warranty of merchantability and fitness for my particular purpose? <br>
**A:** No.

**Q:** Good enough, how can I buy you that coffee? <br>
**A:** The coffee was a metaphor and all donations will be used to buy drugs. Please contribute here: (todo)

