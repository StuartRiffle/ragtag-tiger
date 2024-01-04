
# RAG/TAG Tiger

This is a LlamaIndex wrapper that provides a point-and-shoot command line interface for doing RAG queries on local documents/code.

It is mostly the same boilerplate/glue you were going to have to write anyway, so I'm posting it with the hope it can save somebody an afternoon googling LlamaIndex tutorials and arguing with ChatGPT. If this code improves your life, feel free to buy me a coffee!

## Basic usage

`python ragtag.py --source-folder /my/docs --query "But, why?"`

## Advanced usage

`python ragtag.py --help`

- runs queries on a local LLM, an internal inference server, or a commercial endpoint
- loads/updates/stores the vector index to avoid redundant processing
- provides fine grained control over which files to index or be excluded
- downloads loaders from the LlamaIndex hub to support custom file types
- allows tuning of LLM inference parameters (temperature, etc)
- combines multiple sources to build system prompts and queries
- supports interactive chat on the command line

## FAQ

Q: What does the name mean?
A: The term "RAG" means Retrieval Augmented Generation. Instead of fine tuning language model weights to remember the details of your documents, the idea is to use a stock language model, but give it a way to search your data for details on the fly.

Q: What about "TAG"?
A: That's a blanket term for tiger-augmented methods.

Q: Are those common?
A: Not in production.

Q: But why is there a tiger here at all?
A: I tend to anthropomorphize small programs (they like that), but honestly most species can handle RAG queries in a pinch, and the choice of tiger here was arbitrary. We can revisit this.

Q: May I have a warranty of merchantability and fitness for my particular purpose?
A: No.

Q: Good enough, how can I buy you that coffee?
A: The coffee was a metaphor and your donation will be used to buy drugs. Please contribute here:

