import logging, files
from files import *
from timer import TimerUntil

def load_llm(provider, model, server, api_key, params, set_service_context=True):
    result = None
    streaming_supported = True
    try:
        with TimerUntil("ready"):
            model_kwargs = dict([param.split("=") for param in params]) if params else {}

            ### OpenAI
            if provider == "openai":
                api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
                if not server:
                    model_name = model or openai_model_default
                    log(f"Preparing OpenAI model \"{model_name}\"...")
                    from llama_index.llms import OpenAI
                    result = OpenAI(
                        model=model_name,
                        timeout=default_timeout,
                        api_key=api_key,
                        additional_kwargs=model_kwargs,
                        verbose=args.llm_verbose)
                else:
                    # API compatible server
                    model_name = model or "default"
                    log(f"Preparing model \"{model_name}\" on server \"{server}\"...")
                    from llama_index.llms import OpenAILike
                    result = OpenAILike(
                        model=model_name,
                        additional_kwargs=model_kwargs,
                        api_base=server,
                        max_tokens=1000,
                        max_iterations=100,
                        timeout=default_timeout,
                        verbose=args.llm_verbose)
                
            ### Google
            elif provider == "google":
                api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
                model_name = model or google_model_default
                log(f"Preparing Google model \"{model_name}\"...")
                from llama_index.llms import PaLM
                result = PaLM(
                    api_key=api_key,
                    model_name=model_name,
                    generate_kwargs=model_kwargs)
                streaming_supported = False
                
            ### Llama.cpp
            elif provider == "llamacpp":
                if torch.cuda.is_available():
                    # FIXME - this does nothing
                    model_kwargs["n_gpu_layers"] = -1
                    model_kwargs["device"] = "cuda"
                log(f"Preparing llama.cpp model \"{os.path.normpath(model)}\"...")
                from llama_index.llms import LlamaCPP
                result = LlamaCPP(
                    model_path=model,
                    model_kwargs=model_kwargs,
                    verbose=args.llm_verbose)
                
            ### Perplexity
            elif provider == "perplexity":
                api_key = api_key or os.environ.get("PERPLEXITYAI_API_KEY", "")
                model_name = model or perplexity_default
                log(f"Preparing Perplexity model \"{model_name}\"...")
                from llama_index.llms import Perplexity
                result = Perplexity(
                    api_key=api_key,
                    model=model_name,
                    model_kwargs=model_kwargs)
                
            ### Replicate
            elif provider == "replicate":
                api_key = api_key or os.environ.get("REPLICATE_API_TOKEN", "")
                model_name = model or replicate_default
                log(f"Preparing Replicate model \"model_name)\"...")
                from llama_index.llms import Replicate
                result = Replicate(
                    model=model_name,
                    additional_kwargs=model_kwargs)
            
            ### HuggingFace
            else:
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                model_desc = ""
                model_name = model or "default"
                if model_name in hf_model_nicknames:
                    model_desc = f" (\"{model_name}\")"
                    model_name = hf_model_nicknames[model_name]
                log(f"Preparing HuggingFace model \"{model_name}\"{model_desc}...")
                from llama_index.llms import HuggingFaceLLM
                result = HuggingFaceLLM(
                    model_name=model_name,
                    model_kwargs=model_kwargs, 
                    device_map=args.torch_device or "auto",

                    system_prompt=system_prompt)

            if set_service_context:
                from llama_index import ServiceContext
                from llama_index import set_global_service_context

                service_context = ServiceContext.from_defaults(embed_model="local", llm=result)
                set_global_service_context(service_context)

    except Exception as e: 
        log_error(f"failure initializing LLM: {e}", exit_code=1)

    return result, streaming_supported

def split_llm_config(config):
    """Split an LLM from a config string like "provider,model,server,api-key,param1,param2,..." into its components"""
    config   = config.strip("\"' ")
    fields   = config.split(",")
    provider = fields[0].strip() if len(fields) > 0 else default_llm_provider
    model    = fields[1].strip() if len(fields) > 1 else None
    server   = fields[2].strip() if len(fields) > 2 else None
    api_key  = fields[3].strip() if len(fields) > 3 else None
    params   = fields[4:]        if len(fields) > 4 else []
    return provider, model, server, api_key, params

def load_llm_config(config, set_service_context=True):
    """Load an LLM from a config string like "provider,model,server,api-key,param1,param2,..."""
    provider, model, server, api_key, params = split_llm_config(config)
    return load_llm(provider.lower(), model, server, api_key, params, set_service_context)


