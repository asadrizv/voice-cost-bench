OpenAI-compatible chat-completion SSE streams in the documented chunk format.
`stream_with_usage` is what api.openai.com and vLLM send with
`stream_options.include_usage`; `stream_without_usage` exercises the estimation fallback.
